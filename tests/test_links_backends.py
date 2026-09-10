"""The link store's contract, run against every backend it claims to have.

One suite, parametrized. That is the whole guard: ADR-2's pair drifted
because each implementation had its own tests and its own semantics, and
nothing forced them to agree. Here the queries are written once (see
`_Dialect`) and asserted once, so "parity" is not a discipline anybody
has to keep.

Postgres tests skip themselves when no database answers, the same way
the tree-sitter tests skip a grammar that is not installed. `docker
compose up -d postgres` is what makes them run.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from wsindex.links import Link, LinkKind, LinkStore

DSN_ENV = "WSINDEX_TEST_POSTGRES_DSN"
"""Where a test Postgres is named. Defaults to what `docker compose`
brings up, so nobody has to set anything to run them."""

DEFAULT_DSN = "postgresql://wsindex:wsindex@localhost:5432/wsindex"


def postgres_dsn() -> str:
    return os.environ.get(DSN_ENV, DEFAULT_DSN)


def postgres_is_up() -> bool:
    """True when a Postgres answers, so the tests can run rather than skip."""
    try:
        import psycopg
    except ImportError:
        return False
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=2):
            return True
    except Exception:
        return False


needs_postgres = pytest.mark.skipif(
    not postgres_is_up(),
    reason=f"needs a Postgres at {postgres_dsn()} (docker compose up -d postgres)",
)


@pytest.fixture
def sqlite_store(tmp_path: Path) -> Iterator[LinkStore]:
    with LinkStore(tmp_path / "idx") as store:
        yield store


@pytest.fixture
def postgres_store() -> Iterator[LinkStore]:
    # A fresh table per test, and an empty one afterwards. Unlike a
    # tmp_path the database outlives the run, so isolation has to be
    # asked for at both ends — rows left behind turned up in a manual
    # `wsindex refs` half an hour later and read as real links.
    _reset()
    store = LinkStore.postgres(postgres_dsn())
    try:
        with store:
            yield store
    finally:
        _reset()


def _reset() -> None:
    """Empty the links table, on a connection of this function's own.

    Its own, because closing the store under test is what the first
    version of this did.
    """
    scratch = LinkStore.postgres(postgres_dsn())
    try:
        scratch._db.execute("DROP TABLE IF EXISTS links")
        scratch._db.commit()
        scratch._create()
    finally:
        scratch.close()


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=needs_postgres)])
def store(request: pytest.FixtureRequest) -> LinkStore:
    """Every test below runs against both backends."""
    return request.getfixturevalue(f"{request.param}_store")  # type: ignore[no-any-return]


def link(
    name: str, *, kind: LinkKind = LinkKind.READS_KEY, chunk: str = "c1", line: int = 1
) -> Link:
    return Link(src_chunk_id=chunk, kind=kind, name=name, line=line)


# --- the three queries the store exists for -------------------------------


def test_by_name_finds_what_was_written(store: LinkStore) -> None:
    store.add_links([link("8080")], repo="r", path="src/app.py")

    found = store.by_name("8080")

    assert [(edge.name, edge.repo, edge.path, edge.line) for edge in found] == [
        ("8080", "r", "src/app.py", 1)
    ]


def test_out_of_reads_links_of_given_chunks(store: LinkStore) -> None:
    store.add_links([link("8080", chunk="a"), link("9090", chunk="b")], repo="r", path="p.py")

    assert {edge.name for edge in store.out_of(["a"])} == {"8080"}
    assert {edge.name for edge in store.out_of(["a", "b"])} == {"8080", "9090"}
    assert store.out_of([]) == []


def test_out_of_narrows_by_kind(store: LinkStore) -> None:
    store.add_links(
        [link("8080", chunk="a"), link("8080", chunk="a", kind=LinkKind.DECLARES, line=2)],
        repo="r",
        path="p.py",
    )

    found = store.out_of(["a"], kind=LinkKind.DECLARES)

    assert [edge.kind for edge in found] == [LinkKind.DECLARES]


def test_dangling_is_the_anti_join(store: LinkStore) -> None:
    # The query a vector store cannot express at all: which reads_key
    # names have no declares anywhere in the workspace.
    store.add_links([link("8080"), link("9090", chunk="c2")], repo="app", path="src/client.py")
    store.add_links(
        [link("8080", kind=LinkKind.DECLARES, chunk="c3")], repo="app", path="compose.yml"
    )

    assert [edge.name for edge in store.dangling()] == ["9090"]


# --- the lifetime rules ---------------------------------------------------


def test_writing_the_same_link_twice_stores_it_once(store: LinkStore) -> None:
    first = store.add_links([link("8080")], repo="r", path="p.py")
    second = store.add_links([link("8080")], repo="r", path="p.py")

    assert (first, second) == (1, 0)
    assert store.count() == 1


def test_forgetting_a_chunk_forgets_its_links(store: LinkStore) -> None:
    store.add_links([link("8080", chunk="a"), link("9090", chunk="b")], repo="r", path="p.py")

    removed = store.delete_by_source(["a"])

    assert removed == 1
    assert {edge.name for edge in store.by_name("8080")} == set()
    assert {edge.name for edge in store.by_name("9090")} == {"9090"}


def test_nothing_to_write_or_forget_is_not_an_error(store: LinkStore) -> None:
    assert store.add_links([], repo="r", path="p.py") == 0
    assert store.delete_by_source([]) == 0
    assert store.count() == 0


def test_a_resolved_link_keeps_both_ends(store: LinkStore) -> None:
    store.add_links(
        [
            Link(
                src_chunk_id="c1",
                kind=LinkKind.BLAMED_BY,
                name="e29017f",
                line=1,
                dst_chunk_id="commit-chunk",
                url=None,
            )
        ],
        repo="r",
        path="p.py",
    )

    edge = store.by_name("e29017f")[0]

    assert (edge.dst_chunk_id, edge.url) == ("commit-chunk", None)


def test_a_reference_keeps_its_url(store: LinkStore) -> None:
    store.add_links(
        [
            Link(
                src_chunk_id="c1",
                kind=LinkKind.REFERENCES,
                name="PROJ-412",
                line=3,
                url="https://tracker/browse/PROJ-412",
            )
        ],
        repo="r",
        path="commits",
    )

    assert store.by_name("PROJ-412")[0].url == "https://tracker/browse/PROJ-412"


def test_rows_come_back_in_reading_order(store: LinkStore) -> None:
    store.add_links([link("8080", chunk="b", line=9)], repo="r", path="src/z.py")
    store.add_links([link("8080", chunk="a", line=2)], repo="r", path="src/a.py")

    assert [edge.path for edge in store.by_name("8080")] == ["src/a.py", "src/z.py"]


# --- what each backend says about itself ----------------------------------


def test_sqlite_is_the_default_and_needs_no_service(tmp_path: Path) -> None:
    with LinkStore(tmp_path / "idx") as store:
        assert store.path is not None
        assert store.path.name == "links.db"


@needs_postgres
def test_postgres_has_no_file_of_its_own() -> None:
    store = LinkStore.postgres(postgres_dsn())
    try:
        assert store.path is None
    finally:
        store.close()
