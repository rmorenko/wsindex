"""Link store and the lifetime rule ADR-9 owed.

The debt this closes: a link is keyed by `chunk_id`, and a chunk id is
`sha256(text, path)` — it does not survive an edit. Without deleting a
chunk's links when the chunk goes, two things rot. The store grows edges
pointing at nothing forever, and those orphans are indistinguishable from
real dangling links, so the drift report fills with references from code
that no longer exists and stops being worth reading.

Most of what follows tests the store in isolation; the end-to-end cases
at the bottom drive a real `Pipeline.index` over a real git repo, because
the guarantee is about the two deletions happening together and only a
real run proves that.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from wsindex.config import Config, Repository
from wsindex.embed import FakeEmbedder
from wsindex.links import Link, LinkKind, LinkStore
from wsindex.model import Chunk, Kind
from wsindex.pipeline import Pipeline
from wsindex.store import LanceDBStore

Committer = Callable[[Path], None]


def link(src: str, kind: LinkKind, name: str, line: int = 1) -> Link:
    return Link(src_chunk_id=src, kind=kind, name=name, line=line)


@pytest.fixture
def links(tmp_path: Path) -> LinkStore:
    return LinkStore(tmp_path / "idx")


# --- the store on its own ------------------------------------------------


def test_links_round_trip(links: LinkStore) -> None:
    assert links.add_links([link("a", LinkKind.READS_KEY, "8080")], repo="r", path="a.py") == 1
    assert links.count() == 1


def test_adding_the_same_link_twice_writes_once(links: LinkStore) -> None:
    # Idempotent for the same reason `add_chunks` is: a re-index must not
    # double what it finds.
    one = [link("a", LinkKind.READS_KEY, "8080")]
    links.add_links(one, repo="r", path="a.py")
    assert links.add_links(one, repo="r", path="a.py") == 0
    assert links.count() == 1


def test_adding_nothing_is_not_an_error(links: LinkStore) -> None:
    assert links.add_links([], repo="r", path="a.py") == 0


def test_the_database_survives_reopening(tmp_path: Path) -> None:
    with LinkStore(tmp_path / "idx") as first:
        first.add_links([link("a", LinkKind.READS_KEY, "8080")], repo="r", path="a.py")
    with LinkStore(tmp_path / "idx") as second:
        assert second.count() == 1


# --- the drift query -----------------------------------------------------


def test_an_unanswered_reference_is_drift(links: LinkStore) -> None:
    links.add_links([link("code", LinkKind.READS_KEY, "8080")], repo="r", path="config.py")
    found = links.dangling()
    assert [(d.name, d.path) for d in found] == [("8080", "config.py")]


def test_a_reference_a_config_declares_is_not_drift(links: LinkStore) -> None:
    links.add_links([link("code", LinkKind.READS_KEY, "8000")], repo="r", path="config.py")
    links.add_links([link("cfg", LinkKind.DECLARES, "8000")], repo="r", path="compose.yml")
    assert links.dangling() == []


def test_a_declaration_alone_is_never_drift(links: LinkStore) -> None:
    # A config publishing a port nobody reads is not a problem.
    links.add_links([link("cfg", LinkKind.DECLARES, "9999")], repo="r", path="compose.yml")
    assert links.dangling() == []


def test_drift_is_ordered_for_reading(links: LinkStore) -> None:
    links.add_links(
        [link("b", LinkKind.READS_KEY, "1", line=9), link("a", LinkKind.READS_KEY, "2", line=2)],
        repo="r",
        path="z.py",
    )
    links.add_links([link("c", LinkKind.READS_KEY, "3", line=5)], repo="r", path="a.py")
    assert [d.path for d in links.dangling()] == ["a.py", "z.py", "z.py"]


# --- deletion: the debt itself -------------------------------------------


def test_deleting_a_chunk_takes_its_links(links: LinkStore) -> None:
    links.add_links(
        [link("gone", LinkKind.READS_KEY, "8080"), link("kept", LinkKind.READS_KEY, "9090")],
        repo="r",
        path="a.py",
    )
    assert links.delete_by_source(["gone"]) == 1
    assert [d.name for d in links.dangling()] == ["9090"]


def test_deleting_nothing_is_not_an_error(links: LinkStore) -> None:
    assert links.delete_by_source([]) == 0


def test_deleting_an_unknown_chunk_is_a_no_op(links: LinkStore) -> None:
    links.add_links([link("a", LinkKind.READS_KEY, "1")], repo="r", path="a.py")
    assert links.delete_by_source(["never-existed"]) == 0
    assert links.count() == 1


def test_losing_a_declaration_brings_drift_back(links: LinkStore) -> None:
    # The lifetime rule and the feature are the same mechanism: delete
    # the config that published a port and the code reading it becomes
    # dangling on the next query, with nothing else to update.
    links.add_links([link("code", LinkKind.READS_KEY, "8000")], repo="r", path="config.py")
    links.add_links([link("cfg", LinkKind.DECLARES, "8000")], repo="r", path="compose.yml")
    assert links.dangling() == []

    links.delete_by_source(["cfg"])
    assert [d.name for d in links.dangling()] == ["8000"]


# --- end to end, through a real index run --------------------------------


@pytest.fixture
def commit(monkeypatch: pytest.MonkeyPatch) -> Committer:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{name}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{name}_EMAIL", "test@example.invalid")

    def run(root: Path) -> None:
        if not (root / ".git").exists():
            subprocess.run(
                ["git", "init", "-q", "--initial-branch=main"],
                cwd=root,
                check=True,
                capture_output=True,
            )
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-qm", "snapshot", "--allow-empty"],
            cwd=root,
            check=True,
            capture_output=True,
        )

    return run


@pytest.fixture
def workspace(tmp_path: Path, commit: Committer) -> tuple[Pipeline, LinkStore, Path]:
    """The 8080/8000 shape, reconstructed: code says 8080, compose says 8000."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.py").write_text('BASE_URL = "http://localhost:8080"\n')
    (repo / "docker-compose.yml").write_text('services:\n  app:\n    ports:\n      - "8000:7860"\n')
    commit(repo)
    config = Config.default("test")
    config.add_repo(Repository(id="r", path=str(repo)))
    store = LinkStore(tmp_path / "idx")
    pipeline = Pipeline(
        store=LanceDBStore(uri=str(tmp_path / "db"), embedder=FakeEmbedder()),
        state_dir=tmp_path / "state",
        links=store,
    )
    return pipeline, store, repo


def test_indexing_finds_the_8080_drift(
    workspace: tuple[Pipeline, LinkStore, Path],
) -> None:
    # The step-26 exam, now as production behaviour rather than a probe.
    pipeline, links, _ = workspace
    pipeline.index()
    found = links.dangling()
    assert [(d.name, d.path) for d in found] == [("8080", "config.py")]


def test_fixing_the_code_clears_the_drift(
    workspace: tuple[Pipeline, LinkStore, Path], commit: Committer
) -> None:
    pipeline, links, repo = workspace
    pipeline.index()
    (repo / "config.py").write_text('BASE_URL = "http://localhost:8000"\n')
    commit(repo)

    pipeline.index()
    assert links.dangling() == []


def test_a_rewritten_file_leaves_no_orphan_links(
    workspace: tuple[Pipeline, LinkStore, Path], commit: Committer
) -> None:
    # The debt, asserted directly: every link's source must still be a
    # chunk that exists. An orphan here is indistinguishable from real
    # drift, which is what would poison the report.
    pipeline, links, repo = workspace
    pipeline.index()
    (repo / "config.py").write_text('BASE_URL = "http://localhost:8000"\n')
    commit(repo)
    pipeline.index()

    sources = {row[0] for row in links._db.execute("SELECT DISTINCT src_chunk_id FROM links")}
    live = pipeline.store.chunk_ids(dataset_name="r")
    assert sources <= live


def test_deleting_the_config_brings_drift_back_through_a_real_run(
    workspace: tuple[Pipeline, LinkStore, Path], commit: Committer
) -> None:
    # End to end, the direction that proves the two halves are one
    # mechanism: fix the code, then delete the config it now agrees with.
    pipeline, links, repo = workspace
    pipeline.index()
    (repo / "config.py").write_text('BASE_URL = "http://localhost:8000"\n')
    commit(repo)
    pipeline.index()
    assert links.dangling() == []

    (repo / "docker-compose.yml").unlink()
    commit(repo)
    pipeline.index()
    assert [d.name for d in links.dangling()] == ["8000"]


def test_a_pipeline_without_a_link_store_still_indexes(tmp_path: Path, commit: Committer) -> None:
    # Links are optional: nothing about indexing depends on them.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text('X = "http://localhost:8080"\n')
    commit(repo)
    config = Config.default("test")
    config.add_repo(Repository(id="r", path=str(repo)))
    pipeline = Pipeline(
        store=LanceDBStore(uri=str(tmp_path / "db"), embedder=FakeEmbedder()),
        state_dir=tmp_path / "state",
    )
    assert pipeline.index().written > 0


# --- what each side of the rule contributes ------------------------------


def test_a_port_in_prose_is_neither_side_of_the_drift_rule() -> None:
    # A sentence naming a port is not a claim anyone can be held to, so a
    # doc chunk is neither a reference nor a declaration. Since
    # external references arrived,
    # it does contribute a REFERENCES link for the url itself, which is a
    # different assertion entirely.
    from wsindex.ingest.link_extract import links_for
    from wsindex.model import Chunk, Kind

    doc = Chunk(
        repo="r",
        path="README.md",
        lang="markdown",
        kind=Kind.DOC,
        symbol=None,
        node_type=None,
        start_line=1,
        end_line=1,
        text="The service runs on http://localhost:8080 by default.",
    )
    kinds = {link.kind for link in links_for([doc])}
    assert LinkKind.READS_KEY not in kinds
    assert LinkKind.DECLARES not in kinds


def test_link_lines_are_file_lines_not_chunk_lines() -> None:
    # A link points a person at a file; a chunk offset would send them to
    # the wrong place in every chunk but the first.
    from wsindex.ingest.link_extract import links_for
    from wsindex.model import Chunk, Kind

    chunk = Chunk(
        repo="r",
        path="a.py",
        lang="python",
        kind=Kind.CODE,
        symbol=None,
        node_type=None,
        start_line=40,
        end_line=42,
        text='def f():\n    return "http://localhost:8080"\n',
    )
    assert [link.line for link in links_for([chunk])] == [41]


def test_a_port_key_in_a_config_is_a_declaration() -> None:
    # Not every config publishes ports the compose way. `port: 8000` in
    # a yaml or `port = 8000` in a toml declares one just as much.
    from wsindex.ingest.link_extract import links_for
    from wsindex.model import Chunk, Kind

    cfg = Chunk(
        repo="r",
        path="app.yaml",
        lang="yaml",
        kind=Kind.CONFIG,
        symbol=None,
        node_type=None,
        start_line=1,
        end_line=1,
        text="server:\n  port: 8000\n",
    )
    assert [(link.kind, link.name) for link in links_for([cfg])] == [(LinkKind.DECLARES, "8000")]


# --- the name pair -------------------------------------------------------


def code(text: str, *, symbol: str | None = None, path: str = "a.go", line: int = 1) -> Chunk:
    return Chunk(
        repo="r",
        path=path,
        lang="go",
        kind=Kind.CODE,
        symbol=symbol,
        node_type=None,
        start_line=line,
        end_line=line,
        text=text,
    )


def test_a_definition_and_a_use_of_the_same_name_meet(links: LinkStore) -> None:
    # The join this pair exists for, and the thing the config pair could
    # almost never do: across five workspaces exactly three names of
    # 6 896 had both a READS_KEY and a DECLARES. Both sides are stored
    # unresolved, because the extractor sees one file and the definition
    # is usually in another repository entirely.
    from wsindex.ingest.link_extract import links_for

    definition = links_for([code("func ServeHTTP() {}", symbol="ServeHTTP")])
    use = links_for([code("h.ServeHTTP(w, r)", path="b.go")])
    links.add_links(definition, repo="r", path="a.go")
    links.add_links(use, repo="r", path="b.go")

    found = {(edge.kind, edge.path) for edge in links.by_name("ServeHTTP")}
    assert found == {(LinkKind.DEFINES, "a.go"), (LinkKind.MENTIONS, "b.go")}


def test_a_chunk_does_not_mention_the_name_it_defines() -> None:
    # Otherwise every definition answers its own query and `refs` reports
    # the definition twice, once under each label.
    from wsindex.ingest.link_extract import links_for

    found = links_for([code("func ServeHTTP() {\n  return ServeHTTP\n}", symbol="ServeHTTP")])

    assert [edge.kind for edge in found] == [LinkKind.DEFINES]


def test_a_keyword_is_not_a_name_worth_an_edge() -> None:
    # The filter that makes this affordable, and it is a measured choice
    # rather than taste. Storing every token of four characters or more
    # kept all 810 of caddyserver's cross-file names but cost 186 100
    # edges — nineteen per chunk, 1.9M rows on a 100k-chunk workspace.
    # Requiring an internal word boundary keeps 664 of the 810 for
    # 31 839 edges, and it needs no per-language stop-list precisely
    # because keywords are single lowercase words in all sixteen
    # grammars.
    from wsindex.ingest.link_extract import links_for

    found = links_for([code("return errors.New(string(value))\nreadConfig()\n")])

    assert [edge.name for edge in found] == ["readConfig"]


def test_a_name_is_recorded_once_at_its_first_line() -> None:
    # A variable used nine times inside its own function is one fact, not
    # nine, and the nine tell a reader nothing they cannot see once the
    # file is open.
    from wsindex.ingest.link_extract import links_for

    found = links_for([code("x := maxRetries\ny := maxRetries\n", line=40)])

    assert [(edge.name, edge.line) for edge in found] == [("maxRetries", 40)]


def test_a_method_is_stored_under_its_bare_name() -> None:
    # The store holds `Cls.method` for `search --symbol`, but a call site
    # spells it `method`, and a qualified name here would join with
    # nothing.
    from wsindex.ingest.link_extract import links_for

    found = links_for([code("def run_once(self): pass", symbol="Scheduler.run_once")])

    assert [(edge.kind, edge.name) for edge in found] == [(LinkKind.DEFINES, "run_once")]


def test_a_database_from_before_a_column_existed_still_reads(tmp_path: Path) -> None:
    # `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
    # exists, so a links.db written before `url` arrived kept its old
    # shape and every read failed with `no such column`. Found the hard
    # way, on an index built one step earlier.
    import sqlite3

    (tmp_path / "idx").mkdir()
    old = sqlite3.connect(tmp_path / "idx" / "links.db")
    old.execute(
        "CREATE TABLE links (src_chunk_id TEXT NOT NULL, kind TEXT NOT NULL, "
        "name TEXT NOT NULL, line INTEGER NOT NULL, dst_chunk_id TEXT, "
        "repo TEXT NOT NULL, path TEXT NOT NULL, "
        "PRIMARY KEY (src_chunk_id, kind, name, line))"
    )
    old.execute("INSERT INTO links VALUES ('c1', 'reads_key', '8080', 1, NULL, 'r', 'a.py')")
    old.commit()
    old.close()

    with LinkStore(tmp_path / "idx") as store:
        edges = store.by_name("8080")
        assert [(edge.name, edge.url) for edge in edges] == [("8080", None)]
        # And it still accepts rows that use the new column.
        store.add_links(
            [
                Link(
                    src_chunk_id="c2",
                    kind=LinkKind.REFERENCES,
                    name="#7",
                    line=2,
                    url="https://x.invalid/7",
                )
            ],
            repo="r",
            path="b.md",
        )
        assert store.by_name("#7")[0].url == "https://x.invalid/7"


def test_by_name_is_the_inverted_index(links: LinkStore) -> None:
    links.add_links([link("code", LinkKind.READS_KEY, "8000")], repo="r", path="a.py")
    links.add_links([link("cfg", LinkKind.DECLARES, "8000")], repo="r", path="compose.yml")
    links.add_links([link("other", LinkKind.READS_KEY, "9999")], repo="r", path="b.py")

    found = links.by_name("8000")
    assert {edge.kind for edge in found} == {LinkKind.READS_KEY, LinkKind.DECLARES}
    assert {edge.path for edge in found} == {"a.py", "compose.yml"}


def test_by_name_on_an_unknown_name_is_empty(links: LinkStore) -> None:
    assert links.by_name("nothing") == []


def test_out_of_reads_links_leaving_a_chunk(links: LinkStore) -> None:
    links.add_links(
        [link("c1", LinkKind.READS_KEY, "8000"), link("c1", LinkKind.BLAMED_BY, "abc1234", line=2)],
        repo="r",
        path="a.py",
    )
    assert len(links.out_of(["c1"])) == 2
    assert [e.name for e in links.out_of(["c1"], kind=LinkKind.BLAMED_BY)] == ["abc1234"]


def test_out_of_nothing_is_empty(links: LinkStore) -> None:
    assert links.out_of([]) == []
