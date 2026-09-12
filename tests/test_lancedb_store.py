"""Tests for LanceDBStore: the single-table-with-repo-column layout (ADR-7).

Ported from the LocalStore suite where the contract is backend-agnostic;
reworked where the mechanics changed (a missing dataset is a registry
miss, not a missing directory; a foreign embedder dim fails at
construction, not at search). New, mandated by ADR-7: prefilter parity,
deterministic tie-breaking, KNN == numpy brute force.
"""

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from wsindex.embed.embedder import FakeEmbedder
from wsindex.model import Chunk, Hit, Kind, SearchFilter, SourceFile
from wsindex.store.lancedb import LanceDBStore, retrieval_text

DIM = 8


class CountingFake(FakeEmbedder):
    """FakeEmbedder that records every text it was asked to embed."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__(dim=dim)
        self.embedded: list[str] = []

    def _embed(self, texts: "Sequence[str]") -> list[list[float]]:
        self.embedded.extend(texts)
        return super()._embed(texts)


def make_chunk(
    text: str,
    path: str = "doc.md",
    *,
    lang: str = "text",
    kind: Kind = Kind.DOC,
    symbol: str | None = None,
) -> Chunk:
    return Chunk(
        repo="r",
        path=path,
        lang=lang,
        kind=kind,
        symbol=symbol,
        node_type=None,
        start_line=1,
        end_line=1,
        text=text,
    )


@pytest.fixture
def store(tmp_path: Path) -> LanceDBStore:
    s = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    s.create_dataset("ds", metric="cosine")
    return s


# --- ported as-is: the contract is backend-agnostic ------------------------


def test_add_chunks_then_search_nearest_first(store: LanceDBStore) -> None:
    a, b = make_chunk("a"), make_chunk("b")
    assert store.add_chunks("ds", chunks=[a, b]) == 2
    # Queried with what was embedded, not with the stored slice: the two
    # are different strings now, and only the first is a vector in there.
    hits = store.search("ds", query=retrieval_text(a), k=2)
    assert [h.metadata["text"] for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0)  # the same string embeds identically
    assert hits[0].score > hits[1].score
    assert hits[0].native_id == a.id
    assert hits[0].metadata == a.to_metadata()  # full metadata round-trip


def test_add_chunks_skips_duplicates(store: LanceDBStore) -> None:
    a = make_chunk("a")
    assert store.add_chunks("ds", chunks=[a]) == 1
    # Same chunk again: nothing written, store did not grow.
    assert store.add_chunks("ds", chunks=[a]) == 0
    assert len(store.search("ds", query="a", k=10)) == 1
    # Duplicate inside a single batch counts once.
    b = make_chunk("b")
    assert store.add_chunks("ds", chunks=[b, b]) == 1
    assert len(store.search("ds", query="a", k=10)) == 2


def test_duplicates_are_not_reembedded(tmp_path: Path) -> None:
    # Dedup happens BEFORE the (expensive) embedding call — including
    # duplicates inside one batch, which LanceDB would happily store twice.
    embedder = CountingFake()
    store = LanceDBStore(str(tmp_path / "db"), embedder=embedder)
    store.create_dataset("ds", metric="cosine")
    a = make_chunk("a")
    store.add_chunks("ds", chunks=[a, a])
    # What reaches the embedder is the chunk's name and place followed by
    # its text, not the text alone — see `retrieval_text`.
    assert embedder.embedded == [retrieval_text(a)]  # batch-internal duplicate embedded once
    store.add_chunks("ds", chunks=[a])
    assert embedder.embedded == [retrieval_text(a)]  # second run embedded nothing


def test_incremental_add_chunks_keeps_old_vectors(store: LanceDBStore) -> None:
    a, b = make_chunk("a"), make_chunk("b")
    assert store.add_chunks("ds", chunks=[a]) == 1
    assert store.add_chunks("ds", chunks=[b]) == 1
    hits = store.search("ds", query=retrieval_text(a), k=10)
    assert len(hits) == 2
    # The *stored* text is still the verbatim slice, whatever was embedded.
    assert hits[0].metadata["text"] == "a"


def test_create_is_idempotent(store: LanceDBStore) -> None:
    store.add_chunks("ds", chunks=[make_chunk("a")])
    store.create_dataset("ds", metric="cosine")  # same metric: no-op
    assert len(store.search("ds", query="a", k=10)) == 1  # data survived


def test_search_empty_dataset(store: LanceDBStore) -> None:
    assert store.search("ds", query="anything", k=5) == []


def test_search_unknown_dataset_raises(store: LanceDBStore) -> None:
    with pytest.raises(ValueError, match="not present"):
        store.search("nope", query="anything", k=5)


def test_k_larger_than_store(store: LanceDBStore) -> None:
    store.add_chunks("ds", chunks=[make_chunk("a")])
    assert len(store.search("ds", query="a", k=50)) == 1


def test_persistence_across_instances(store: LanceDBStore, tmp_path: Path) -> None:
    a = make_chunk("a")
    store.add_chunks("ds", chunks=[a])
    # A fresh instance must see both the registry and the data on disk.
    reopened = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    hits = reopened.search("ds", query="a", k=5)
    assert [h.native_id for h in hits] == [a.id]


# --- reworked: same contract, different mechanics --------------------------


def test_create_wrong_metric_raises(store: LanceDBStore) -> None:
    # LanceDB binds the metric at query time, so this guard is ours.
    with pytest.raises(ValueError, match="cosine"):
        store.create_dataset("other", metric="dot")


def test_foreign_embedder_dim_fails_at_construction(tmp_path: Path) -> None:
    # The dim is baked into the single table's schema, so a changed
    # embedder fails already at connect/ensure-table — not at search.
    LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    with pytest.raises(ValueError):
        LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM + 1))


def test_add_chunks_unknown_dataset_raises(store: LanceDBStore) -> None:
    # LocalStore surfaced this as FileNotFoundError from the missing
    # directory; here the registry answers, and the contract says ValueError.
    with pytest.raises(ValueError, match="not present"):
        store.add_chunks("nope", chunks=[make_chunk("a")])


# --- new, mandated by ADR-7 ------------------------------------------------


def test_knn_matches_bruteforce(store: LanceDBStore) -> None:
    texts = [f"text {i}" for i in range(20)]
    chunks = [make_chunk(t, path=f"f{i}.md") for i, t in enumerate(texts)]
    store.add_chunks("ds", chunks=chunks)

    query = "where is cosine similarity computed"
    # Brute force over what was actually stored: the store embeds each
    # chunk's name and place with its text, so comparing against the bare
    # texts would be comparing against vectors that were never written.
    vecs = np.array(store.embedder.embed([retrieval_text(c) for c in chunks]))
    q = np.array(store.embedder.embed([query])[0])
    cos = (vecs @ q) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q))
    order = np.argsort(cos)[::-1][:5]

    hits = store.search("ds", query=query, k=5)
    assert [h.native_id for h in hits] == [chunks[i].id for i in order]
    assert np.allclose([h.score for h in hits], np.sort(cos)[::-1][:5], atol=1e-5)


def test_prefilter_parity(tmp_path: Path) -> None:
    # ADR-7 mandate: KNN over a repo subset must equal KNN over an
    # equivalent standalone table — the prefilter must not distort top-k.
    targets = [make_chunk(f"target {i}", path=f"t{i}.md") for i in range(10)]
    noise = [make_chunk(f"noise {i}", path=f"n{i}.md") for i in range(50)]

    full = LanceDBStore(str(tmp_path / "full"), embedder=FakeEmbedder(dim=DIM))
    full.create_dataset("ds", metric="cosine")
    full.create_dataset("other", metric="cosine")
    full.add_chunks("ds", chunks=targets)
    full.add_chunks("other", chunks=noise)

    solo = LanceDBStore(str(tmp_path / "solo"), embedder=FakeEmbedder(dim=DIM))
    solo.create_dataset("ds", metric="cosine")
    solo.add_chunks("ds", chunks=targets)

    a = full.search("ds", query="target 3", k=5)
    b = solo.search("ds", query="target 3", k=5)
    assert [h.native_id for h in a] == [h.native_id for h in b]
    assert np.allclose([h.score for h in a], [h.score for h in b], atol=1e-6)


def test_equal_scores_are_deterministic(store: LanceDBStore) -> None:
    # Same text at two paths: identical vectors, identical scores —
    # repeated searches must not shuffle the order.
    twins = [make_chunk("same text", path="one.md"), make_chunk("same text", path="two.md")]
    store.add_chunks("ds", chunks=twins)
    first = [h.native_id for h in store.search("ds", query="same text", k=2)]
    for _ in range(5):
        assert [h.native_id for h in store.search("ds", query="same text", k=2)] == first


def test_dataset_name_with_quote(tmp_path: Path) -> None:
    # The repo predicate is built as a SQL string — a quote in the name
    # must be escaped, not break the query (or worse).
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    store.create_dataset("o'reilly", metric="cosine")
    a = make_chunk("a")
    assert store.add_chunks("o'reilly", chunks=[a]) == 1
    assert store.add_chunks("o'reilly", chunks=[a]) == 0  # dedup scan escaped too
    hits = store.search("o'reilly", query="a", k=5)
    assert [h.native_id for h in hits] == [a.id]


# --- structural filters, prefilter semantics ------------------------------


@pytest.fixture
def mixed_store(tmp_path: Path) -> LanceDBStore:
    """Store with a mixed corpus: 3 langs, 2 kinds, path variety."""
    s = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    s.create_dataset("ds", metric="cosine")
    chunks = [
        make_chunk("py code alpha", "src/a.py", lang="python", kind=Kind.CODE, symbol="alpha"),
        make_chunk("py code beta", "src/b.py", lang="python", kind=Kind.CODE, symbol="beta"),
        make_chunk("ts code gamma", "web/a.ts", lang="typescript", kind=Kind.CODE, symbol="gamma"),
        make_chunk("ts code delta", "web/b.ts", lang="typescript", kind=Kind.CODE, symbol="delta"),
        make_chunk("md doc one", "docs/one.md", lang="markdown", kind=Kind.DOC),
        make_chunk("md doc two", "docs/two.md", lang="markdown", kind=Kind.DOC),
        make_chunk("toml cfg", "pyproject.toml", lang="toml", kind=Kind.CONFIG),
    ]
    s.add_chunks("ds", chunks=chunks)
    return s


def test_filter_by_single_lang(mixed_store: LanceDBStore) -> None:
    hits = mixed_store.search("ds", query="x", k=10, filters=SearchFilter(lang=("python",)))
    assert {h.metadata["lang"] for h in hits} == {"python"}
    assert len(hits) == 2


def test_filter_by_multiple_langs_is_OR(mixed_store: LanceDBStore) -> None:
    hits = mixed_store.search(
        "ds", query="x", k=10, filters=SearchFilter(lang=("python", "typescript"))
    )
    assert {h.metadata["lang"] for h in hits} == {"python", "typescript"}
    assert len(hits) == 4


def test_filter_by_kind(mixed_store: LanceDBStore) -> None:
    hits = mixed_store.search("ds", query="x", k=10, filters=SearchFilter(kind=(Kind.DOC,)))
    assert {h.metadata["kind"] for h in hits} == {"doc"}
    assert len(hits) == 2


def test_filter_by_path_prefix_glob(mixed_store: LanceDBStore) -> None:
    hits = mixed_store.search("ds", query="x", k=10, filters=SearchFilter(path="src/*"))
    assert {h.metadata["path"] for h in hits} == {"src/a.py", "src/b.py"}


def test_filter_by_path_suffix_glob(mixed_store: LanceDBStore) -> None:
    hits = mixed_store.search("ds", query="x", k=10, filters=SearchFilter(path="*.md"))
    assert {h.metadata["path"] for h in hits} == {"docs/one.md", "docs/two.md"}


def test_filter_by_path_literal(mixed_store: LanceDBStore) -> None:
    # No wildcard: exact-string match through LIKE.
    hits = mixed_store.search("ds", query="x", k=10, filters=SearchFilter(path="pyproject.toml"))
    assert [h.metadata["path"] for h in hits] == ["pyproject.toml"]


def test_filter_by_single_char_wildcard(mixed_store: LanceDBStore) -> None:
    hits = mixed_store.search("ds", query="x", k=10, filters=SearchFilter(path="src/?.py"))
    assert {h.metadata["path"] for h in hits} == {"src/a.py", "src/b.py"}


def test_filter_by_symbol_substring(mixed_store: LanceDBStore) -> None:
    hits = mixed_store.search("ds", query="x", k=10, filters=SearchFilter(symbol="lph"))
    assert [h.metadata["symbol"] for h in hits] == ["alpha"]


def test_filter_combined_is_AND(mixed_store: LanceDBStore) -> None:
    # python AND CODE AND src/* — same set as python alone here, but the
    # AND semantic must hold when fields would disagree.
    hits = mixed_store.search(
        "ds",
        query="x",
        k=10,
        filters=SearchFilter(lang=("python",), kind=(Kind.DOC,)),
    )
    assert hits == []  # python doesn't intersect DOC


def test_empty_filter_matches_all(mixed_store: LanceDBStore) -> None:
    # An all-empty SearchFilter is functionally the same as passing None:
    # the store must not build a broken WHERE with no predicates.
    hits_none = mixed_store.search("ds", query="x", k=10, filters=None)
    hits_empty = mixed_store.search("ds", query="x", k=10, filters=SearchFilter())
    assert [h.native_id for h in hits_none] == [h.native_id for h in hits_empty]
    assert len(hits_none) == 7  # every chunk in the store


def test_prefilter_returns_k_when_selective(tmp_path: Path) -> None:
    # THE key invariant: with a selective filter, top-k is computed over
    # the filtered subset — the caller gets k hits, not k-minus-filtered.
    # A postfilter over top-k would return under-full lists.
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    store.create_dataset("ds", metric="cosine")
    targets = [
        make_chunk(f"py {i}", path=f"src/t{i}.py", lang="python", kind=Kind.CODE) for i in range(5)
    ]
    noise = [
        make_chunk(f"noise {i}", path=f"n/n{i}.md", lang="markdown", kind=Kind.DOC)
        for i in range(50)
    ]
    store.add_chunks("ds", chunks=targets + noise)
    hits = store.search("ds", query="query", k=5, filters=SearchFilter(lang=("python",)))
    assert len(hits) == 5  # the whole target set, none dropped by a would-be postfilter
    assert {h.metadata["lang"] for h in hits} == {"python"}


def test_filter_sql_injection_in_lang_is_escaped(mixed_store: LanceDBStore) -> None:
    # Malicious value must not break the query or escape the predicate;
    # the store escapes single quotes just like it does for the dataset name.
    hits = mixed_store.search(
        "ds", query="x", k=10, filters=SearchFilter(lang=("python' OR '1'='1",))
    )
    assert hits == []  # no chunk has that literal language


def test_filter_path_escapes_like_metachars(tmp_path: Path) -> None:
    # `_` in a path is a SQL LIKE wildcard; the store must escape it so
    # `test_a.py` filter does NOT match `testXa.py`.
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    store.create_dataset("ds", metric="cosine")
    match = make_chunk("m", path="test_a.py")
    almost = make_chunk("n", path="testXa.py")
    store.add_chunks("ds", chunks=[match, almost])
    hits = store.search("ds", query="q", k=10, filters=SearchFilter(path="test_a.py"))
    assert [h.metadata["path"] for h in hits] == ["test_a.py"]


# --- delete_chunks --------------------------------------------------------


def test_delete_makes_chunks_invisible_to_search(store: LanceDBStore) -> None:
    # THE invariant of the method: everything else is "syntax works".
    a, b = make_chunk("a"), make_chunk("b", path="b.md")
    store.add_chunks("ds", chunks=[a, b])
    assert len(store.search("ds", query="a", k=10)) == 2  # control
    store.delete_chunks("ds", ids=[a.id])
    hits = store.search("ds", query="a", k=10)
    assert [h.native_id for h in hits] == [b.id]


def test_delete_returns_num_deleted_rows(store: LanceDBStore) -> None:
    # Returns what LanceDB actually removed, not what the caller asked to.
    a, b = make_chunk("a"), make_chunk("b", path="b.md")
    store.add_chunks("ds", chunks=[a, b])
    assert store.delete_chunks("ds", ids=[a.id, b.id]) == 2


def test_delete_empty_ids_is_noop_returns_zero(store: LanceDBStore) -> None:
    # Short-circuit before SQL: DataFusion rejects `IN ()`, so an empty
    # batch must never reach the query builder.
    a = make_chunk("a")
    store.add_chunks("ds", chunks=[a])
    assert store.delete_chunks("ds", ids=[]) == 0
    assert len(store.search("ds", query="a", k=10)) == 1  # nothing was touched


def test_delete_unknown_ids_returns_zero(store: LanceDBStore) -> None:
    # Idempotent on missing ids: no crash, no partial-delete, returns 0.
    a = make_chunk("a")
    store.add_chunks("ds", chunks=[a])
    ghost = Chunk.chunk_id("never added", path="nowhere.md")
    assert store.delete_chunks("ds", ids=[ghost]) == 0
    assert len(store.search("ds", query="a", k=10)) == 1


def test_delete_bare_string_ids_raises_typeerror(store: LanceDBStore) -> None:
    # `str` is itself a Sequence[str] (by character); the guard prevents
    # the caller from silently deleting 64 one-char "ids".
    with pytest.raises(TypeError, match="batch of ids"):
        store.delete_chunks("ds", ids="abc")


def test_delete_unknown_dataset_raises_valueerror(store: LanceDBStore) -> None:
    # Symmetry with add_chunks: same guard, same message.
    with pytest.raises(ValueError, match="not present"):
        store.delete_chunks("nope", ids=["anything"])


def test_delete_scopes_by_dataset(tmp_path: Path) -> None:
    # Cross-dataset isolation: chunk_id = sha256(text, path) — repo is NOT
    # in the hash, so the same chunk lives with the same id in both
    # datasets. Deleting from ds1 must NOT touch ds2 — a naive
    # `id IN (...)` without a dataset predicate would (proved by probe B).
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    store.create_dataset("ds1", metric="cosine")
    store.create_dataset("ds2", metric="cosine")
    shared = make_chunk("shared", path="x.md")
    ds2_only = make_chunk("only ds2", path="y.md")
    store.add_chunks("ds1", chunks=[shared])
    store.add_chunks("ds2", chunks=[shared, ds2_only])

    assert store.delete_chunks("ds1", ids=[shared.id]) == 1

    assert store.search("ds1", query="shared", k=10) == []
    hits2 = store.search("ds2", query="shared", k=10)
    assert {h.native_id for h in hits2} == {shared.id, ds2_only.id}


# --- chunk_ids, the read half of incremental indexing --------------------


def test_chunk_ids_returns_what_was_stored(store: LanceDBStore) -> None:
    store.create_dataset("repo", metric="cosine")
    chunks = [make_chunk("a", path="src/a.py"), make_chunk("b", path="src/b.py")]
    store.add_chunks("repo", chunks=chunks)
    assert store.chunk_ids(dataset_name="repo", paths=["src/a.py"]) == {chunks[0].id}


def test_chunk_ids_without_paths_returns_the_whole_dataset(store: LanceDBStore) -> None:
    store.create_dataset("repo", metric="cosine")
    chunks = [make_chunk("a", path="src/a.py"), make_chunk("b", path="src/b.py")]
    store.add_chunks("repo", chunks=chunks)
    assert store.chunk_ids(dataset_name="repo") == {c.id for c in chunks}


def test_chunk_ids_with_an_empty_path_list_returns_nothing(store: LanceDBStore) -> None:
    # Asking about no paths is not asking about all of them — the None
    # default is the only way to say "everything".
    store.create_dataset("repo", metric="cosine")
    store.add_chunks("repo", chunks=[make_chunk("a", path="src/a.py")])
    assert store.chunk_ids(dataset_name="repo", paths=[]) == set()


def test_chunk_ids_does_not_leak_across_datasets(store: LanceDBStore) -> None:
    # chunk_id = sha256(text, path) carries no repo, so the same file in
    # two datasets shares an id; without the dataset predicate this would
    # report the other dataset's chunk as ours.
    store.create_dataset("one", metric="cosine")
    store.create_dataset("two", metric="cosine")
    shared = make_chunk("same text", path="src/a.py")
    store.add_chunks("one", chunks=[shared])
    assert store.chunk_ids(dataset_name="two", paths=["src/a.py"]) == set()


def test_chunk_ids_on_unknown_dataset_raises(store: LanceDBStore) -> None:
    with pytest.raises(ValueError, match="not present"):
        store.chunk_ids(dataset_name="nope", paths=["a.py"])


def test_chunk_ids_rejects_a_bare_string(store: LanceDBStore) -> None:
    # No `type: ignore` needed here, and that is the whole point: `str`
    # IS a `Sequence[str]`, so mypy cannot catch this call. A bare string
    # would be iterated character by character and query for one-letter
    # paths, so the guard has to exist at runtime.
    store.create_dataset("repo", metric="cosine")
    with pytest.raises(TypeError, match="batch of paths"):
        store.chunk_ids(dataset_name="repo", paths="src/a.py")


def test_chunk_ids_escapes_quotes_in_paths(store: LanceDBStore) -> None:
    store.create_dataset("repo", metric="cosine")
    chunk = make_chunk("x", path="it's/a.py")
    store.add_chunks("repo", chunks=[chunk])
    assert store.chunk_ids(dataset_name="repo", paths=["it's/a.py"]) == {chunk.id}


# --- compact, the reclaim half of delete_chunks --------------------------


def du(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def churn(store: LanceDBStore, batches: int = 12, per_batch: int = 10) -> None:
    """Write many small commits, then delete most of them.

    Reproduces what incremental indexing does over time: every run adds a
    version, and every delete adds another without freeing anything.
    """
    store.create_dataset("repo", metric="cosine")
    for batch in range(batches):
        store.add_chunks(
            "repo",
            chunks=[
                make_chunk(f"chunk text {batch}-{i}" * 10, path=f"f{i}.py")
                for i in range(per_batch)
            ],
        )
    ids = sorted(store.chunk_ids(dataset_name="repo"))
    store.delete_chunks("repo", ids=ids[: len(ids) // 2])


def test_compact_frees_disk(tmp_path: Path) -> None:
    root = tmp_path / "db"
    store = LanceDBStore(str(root), embedder=FakeEmbedder(dim=DIM))
    churn(store)
    before = du(root)

    report = store.compact()

    assert report.bytes_before == before
    assert report.bytes_after is not None
    assert report.bytes_after < before
    assert report.bytes_freed == before - report.bytes_after


def test_compact_collapses_versions(tmp_path: Path) -> None:
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    churn(store)

    report = store.compact()

    # Two tables, one surviving version each: the current one is never removed.
    assert report.versions_before > report.versions_after
    assert report.versions_after == 2


def test_compact_keeps_the_data_searchable(tmp_path: Path) -> None:
    # The point of the guard: reclaiming space must not lose rows.
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    churn(store)
    survivors = store.chunk_ids(dataset_name="repo")

    store.compact()

    assert store.chunk_ids(dataset_name="repo") == survivors
    assert store.search(dataset_name="repo", query="chunk text", k=3)


def test_compact_with_keep_window_retains_history(tmp_path: Path) -> None:
    # Nothing here is a day old, so a one-day window must prune nothing —
    # this is what protects a concurrent reader on a shared store.
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    churn(store)

    report = store.compact(older_than=timedelta(days=1))

    assert report.versions_after >= report.versions_before


def test_compact_on_an_untouched_store_is_harmless(tmp_path: Path) -> None:
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    report = store.compact()
    assert report.versions_after >= 1
    assert report.bytes_after is not None


def test_compact_reports_no_size_for_a_remote_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An s3:// prefix cannot be walked from here. Reporting None is the
    # honest answer; a 0 would read like "nothing was freed".
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    monkeypatch.setattr(type(store.db), "uri", property(lambda self: "s3://bucket/prefix"))

    report = store.compact()

    assert report.bytes_before is None
    assert report.bytes_after is None
    assert report.bytes_freed is None


def test_on_disk_bytes_is_none_when_the_directory_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A local uri that no longer resolves to a directory: measuring is
    # impossible, and None says so rather than claiming zero bytes.
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=DIM))
    monkeypatch.setattr(type(store.db), "uri", property(lambda self: str(tmp_path / "vanished")))
    assert store._on_disk_bytes() is None


def test_chunk_text_fetches_by_id(store: LanceDBStore) -> None:
    store.create_dataset("repo", metric="cosine")
    chunk = make_chunk("the message body", path="commits/x")
    store.add_chunks("repo", chunks=[chunk])
    assert store.chunk_text("repo", ids=[chunk.id]) == {chunk.id: "the message body"}


def test_chunk_text_omits_ids_that_are_not_stored(store: LanceDBStore) -> None:
    # Asking about a chunk a later run re-indexed away is normal, not an
    # error — `why` follows a blame edge that may point at one.
    store.create_dataset("repo", metric="cosine")
    assert store.chunk_text("repo", ids=["never-stored"]) == {}


def test_chunk_text_with_no_ids_reads_nothing(store: LanceDBStore) -> None:
    store.create_dataset("repo", metric="cosine")
    assert store.chunk_text("repo", ids=[]) == {}


def test_chunk_text_rejects_a_bare_string(store: LanceDBStore) -> None:
    # `str` IS a `Sequence[str]`, so the type system cannot catch this:
    # a bare id would be iterated character by character.
    store.create_dataset("repo", metric="cosine")
    with pytest.raises(TypeError, match="batch of ids"):
        store.chunk_text("repo", ids="some-id")


def test_chunk_text_on_unknown_dataset_raises(store: LanceDBStore) -> None:
    with pytest.raises(ValueError, match="not present"):
        store.chunk_text("nope", ids=["x"])


def test_chunk_text_does_not_read_across_datasets(store: LanceDBStore) -> None:
    # One physical table holds every repo (ADR-7) and chunk ids carry no
    # repo, so the dataset predicate is what keeps them apart.
    store.create_dataset("one", metric="cosine")
    store.create_dataset("two", metric="cosine")
    shared = make_chunk("same text", path="a.py")
    store.add_chunks("one", chunks=[shared])
    assert store.chunk_text("two", ids=[shared.id]) == {}


# --- staleness: a handle is pinned to the version it opened at ------------


def test_a_second_handle_does_not_see_writes_until_it_refreshes(tmp_path: Path) -> None:
    # Measured with real processes (ADR-10); reproduced here
    # with two handles, since the snapshot is per handle rather than per
    # process. This is the bug `refresh` exists for: a long-lived reader
    # answers from the corpus it opened with, and never fails doing it.
    uri = str(tmp_path / "db")
    writer = LanceDBStore(uri, embedder=FakeEmbedder(dim=DIM))
    writer.create_dataset("ds", metric="cosine")
    writer.add_chunks("ds", chunks=[make_chunk("first")])
    reader = LanceDBStore(uri, embedder=FakeEmbedder(dim=DIM))
    assert len(reader.search("ds", query="first", k=10)) == 1

    writer.add_chunks("ds", chunks=[make_chunk("second", path="two.md")])

    assert len(reader.search("ds", query="first", k=10)) == 1  # still the old snapshot
    reader.refresh()
    assert len(reader.search("ds", query="first", k=10)) == 2


def test_refresh_finds_a_dataset_another_handle_created(tmp_path: Path) -> None:
    # Both tables, not just the data one: the dataset registry is written
    # by whoever ran `create_dataset`, and a store that refreshed only
    # half would keep answering "no such dataset" for a repo somebody
    # else registered.
    uri = str(tmp_path / "db")
    first = LanceDBStore(uri, embedder=FakeEmbedder(dim=DIM))
    first.create_dataset("ds", metric="cosine")
    second = LanceDBStore(uri, embedder=FakeEmbedder(dim=DIM))
    first.create_dataset("later", metric="cosine")

    with pytest.raises(ValueError):
        second.search("later", query="x", k=1)

    second.refresh()
    assert second.search("later", query="x", k=1) == []


# --- doing each thing once ------------------------------------------------


def test_one_search_embeds_the_query_once_per_dataset(tmp_path: Path) -> None:
    # `Pipeline.search` asks one dataset at a time, so a workspace of R
    # repos ran the same sentence through the model R times — 3.9 ms
    # each, and 48% of a single-repo search.
    class Counting(FakeEmbedder):
        calls = 0

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            Counting.calls += 1
            return super().embed(texts)

    store = LanceDBStore(str(tmp_path / "db"), embedder=Counting(dim=8))
    for name in ("a", "b", "c"):
        store.create_dataset(name, metric="cosine")
        src = SourceFile(repo=name, path="x.py", lang="python", kind=Kind.CODE)
        chunk = src.chunk(text=f"def {name}(): pass", start_line=1, end_line=1)
        store.add_chunks(name, chunks=[chunk])

    Counting.calls = 0
    for name in ("a", "b", "c"):
        store.search(name, query="the same question", k=3)

    assert Counting.calls == 1


def test_a_different_query_is_embedded_again(tmp_path: Path) -> None:
    class Counting(FakeEmbedder):
        calls = 0

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            Counting.calls += 1
            return super().embed(texts)

    store = LanceDBStore(str(tmp_path / "db"), embedder=Counting(dim=8))
    store.create_dataset("a", metric="cosine")
    src = SourceFile(repo="a", path="x.py", lang="python", kind=Kind.CODE)
    store.add_chunks("a", chunks=[src.chunk(text="def f(): pass", start_line=1, end_line=1)])

    Counting.calls = 0
    store.search("a", query="first", k=3)
    store.search("a", query="second", k=3)
    store.search("a", query="first", k=3)

    assert Counting.calls == 3


def test_the_id_set_is_read_once_per_dataset(tmp_path: Path) -> None:
    # Dedup used to re-read the whole dataset for every batch of chunks,
    # so indexing N chunks cost O(N²/B): 200k chunks took 5.38 s against
    # 2.12 s when the set is read once. Object identity is the check —
    # a second read would build a second set.
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=8))
    store.create_dataset("a", metric="cosine")
    store.create_dataset("b", metric="cosine")
    src = SourceFile(repo="a", path="x.py", lang="python", kind=Kind.CODE)

    def write(dataset: str, i: int) -> None:
        chunk = src.chunk(text=f"def f{i}(): pass", start_line=i + 1, end_line=i + 1)
        store.add_chunks(dataset, chunks=[chunk])

    write("a", 0)
    held = store._ids_of
    for i in range(1, 5):
        write("a", i)

    assert store._ids_of is held
    assert len(store.chunk_ids("a")) == 5

    # One dataset at a time: the pipeline indexes repo by repo, so this
    # holds one repo's worth of ids rather than the sum over all of them.
    write("b", 0)
    assert store._ids_of is not held


def test_a_deleted_chunk_can_be_written_again_in_the_same_run(tmp_path: Path) -> None:
    # The hazard the kept set introduces: it would still claim to hold an
    # id the delete just removed, and the re-index would silently skip
    # writing the chunk back.
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=8))
    store.create_dataset("a", metric="cosine")
    src = SourceFile(repo="a", path="x.py", lang="python", kind=Kind.CODE)
    chunk = src.chunk(text="def f(): pass", start_line=1, end_line=1)

    store.add_chunks("a", chunks=[chunk])
    store.delete_chunks("a", ids=[chunk.id])

    assert store.add_chunks("a", chunks=[chunk]) == 1
    assert store.chunk_ids("a") == {chunk.id}


def test_refresh_drops_both_memos(tmp_path: Path) -> None:
    # A long-lived server calls refresh before it reads; anything
    # remembered from before must not survive it.
    store = LanceDBStore(str(tmp_path / "db"), embedder=FakeEmbedder(dim=8))
    store.create_dataset("a", metric="cosine")
    src = SourceFile(repo="a", path="x.py", lang="python", kind=Kind.CODE)
    store.add_chunks("a", chunks=[src.chunk(text="def f(): pass", start_line=1, end_line=1)])
    store.search("a", query="anything", k=1)
    assert store._query_memo is not None
    assert store._ids_of is not None

    store.refresh()

    assert store._query_memo is None
    assert store._ids_of is None


# --- what the embedder reads, which is not what the store keeps ----------


def test_a_chunk_is_embedded_with_its_own_name_and_place() -> None:
    # Until this existed the vector knew nothing a reader has before they
    # open a file: what it is called and what the definition is named.
    # Measured on 84 blind questions, adding both took identifier answers
    # in the top three from 11 to 15 and descriptive ones from 0 to 2.
    source = SourceFile(repo="r", path="ingest/text_chunker.py", lang="python", kind=Kind.CODE)
    chunk = source.chunk(text="def go():\n    return 1", start_line=1, end_line=2, symbol="go")

    read = retrieval_text(chunk)

    assert read.startswith("go ingest text chunker py\n")
    assert read.endswith(chunk.text)


def test_the_separators_a_sentence_model_cannot_read_become_spaces() -> None:
    # `ingest/text_chunker.py` tokenises into fragments; `ingest text
    # chunker py` into words. Symbol alone moved nothing and path alone
    # moved one answer — together they moved six, which is why both are
    # here and why they are spelled out rather than pasted in raw.
    source = SourceFile(repo="r", path="a-b/c_d.e.py", lang="python", kind=Kind.CODE)
    chunk = source.chunk(text="x = 1", start_line=1, end_line=1)

    assert retrieval_text(chunk).startswith("a b c d e py\n")


def test_what_the_store_keeps_is_untouched_by_what_the_embedder_reads() -> None:
    # The promise this had to not break: stored text is a verbatim slice
    # of its line range, because a hit points at `file:line` and the two
    # must agree. And the id hashes that text with the path, so neither
    # moves — an index does not have to be rebuilt to be readable, only
    # to be re-ranked.
    source = SourceFile(repo="r", path="src/thing.py", lang="python", kind=Kind.CODE)
    chunk = source.chunk(text="def go():\n    pass", start_line=7, end_line=8, symbol="go")

    assert chunk.text == "def go():\n    pass"
    assert chunk.id == Chunk.chunk_id("def go():\n    pass", path="src/thing.py")
    assert retrieval_text(chunk) != chunk.text


def test_a_chunk_with_nothing_to_call_it_is_embedded_as_it_was() -> None:
    # A commit chunk has no symbol worth reading and a synthetic path; a
    # nameless chunk must come out exactly as before, or this would move
    # vectors it has nothing to say about.
    source = SourceFile(repo="r", path="", lang="text", kind=Kind.DOC)
    chunk = source.chunk(text="plain words", start_line=1, end_line=1)

    assert retrieval_text(chunk) == "plain words"


# --- the lexical arm and rank fusion --------------------------------------


@pytest.fixture
def hybrid(tmp_path: Path) -> LanceDBStore:
    s = LanceDBStore(str(tmp_path / "hy"), embedder=FakeEmbedder(dim=DIM), hybrid=True)
    s.create_dataset("ds", metric="cosine")
    return s


def test_a_literal_is_found_by_the_lexical_arm(hybrid: LanceDBStore) -> None:
    # What the whole thing is for. The fake embedder's vectors carry no
    # meaning, so anything found here was found by BM25 — which is the
    # point: a person pasting a string out of a stack trace is served by
    # matching, not by similarity.
    hybrid.add_chunks(
        "ds",
        chunks=[make_chunk(f"padding {n}", path=f"p{n}.md") for n in range(20)]
        + [make_chunk("raise DeletedMoreThanStored(count)", path="pool.go")],
    )
    hybrid.refresh_text_index()

    found = hybrid.lexical("ds", query="DeletedMoreThanStored", k=5)

    assert found[0].metadata["path"] == "pool.go"


def test_without_the_index_neither_arm_raises(hybrid: LanceDBStore) -> None:
    # A store written before hybrid shipped has no text index, and an
    # arm that raised there would break every existing workspace on the
    # day it landed. Measured while writing this: LanceDB answers a
    # full-text query without one anyway, so the guarantee is the weaker
    # and more useful "no index, no crash" rather than "no index, no
    # results" — which is what the first version of this test asserted
    # and was wrong about.
    hybrid.add_chunks("ds", chunks=[make_chunk("a"), make_chunk("b", path="b.md")])

    assert hybrid.lexical("ds", query="a", k=2)
    assert len(hybrid.search("ds", query="a", k=2)) == 2


def test_fusion_prefers_what_both_arms_found() -> None:
    # The property rank fusion is chosen for: the arms fail differently,
    # so agreement is evidence where a single arm's confidence is not.
    # `second` is nobody's top hit and beats two chunks that are.
    from wsindex.pipeline import _fuse

    def hit(name: str) -> Hit:
        return Hit(score=1.0, native_id=name, metadata={"path": name})

    fused = _fuse([hit("first"), hit("second")], [hit("other"), hit("second")])

    assert fused[0].native_id == "second"


def test_the_lexical_arm_respects_the_dataset_boundary(tmp_path: Path) -> None:
    # One table holds every repo (ADR-7), so a BM25 pass without the
    # `dataset` predicate would answer with another repository's code —
    # a leak the vector arm cannot have because it prefilters.
    s = LanceDBStore(str(tmp_path / "two"), embedder=FakeEmbedder(dim=DIM), hybrid=True)
    s.create_dataset("mine", metric="cosine")
    s.create_dataset("theirs", metric="cosine")
    s.add_chunks("theirs", chunks=[make_chunk("DeletedMoreThanStored", path="x.go")])
    s.add_chunks("mine", chunks=[make_chunk("unrelated", path="y.go")])
    s.refresh_text_index()

    assert [h.metadata["path"] for h in s.lexical("mine", query="DeletedMoreThanStored", k=5)] == []
