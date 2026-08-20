"""Tests for LanceDBStore: the single-table-with-repo-column layout (ADR-7).

Ported from the LocalStore suite where the contract is backend-agnostic;
reworked where the mechanics changed (a missing dataset is a registry
miss, not a missing directory; a foreign embedder dim fails at
construction, not at search). New, mandated by ADR-7: prefilter parity,
deterministic tie-breaking, KNN == numpy brute force.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from wsindex.embed.embedder import FakeEmbedder
from wsindex.model import Chunk, Kind
from wsindex.store.lancedb import LanceDBStore

DIM = 8


class CountingFake(FakeEmbedder):
    """FakeEmbedder that records every text it was asked to embed."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__(dim=dim)
        self.embedded: list[str] = []

    def _embed(self, texts: "Sequence[str]") -> list[list[float]]:
        self.embedded.extend(texts)
        return super()._embed(texts)


def make_chunk(text: str, path: str = "doc.md") -> Chunk:
    return Chunk(
        repo="r",
        path=path,
        lang="text",
        kind=Kind.DOC,
        symbol=None,
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
    hits = store.search("ds", query="a", k=2)
    assert [h.metadata["text"] for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0)  # exact text match embeds identically
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
    assert embedder.embedded == ["a"]  # batch-internal duplicate embedded once
    store.add_chunks("ds", chunks=[a])
    assert embedder.embedded == ["a"]  # second run embedded nothing


def test_incremental_add_chunks_keeps_old_vectors(store: LanceDBStore) -> None:
    a, b = make_chunk("a"), make_chunk("b")
    assert store.add_chunks("ds", chunks=[a]) == 1
    assert store.add_chunks("ds", chunks=[b]) == 1
    hits = store.search("ds", query="a", k=10)
    assert len(hits) == 2
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
    vecs = np.array(store.embedder.embed(texts))
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
