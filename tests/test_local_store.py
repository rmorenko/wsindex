"""Tests for LocalStore: layout, dedup by chunk id, cosine ordering, persistence.

The store owns embedding now: tests feed texts and let the injected
FakeEmbedder produce vectors. Same text always embeds to the same vector,
so searching for a stored text must return it with a perfect score.
"""

from pathlib import Path

import pytest

from wsindex.embed.embedder import FakeEmbedder
from wsindex.model import Chunk, Kind
from wsindex.store.local import LocalStore

DIM = 3


class CountingFake(FakeEmbedder):
    """FakeEmbedder that records every text it was asked to embed."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__(dim=dim)
        self.embedded: list[str] = []

    def _embed(self, texts: "list[str]") -> list[list[float]]:  # type: ignore[override]
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
def store(tmp_path: Path) -> LocalStore:
    s = LocalStore(root=tmp_path / ".wsindex", embedder=FakeEmbedder(dim=DIM))
    s.create("ds", metric="cosine")
    return s


def test_upsert_then_search_nearest_first(store: LocalStore) -> None:
    a, b = make_chunk("a"), make_chunk("b")
    assert store.upsert("ds", [a, b]) == 2
    hits = store.search("ds", "a", k=2)
    assert [h.metadata["text"] for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0)  # exact text match embeds identically
    assert hits[0].score > hits[1].score
    assert hits[0].native_id == a.id
    assert hits[0].metadata["path"] == "doc.md"


def test_upsert_skips_duplicates(store: LocalStore) -> None:
    a = make_chunk("a")
    assert store.upsert("ds", [a]) == 1
    # Same chunk again: nothing written, store did not grow.
    assert store.upsert("ds", [a]) == 0
    assert len(store.search("ds", "a", k=10)) == 1
    # Duplicate inside a single batch counts once.
    b = make_chunk("b")
    assert store.upsert("ds", [b, b]) == 1
    assert len(store.search("ds", "a", k=10)) == 2


def test_duplicates_are_not_reembedded(tmp_path: Path) -> None:
    # The point of moving embedding into the store: dedup happens BEFORE
    # the (expensive) embedding call, so a re-index embeds nothing.
    embedder = CountingFake()
    store = LocalStore(root=tmp_path / ".wsindex", embedder=embedder)
    store.create("ds", metric="cosine")
    a = make_chunk("a")
    store.upsert("ds", [a])
    assert embedder.embedded == ["a"]
    store.upsert("ds", [a])
    assert embedder.embedded == ["a"]  # second run embedded nothing


def test_incremental_upsert_keeps_old_vectors(store: LocalStore) -> None:
    a, b = make_chunk("a"), make_chunk("b")
    assert store.upsert("ds", [a]) == 1
    assert store.upsert("ds", [b]) == 1
    hits = store.search("ds", "a", k=10)
    assert len(hits) == 2
    assert hits[0].metadata["text"] == "a"


def test_create_is_idempotent(store: LocalStore) -> None:
    store.upsert("ds", [make_chunk("a")])
    store.create("ds", metric="cosine")  # same embedder dim + metric: no-op
    assert len(store.search("ds", "a", k=10)) == 1  # data survived


def test_create_conflicting_params_raise(store: LocalStore) -> None:
    other_dim = LocalStore(root=store.root, embedder=FakeEmbedder(dim=DIM + 1))
    with pytest.raises(ValueError):
        other_dim.create("ds", metric="cosine")
    with pytest.raises(ValueError):
        store.create("other", metric="dot")


def test_search_empty_dataset(store: LocalStore) -> None:
    assert store.search("ds", "anything", k=5) == []


def test_search_unknown_dataset_raises(store: LocalStore) -> None:
    with pytest.raises(ValueError):
        store.search("nope", "anything", k=5)


def test_search_with_changed_embedder_dim_raises(store: LocalStore) -> None:
    # The on-disk index was built with DIM; switching the provider/model
    # without re-indexing must fail loudly, not return garbage scores.
    store.upsert("ds", [make_chunk("a")])
    stale = LocalStore(root=store.root, embedder=FakeEmbedder(dim=DIM - 1))
    with pytest.raises(ValueError):
        stale.search("ds", "a", k=5)


def test_k_larger_than_store(store: LocalStore) -> None:
    store.upsert("ds", [make_chunk("a")])
    assert len(store.search("ds", "a", k=50)) == 1


def test_persistence_across_instances(store: LocalStore) -> None:
    a = make_chunk("a")
    store.upsert("ds", [a])
    # A fresh embedder instance embeds the same text identically, so a
    # reopened store finds what the first one wrote.
    reopened = LocalStore(root=store.root, embedder=FakeEmbedder(dim=DIM))
    hits = reopened.search("ds", "a", k=5)
    assert [h.native_id for h in hits] == [a.id]
