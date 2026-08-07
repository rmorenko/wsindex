"""Tests for LocalStore: layout, dedup by chunk id, cosine ordering, persistence."""

from pathlib import Path

import pytest

from wsindex.model import Chunk, Kind
from wsindex.store.local import LocalStore

DIM = 3

VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]
NEAR_A = [0.9, 0.1, 0.0]


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
    s = LocalStore(tmp_path / ".wsindex")
    s.create("ds", dim=DIM, metric="cosine")
    return s


def test_upsert_then_search_nearest_first(store: LocalStore) -> None:
    a, b = make_chunk("a"), make_chunk("b")
    assert store.upsert("ds", [a, b], [VEC_A, VEC_B]) == 2
    hits = store.search("ds", NEAR_A, k=2)
    assert [h.metadata["text"] for h in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score
    assert hits[0].native_id == a.id
    assert hits[0].metadata["path"] == "doc.md"


def test_upsert_skips_duplicates(store: LocalStore) -> None:
    a = make_chunk("a")
    assert store.upsert("ds", [a], [VEC_A]) == 1
    # Same chunk again: nothing written, store did not grow.
    assert store.upsert("ds", [a], [VEC_A]) == 0
    assert len(store.search("ds", NEAR_A, k=10)) == 1
    # Duplicate inside a single batch counts once.
    b = make_chunk("b")
    assert store.upsert("ds", [b, b], [VEC_B, VEC_B]) == 1
    assert len(store.search("ds", NEAR_A, k=10)) == 2


def test_incremental_upsert_keeps_old_vectors(store: LocalStore) -> None:
    a, b = make_chunk("a"), make_chunk("b")
    assert store.upsert("ds", [a], [VEC_A]) == 1
    assert store.upsert("ds", [b], [VEC_B]) == 1
    hits = store.search("ds", NEAR_A, k=10)
    assert len(hits) == 2
    assert hits[0].metadata["text"] == "a"


def test_upsert_length_mismatch_raises(store: LocalStore) -> None:
    with pytest.raises(ValueError):
        store.upsert("ds", [make_chunk("a"), make_chunk("b")], [VEC_A])


def test_create_is_idempotent(store: LocalStore) -> None:
    store.upsert("ds", [make_chunk("a")], [VEC_A])
    store.create("ds", dim=DIM, metric="cosine")  # same params: no-op
    assert len(store.search("ds", NEAR_A, k=10)) == 1  # data survived


def test_create_conflicting_params_raise(store: LocalStore) -> None:
    with pytest.raises(ValueError):
        store.create("ds", dim=DIM + 1, metric="cosine")
    with pytest.raises(ValueError):
        store.create("other", dim=DIM, metric="dot")


def test_search_empty_dataset(store: LocalStore) -> None:
    assert store.search("ds", NEAR_A, k=5) == []


def test_search_unknown_dataset_raises(store: LocalStore) -> None:
    with pytest.raises(ValueError):
        store.search("nope", NEAR_A, k=5)


def test_search_wrong_query_dim_raises(store: LocalStore) -> None:
    store.upsert("ds", [make_chunk("a")], [VEC_A])
    with pytest.raises(ValueError):
        store.search("ds", [1.0, 0.0], k=5)


def test_k_larger_than_store(store: LocalStore) -> None:
    store.upsert("ds", [make_chunk("a")], [VEC_A])
    assert len(store.search("ds", NEAR_A, k=50)) == 1


def test_persistence_across_instances(store: LocalStore) -> None:
    a = make_chunk("a")
    store.upsert("ds", [a], [VEC_A])
    reopened = LocalStore(store.root)
    hits = reopened.search("ds", NEAR_A, k=5)
    assert [h.native_id for h in hits] == [a.id]
