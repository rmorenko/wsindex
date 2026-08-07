"""Tests for FakeEmbedder: determinism, batch contract, input validation."""

import pytest

from wsindex.embed.embedder import FakeEmbedder


def test_same_text_same_vector_across_instances() -> None:
    # Different instances must agree: the seed comes from the text, not the object.
    a = FakeEmbedder().embed(["hello"])[0]
    b = FakeEmbedder().embed(["hello"])[0]
    assert a == b


def test_different_texts_differ() -> None:
    vecs = FakeEmbedder().embed(["hello", "world"])
    assert vecs[0] != vecs[1]


def test_batch_preserves_length_and_order() -> None:
    emb = FakeEmbedder()
    batch = emb.embed(["a", "b", "c"])
    assert len(batch) == 3
    assert batch[0] == emb.embed(["a"])[0]
    assert batch[2] == emb.embed(["c"])[0]


def test_dim_is_respected() -> None:
    assert FakeEmbedder().dim == 8
    emb = FakeEmbedder(dim=3)
    assert emb.dim == 3
    assert all(len(vec) == 3 for vec in emb.embed(["x", "y"]))


def test_vectors_are_plain_floats() -> None:
    # JSON-serializable python floats, not numpy scalars.
    vec = FakeEmbedder().embed(["hello"])[0]
    assert all(type(x) is float for x in vec)


def test_empty_batch() -> None:
    assert FakeEmbedder().embed([]) == []


def test_bare_string_rejected() -> None:
    # mypy happily accepts a bare str (it IS a Sequence[str]) — that is
    # exactly why the guard must exist at runtime.
    with pytest.raises(TypeError):
        FakeEmbedder().embed("hello")
