"""Tests for embedders: the FakeEmbedder contract, SentenceTransformerEmbedder
error branches on stubbed modules, and a slow real-model smoke test."""

import sys
from pathlib import Path

import pytest

from wsindex.embed import FakeEmbedder, SentenceTransformerEmbedder


class FakeST:
    def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder

    def get_embedding_dimension(self) -> None:
        return None


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


def test_missing_ml_extra_raises_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="uv sync --extra ml"):
        SentenceTransformerEmbedder(model_name="irrelevant")


def test_model_without_dim_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    with pytest.raises(RuntimeError, match="does not report an embedding dimension"):
        SentenceTransformerEmbedder(model_name="sentence-transformer")


def test_cache_folder_is_passed_through_and_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Wsindex owns its model cache under $XDG_CACHE_HOME/wsindex/models/ so
    # the composition root can control where a ~90 MB download lands (see
    # ADR-8 amendment). The embedder must both propagate the path to
    # sentence-transformers AND create the directory before use — the
    # library assumes it exists.
    import types

    captured: dict[str, object] = {}

    class CapturingST(FakeST):
        def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
            super().__init__(model_name, cache_folder=cache_folder)
            captured["cache_folder"] = cache_folder

        def get_embedding_dimension(self) -> int:  # type: ignore[override]
            return 8

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = CapturingST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    cache = tmp_path / "cache" / "wsindex" / "models"
    assert not cache.exists()
    SentenceTransformerEmbedder(model_name="irrelevant", cache_folder=cache)
    assert cache.is_dir()
    assert captured["cache_folder"] == str(cache)


def test_cache_folder_none_leaves_library_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Direct users of the embedder (outside the CLI) can skip the argument
    # and get sentence-transformers' own default (~/.cache/huggingface/hub).
    import types

    captured: dict[str, object] = {}

    class CapturingST(FakeST):
        def __init__(self, model_name: str, **kwargs: object) -> None:
            super().__init__(model_name)
            captured["kwargs"] = kwargs

        def get_embedding_dimension(self) -> int:  # type: ignore[override]
            return 8

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = CapturingST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    SentenceTransformerEmbedder(model_name="irrelevant")
    assert captured["kwargs"] == {}


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.mark.slow
def test_real_model_smoke() -> None:
    pytest.importorskip("sentence_transformers")
    emb = SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = emb.embed(["hello"])
    assert emb.dim == len(embeddings[0])
    assert sum(x * x for x in embeddings[0]) == pytest.approx(1.0)
    cat, kitten, spreadsheet = emb.embed(["cat", "kitten", "spreadsheet"])
    assert cos(cat, kitten) > cos(cat, spreadsheet)
