"""Tests for embedders: the FakeEmbedder contract, SentenceTransformerEmbedder
error branches on stubbed modules, and a slow real-model smoke test."""

import sys
from pathlib import Path

import numpy as np
import pytest

from wsindex.embed import FakeEmbedder, SentenceTransformerEmbedder


class FakeST:
    """Stands in for `SentenceTransformer`: constructed, asked, encoded."""

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def get_embedding_dimension(self) -> int | None:
        return None

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.zeros((len(texts), 8), dtype=np.float32)


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


def test_missing_ml_extra_is_reported_when_something_embeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not at construction: checking for the extra means importing torch,
    # which is 1.9 s, and a command that opens a store without embedding
    # should not pay it. `wsindex compact` went from 9.7 s to 0.9 s.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = SentenceTransformerEmbedder(model_name="irrelevant", dim=8)

    assert embedder.dim == 8  # still answerable without the library

    with pytest.raises(RuntimeError, match="uv sync --extra ml"):
        embedder.embed(["x"])


def test_model_without_dim_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    with pytest.raises(RuntimeError, match="does not report an embedding dimension"):
        SentenceTransformerEmbedder(model_name="sentence-transformer").embed(["x"])


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
        def __init__(self, model_name: str, **kwargs: object) -> None:
            super().__init__(model_name, **kwargs)
            captured["cache_folder"] = kwargs.get("cache_folder")

        def get_embedding_dimension(self) -> int:
            return 8

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = CapturingST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    cache = tmp_path / "cache" / "wsindex" / "models"
    assert not cache.exists()
    SentenceTransformerEmbedder(model_name="irrelevant", cache_folder=cache).embed(["x"])
    assert cache.is_dir()
    assert captured["cache_folder"] == str(cache)


def test_cache_folder_none_leaves_library_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Direct users of the embedder (outside the CLI) can skip the argument
    # and get sentence-transformers' own default (~/.cache/huggingface/hub).
    import types

    captured: dict[str, object] = {}

    class CapturingST(FakeST):
        def __init__(self, model_name: str, **kwargs: object) -> None:
            super().__init__(model_name, **kwargs)
            captured["kwargs"] = kwargs

        def get_embedding_dimension(self) -> int:
            return 8

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = CapturingST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    SentenceTransformerEmbedder(model_name="irrelevant").embed(["x"])

    # `local_files_only` is ours; a cache folder is not passed at all.
    assert captured["kwargs"] == {"local_files_only": True}


def test_the_model_name_is_the_callers_business(monkeypatch: pytest.MonkeyPatch) -> None:
    # The embedder is the bottom of the stack: it takes the name it is
    # given and knows nothing about a workspace configuration. The
    # composition root is what reads `config.model`.
    import types

    captured: dict[str, object] = {}

    class CapturingST(FakeST):
        def __init__(self, model_name: str, **kwargs: object) -> None:
            super().__init__(model_name, **kwargs)
            captured["model_name"] = model_name

        def get_embedding_dimension(self) -> int:
            return 8

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = CapturingST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    SentenceTransformerEmbedder("some/model").embed(["x"])

    assert captured["model_name"] == "some/model"


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


def _fake_module(monkeypatch: pytest.MonkeyPatch, cls: object) -> None:
    import types

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_nothing_loads_until_something_embeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # `wsindex compact` opens a store and never embeds a thing. It used to
    # spend 9.7 seconds on an empty index loading a model it never called.
    loads: list[str] = []

    class Counting(FakeST):
        def __init__(self, model_name: str, **kwargs: object) -> None:
            super().__init__(model_name, **kwargs)
            loads.append(model_name)

        def get_embedding_dimension(self) -> int:
            return 8

    _fake_module(monkeypatch, Counting)
    embedder = SentenceTransformerEmbedder("some/model", dim=8)

    assert embedder.dim == 8  # answered by the workspace, not the model
    assert loads == []

    embedder.embed(["now"])
    assert loads == ["some/model"]


def test_the_cache_is_tried_before_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # Confirming a revision the model already has locally costs 4 of the 6
    # seconds a load takes, on every command that embeds anything.
    attempts: list[dict[str, object]] = []

    class Recording(FakeST):
        def __init__(self, model_name: str, **kwargs: object) -> None:
            super().__init__(model_name, **kwargs)
            attempts.append(kwargs)

        def get_embedding_dimension(self) -> int:
            return 8

    _fake_module(monkeypatch, Recording)
    SentenceTransformerEmbedder("some/model").embed(["x"])

    assert attempts == [{"local_files_only": True}]


def test_a_model_that_is_not_cached_yet_is_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    # A machine that has never seen the model must still work: the local
    # attempt fails, and the ordinary path downloads it once.
    attempts: list[dict[str, object]] = []

    class OnlyOnline(FakeST):
        def __init__(self, model_name: str, **kwargs: object) -> None:
            attempts.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("nothing in the cache")
            super().__init__(model_name, **kwargs)

        def get_embedding_dimension(self) -> int:
            return 8

    _fake_module(monkeypatch, OnlyOnline)
    SentenceTransformerEmbedder("some/model").embed(["x"])

    assert attempts == [{"local_files_only": True}, {}]


def test_a_model_of_the_wrong_width_says_both_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    # The workspace was built for one width; swapping the model without
    # changing `dim` would silently write vectors the store cannot use.
    class Wide(FakeST):
        def get_embedding_dimension(self) -> int:
            return 768

    _fake_module(monkeypatch, Wide)
    embedder = SentenceTransformerEmbedder("wide/model", dim=384)

    with pytest.raises(RuntimeError, match=r"768.*384"):
        embedder.embed(["x"])


def test_a_question_and_a_passage_are_not_the_same_call() -> None:
    # The contract used to say "a query is a batch of one", which is a
    # claim about the model rather than a shortcut. For an asymmetric
    # model the two vectors differ by the instruction the model was
    # trained to see on the question side.
    class Asymmetric(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return self.embed(["QUERY: " + query])[0]

    embedder = Asymmetric(dim=8)

    assert embedder.embed_query("where do we retry") != embedder.embed(["where do we retry"])[0]


def test_a_symmetric_embedder_is_untouched_by_the_new_contract() -> None:
    # Every existing embedder must keep answering exactly as it did, or
    # this change would silently rewrite what is already in a store.
    embedder = FakeEmbedder(dim=8)

    assert embedder.embed_query("anything") == embedder.embed(["anything"])[0]
