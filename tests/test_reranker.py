import sys
from unittest.mock import Mock

import numpy as np
import pytest

from wsindex.rank.reranker import CrossEncoderReranker, FakeReranker


def test_fake_is_deterministic() -> None:
    a = FakeReranker().rank(query="hello world", texts=["hello", "world"])
    b = FakeReranker().rank(query="hello world", texts=["hello", "world"])
    assert a == b


def test_fake_different_texts_get_different_scores() -> None:
    scores = FakeReranker().rank(query="q", texts=["alpha", "beta"])
    assert scores[0] != scores[1]


def test_fake_scores_are_in_unit_interval() -> None:
    a = FakeReranker().rank(query="hello world", texts=["hello", "world"])
    assert 0 <= a[0] <= 1
    assert 0 <= a[1] <= 1


def test_empty_texts() -> None:
    a = FakeReranker().rank(query="hello world", texts=[])
    assert len(a) == 0


def test_missing_ml_extra_raises_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="uv sync --extra ml"):
        CrossEncoderReranker(model_name="irrelevant")


def test_bare_string_texts_is_rejected() -> None:
    with pytest.raises(TypeError, match="batch of texts"):
        FakeReranker().rank(query="q", texts="abc")


def test_cross_encoder_rank_builds_pairs_and_returns_floats() -> None:
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)  # bypass __init__
    r._model = Mock()
    r._model.predict.return_value = np.array([0.9, 0.1])
    scores = r.rank(query="q", texts=["a", "b"])
    r._model.predict.assert_called_once_with([("q", "a"), ("q", "b")])
    assert scores == [pytest.approx(0.9), pytest.approx(0.1)]


def test_cross_encoder_empty_texts_skips_the_model() -> None:
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)
    r._model = Mock()
    assert r.rank(query="q", texts=[]) == []
    r._model.predict.assert_not_called()
