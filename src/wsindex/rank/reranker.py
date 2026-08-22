"""Reranker contract: (query, texts) in, per-text scores out.

Second stage of the search funnel (ARCH stage 7): the bi-encoder store
retrieves top-N cheaply, the reranker rescores those N with a cross-
encoder that sees each (query, text) pair together — more precise but
too expensive to run on the whole dataset. Fake for tests, real one
behind the `ml` extra.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Reranker(ABC):
    """Turns (query, texts) into a score per text. Template method: `rank`
    validates once for every backend and delegates to `_rank`."""

    def rank(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score `texts` against `query`; scores are returned in input order.

        Args:
            query: The query text.
            texts: Texts to score; a bare string is rejected even though
                `str` is itself a Sequence[str] — scoring five one-letter
                "texts" is never what the caller meant.

        Returns:
            One float per text, in input order; empty list for empty input.

        Raises:
            TypeError: `texts` is a bare string instead of a batch.
        """
        if isinstance(texts, str):
            raise TypeError("expected a batch of texts, got a single str")
        return self._rank(query, texts)

    @abstractmethod
    def _rank(self, query: str, texts: Sequence[str]) -> list[float]:
        """Do the scoring; override this, input is already validated."""


class FakeReranker(Reranker):
    """Deterministic pseudo-scores for tests. Same (query, text) always
    yields the same float in [0, 1), but the scores carry no semantics:
    similar (query, text) pairs are NOT ranked higher than unrelated
    ones. Tests that need ranking quality use the real cross-encoder.
    """

    def _rank(self, query: str, texts: Sequence[str]) -> list[float]:
        result: list[float] = []
        for text in texts:
            h = hashlib.sha256()
            h.update(query.encode())
            h.update(len(query).to_bytes(8, "big"))
            h.update(text.encode())
            h.update(len(text).to_bytes(8, "big"))
            seed = int.from_bytes(h.digest()[:8], "big")
            rng = np.random.default_rng(seed)
            result.append(rng.random())
        return result


class CrossEncoderReranker(Reranker):
    """Real semantic scores from a sentence-transformers CrossEncoder.

    Needs the optional `ml` extra (`uv sync --extra ml`); the import is
    deferred to `__init__` so the module stays importable without it.
    Wrapped with a Sigmoid activation so scores are in [0, 1] and
    comparable to the bi-encoder cosine scale (see ADR-... / step 18).
    """

    def __init__(self, model_name: str) -> None:
        """Load the cross-encoder; sigmoid activation is applied.

        Args:
            model_name: sentence-transformers cross-encoder id, e.g.
                "cross-encoder/ms-marco-MiniLM-L6-v2".

        Raises:
            RuntimeError: The `ml` extra is not installed.
        """
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed — run `uv sync --extra ml`"
            ) from exc
        self._model = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid())

    def _rank(self, query: str, texts: Sequence[str]) -> list[float]:
        # pragma: no cover
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self._model.predict(pairs)]
