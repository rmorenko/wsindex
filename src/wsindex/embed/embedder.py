"""Embedder contract: text batches in, vector batches out.

The pipeline depends only on the Embedder ABC; FakeEmbedder serves tests and
the offline slice, the real sentence-transformers model arrives in plan
stage 5.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Embedder(ABC):
    """Turns texts into vectors of a fixed dimensionality.

    Template method: the public `embed` validates input once for every
    backend and delegates the actual work to the abstract `_embed`.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of produced vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch: one vector of length `dim` per text, same order.

        A query is a batch of one: `embed([query])[0]`. Raises TypeError on
        a bare string — `str` is itself a Sequence[str], and embedding five
        one-letter "texts" is never what the caller meant.
        """
        if isinstance(texts, str):
            raise TypeError("expected a batch of texts, got a single str")
        return self._embed(texts)

    @abstractmethod
    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Do the embedding; override this, input is already validated."""


class FakeEmbedder(Embedder):
    """Deterministic pseudo-vectors for tests and the offline slice.

    Same text always yields the same vector (sha256 of the text seeds a
    local RNG), but the vectors carry no semantics: similar texts are NOT
    close. Tests that need closeness build vectors by hand or reuse
    identical texts.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=(self.dim,))
            result.append(vec.tolist())
        return result


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed — run `uv sync --extra ml`"
            ) from exc
        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return self._model.get_embedding_dimension() if self._model.get_embedding_dimension() else 0

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return list(vectors.tolist())
