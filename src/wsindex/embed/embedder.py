"""Embedder contract: text batches in, vector batches out.

Stores that embed locally depend only on the Embedder ABC (the pipeline
itself works in plain text); FakeEmbedder serves tests and the offline
slice, SentenceTransformerEmbedder wraps a real model behind the optional
`ml` extra.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import cast

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
        """Embed a batch of texts.

        A query is a batch of one: `embed([query])[0]`.

        Args:
            texts: Texts to embed; a bare string is rejected even though
                `str` is itself a Sequence[str] — embedding five one-letter
                "texts" is never what the caller meant.

        Returns:
            One vector of length `dim` per text, in input order.

        Raises:
            TypeError: `texts` is a bare string instead of a batch.
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
        """Pick the vector size; no other knobs exist.

        Args:
            dim: Dimensionality of the pseudo-vectors; tests keep it small.
        """
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
    """Real semantic vectors from a sentence-transformers model.

    Needs the optional `ml` extra (`uv sync --extra ml`); the import is
    deferred to `__init__` so the module stays importable without it.
    Vectors are L2-normalized, so cosine similarity equals dot product.
    """

    def __init__(self, model_name: str, cache_folder: Path | None = None) -> None:
        """Load the model; `dim` is taken from the model itself.

        Args:
            model_name: sentence-transformers model id, e.g.
                "sentence-transformers/all-MiniLM-L6-v2".
            cache_folder: Where sentence-transformers stores downloaded
                model files. When None, the library uses its own default
                (typically `~/.cache/huggingface/hub/`); the wsindex CLI
                passes an explicit path under `$XDG_CACHE_HOME/wsindex/`
                (see `wsindex.paths.resolve_cache_dir`) so wsindex-owned
                cache is namespaced and safe to `rm -rf`.

        Raises:
            RuntimeError: The `ml` extra is not installed, or the model
                does not report an embedding dimension.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed — run `uv sync --extra ml`"
            ) from exc
        st_kwargs: dict[str, str] = {}
        if cache_folder is not None:
            cache_folder.mkdir(parents=True, exist_ok=True)
            st_kwargs["cache_folder"] = str(cache_folder)
        self._model = SentenceTransformer(model_name, **st_kwargs)
        dim: int | None = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"model {model_name!r} does not report an embedding dimension")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return cast("list[list[float]]", vectors.tolist())
