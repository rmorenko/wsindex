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
from typing import Any, cast

import numpy as np

CHARS_PER_TOKEN = 3
"""Divisor for estimating a token count without a tokeniser.

Measured rather than guessed: real chunks run at 3.56 characters per
token in Python and 4.00 in PHP, with a tenth percentile of 2.81. Three
sits below that, so the estimate errs **high** — a caller asking for four
thousand tokens gets a little less than it could have had rather than
more than it asked for. That is the direction that matters when the
caller is an agent filling a context window it cannot exceed."""


def estimate_tokens(text: str) -> int:
    """A token count for callers that have no tokeniser to hand."""
    return -(-len(text) // CHARS_PER_TOKEN)


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

    def count_tokens(self, text: str) -> int:
        """How many tokens this text costs a caller with a budget.

        Estimated here and exact in whichever backend has a tokeniser.
        """
        return estimate_tokens(text)


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

    def __init__(
        self, model_name: str, cache_folder: Path | None = None, dim: int | None = None
    ) -> None:
        """Prepare an embedder; the model itself loads on first use.

        Lazily, because loading is 6 seconds and not every command that
        opens a store goes on to embed anything — `wsindex compact` took
        9.7 s on an empty index, all of it a neural network it never
        called.

        The model is named by the caller rather than read from the
        workspace config: an embedder is the bottom of the stack and
        should not know that an application configuration exists.

        Args:
            model_name: sentence-transformers model id, e.g.
                "sentence-transformers/all-MiniLM-L6-v2".
            cache_folder: Where sentence-transformers keeps downloaded
                models. None lets the library use its own default; the
                CLI passes a path under `$XDG_CACHE_HOME/wsindex/` so the
                cache is namespaced and safe to `rm -rf`.
            dim: What the workspace says this model's vectors are. Given,
                `dim` answers without loading anything — which is what
                keeps a command that never embeds from paying for a
                model. The claim is checked against the real model the
                first time it loads.

        Nothing is imported here either. Checking for the `ml` extra
        meant importing torch, which is 1.9 s — paid by `compact`, which
        needs no model, and by every other command that opens a store.
        A missing extra is now reported when something asks to embed,
        which is the moment it actually matters.
        """
        self._model_name = model_name
        self._cache_folder = cache_folder
        self._declared_dim = dim
        self._loaded: Any = None

    def _model(self) -> Any:
        """The loaded model, loading it the first time it is wanted.

        Tries the on-disk cache alone before letting the library reach
        the network. That check costs 4 of the 6 seconds a load takes —
        the model is already local and the request only confirms its
        revision — and it is paid on every command that embeds anything.
        A machine that has never downloaded the model falls through to
        the ordinary path and downloads it, once.

        Raises:
            RuntimeError: The `ml` extra is not installed, the model
                reports no dimension, or its width is not the one this
                workspace was built for.
        """
        if self._loaded is not None:
            return self._loaded
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed — run `uv sync --extra ml`"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._cache_folder is not None:
            self._cache_folder.mkdir(parents=True, exist_ok=True)
            kwargs["cache_folder"] = str(self._cache_folder)
        try:
            self._loaded = SentenceTransformer(self._model_name, local_files_only=True, **kwargs)
        except Exception:
            self._loaded = SentenceTransformer(self._model_name, **kwargs)

        actual: int | None = self._loaded.get_embedding_dimension()
        if actual is None:
            raise RuntimeError(f"model {self._model_name!r} does not report an embedding dimension")
        if self._declared_dim is not None and actual != self._declared_dim:
            raise RuntimeError(
                f"model {self._model_name!r} produces {actual}-dimensional vectors, "
                f"but this workspace was built for {self._declared_dim}"
            )
        self._declared_dim = actual
        return self._loaded

    @property
    def dim(self) -> int:
        """Vector width, from the workspace when it said, else the model."""
        if self._declared_dim is None:
            self._model()
        assert self._declared_dim is not None
        return self._declared_dim

    def count_tokens(self, text: str) -> int:
        """Exactly what the model will read, since this one has its tokeniser."""
        return len(self._model().tokenizer.encode(text, add_special_tokens=False))

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model().encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return cast("list[list[float]]", vectors.tolist())
