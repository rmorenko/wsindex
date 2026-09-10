"""Embed package: the Embedder contract and its implementations."""

from wsindex.embed.embedder import (
    CHARS_PER_TOKEN,
    Embedder,
    FakeEmbedder,
    SentenceTransformerEmbedder,
    estimate_tokens,
)

__all__ = [
    "CHARS_PER_TOKEN",
    "Embedder",
    "FakeEmbedder",
    "SentenceTransformerEmbedder",
    "estimate_tokens",
]
