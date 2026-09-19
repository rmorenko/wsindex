"""Embed package: the Embedder contract and its implementations."""

from wsindex.embed.embedder import Embedder, FakeEmbedder, SentenceTransformerEmbedder

__all__ = ["Embedder", "FakeEmbedder", "SentenceTransformerEmbedder"]
