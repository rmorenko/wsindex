"""Store package: the VectorStore contract and its backend implementations."""

from wsindex.store.base import VectorStore
from wsindex.store.lancedb import LanceDBStore

__all__ = ["LanceDBStore", "VectorStore"]
