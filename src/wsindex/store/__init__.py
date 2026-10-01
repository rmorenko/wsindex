"""Store package: the VectorStore contract and its backend implementations."""

from wsindex.store.base import VectorStore
from wsindex.store.local import LocalStore
from wsindex.store.tensorus import TensorusStore

__all__ = ["LocalStore", "TensorusStore", "VectorStore"]
