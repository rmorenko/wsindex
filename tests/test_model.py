"""Tests for the core data model (Chunk, Hit, chunk_id)."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from wsindex.model import Chunk


def make_chunk(**overrides: Any) -> Chunk:
    """Factory with sensible defaults; tests override only what they check."""
    defaults: dict[str, Any] = {
        "repo": "mcp",
        "path": "server.py",
        "lang": "python",
        "kind": "code",
        "symbol": "f",
        "node_type": "function_definition",
        "start_line": 1,
        "end_line": 2,
        "text": "def f(): pass",
    }
    return Chunk(**(defaults | overrides))


def test_chunk_id_is_deterministic() -> None:
    first = Chunk.chunk_id("def f(): pass", "a.py")
    second = Chunk.chunk_id("def f(): pass", "a.py")
    assert first == second
    assert len(first) == 64  # sha256 hex digest


def test_different_text_gives_different_id() -> None:
    assert Chunk.chunk_id("def f(): pass", "a.py") != Chunk.chunk_id("def g(): pass", "a.py")


def test_different_path_gives_different_id() -> None:
    assert Chunk.chunk_id("def f(): pass", "a.py") != Chunk.chunk_id("def f(): pass", "b.py")


def test_boundary_shift_gives_different_id() -> None:
    # Naive concatenation would hash both pairs as "abc"; length-prefixed
    # hashing must keep them distinct.
    assert Chunk.chunk_id("ab", "c") != Chunk.chunk_id("a", "bc")


def test_chunk_id_computed_on_creation() -> None:
    c = make_chunk()
    assert c.id == Chunk.chunk_id(c.text, c.path)


def test_chunk_is_frozen() -> None:
    c = make_chunk()
    with pytest.raises(FrozenInstanceError):
        c.text = "changed"  # type: ignore[misc]


def test_to_metadata_contains_all_fields() -> None:
    c = make_chunk()
    assert c.to_metadata() == {
        "id": Chunk.chunk_id("def f(): pass", "server.py"),
        "repo": "mcp",
        "path": "server.py",
        "lang": "python",
        "kind": "code",
        "symbol": "f",
        "node_type": "function_definition",
        "start_line": 1,
        "end_line": 2,
        "text": "def f(): pass",
    }
