"""Tests for the core data model (Chunk, Hit, chunk_id)."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from wsindex.model import Chunk, Hit, Kind, SourceFile


def make_chunk(**overrides: Any) -> Chunk:
    """Factory with sensible defaults; tests override only what they check."""
    defaults: dict[str, Any] = {
        "repo": "mcp",
        "path": "server.py",
        "lang": "python",
        "kind": Kind.CODE,
        "symbol": "f",
        "node_type": "function_definition",
        "start_line": 1,
        "end_line": 2,
        "text": "def f(): pass",
    }
    return Chunk(**(defaults | overrides))


def test_kind_serializes_as_plain_string() -> None:
    # The tensor `metadata` payload must contain a plain string, not an enum
    # member; StrEnum guarantees this equality holds.
    md = make_chunk().to_metadata()
    assert md["kind"] == "code"


def test_chunk_id_is_deterministic() -> None:
    first = Chunk.chunk_id("def f(): pass", path="a.py")
    second = Chunk.chunk_id("def f(): pass", path="a.py")
    assert first == second
    assert len(first) == 64  # sha256 hex digest


def test_different_text_gives_different_id() -> None:
    assert Chunk.chunk_id("def f(): pass", path="a.py") != Chunk.chunk_id(
        "def g(): pass", path="a.py"
    )


def test_different_path_gives_different_id() -> None:
    assert Chunk.chunk_id("def f(): pass", path="a.py") != Chunk.chunk_id(
        "def f(): pass", path="b.py"
    )


def test_boundary_shift_gives_different_id() -> None:
    # Naive concatenation would hash both pairs as "abc"; length-prefixed
    # hashing must keep them distinct.
    assert Chunk.chunk_id("ab", path="c") != Chunk.chunk_id("a", path="bc")


def test_chunk_id_computed_on_creation() -> None:
    c = make_chunk()
    assert c.id == Chunk.chunk_id(c.text, path=c.path)


def test_chunk_is_frozen() -> None:
    c = make_chunk()
    with pytest.raises(FrozenInstanceError):
        c.text = "changed"  # type: ignore[misc]


def test_to_metadata_contains_all_fields() -> None:
    c = make_chunk()
    assert c.to_metadata() == {
        "id": Chunk.chunk_id("def f(): pass", path="server.py"),
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


def test_a_hit_carries_the_only_name_a_chunk_has() -> None:
    # `repo/path:lines` locates a chunk for a person; one file yields
    # many, and re-indexing moves the boundaries. The id is what dedup
    # and every link are keyed on, and it was the one thing an outside
    # caller could not say back.
    src = SourceFile(repo="r", path="a.py", lang="python", kind=Kind.CODE)
    chunk = src.chunk(text="def f(): pass", start_line=1, end_line=1)
    hit = Hit(score=0.5, metadata=chunk.to_metadata(), native_id=chunk.id)

    assert hit.to_json()["id"] == chunk.id
