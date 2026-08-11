"""Tests for the Python AST chunker: spans, symbols and the coverage invariant.

Parser-dependent tests are skipped on a base install (no `ast` extra); the
HAS_TREE_SITTER guard test runs everywhere because it never touches the
parser.
"""

import pytest

from wsindex.ingest.ast_chunker import HAS_TREE_SITTER, chunk_python
from wsindex.model import Chunk, Kind

requires_tree_sitter = pytest.mark.skipif(
    not HAS_TREE_SITTER, reason="needs the ast extra (tree-sitter)"
)

SAMPLE = '''\
"""Module docstring."""

import os

MAX_SIZE = 100  # a constant


@decorator(arg=1)
def top_func(x):
    """Doc of top_func."""
    return x + MAX_SIZE


class Greeter:
    """Doc of Greeter."""

    default_name = "world"

    def greet(self, name):
        return f"hello {name}"

    @property
    def loud(self):
        return self.greet(self.default_name).upper()


# trailing comment
TAIL = 1
'''


def _chunk(text: str) -> list[Chunk]:
    return chunk_python(text, repo="r", path="sample.py", lang="python", kind=Kind.CODE)


@requires_tree_sitter
def test_sample_spans_symbols_and_node_types() -> None:
    got = [(c.start_line, c.end_line, c.symbol, c.node_type) for c in _chunk(SAMPLE)]
    assert got == [
        (1, 5, None, None),  # module docstring + import + constant
        (8, 11, "top_func", "function_definition"),
        (14, 17, "Greeter", "class_definition"),  # header + docstring + attribute
        (19, 20, "Greeter.greet", "function_definition"),
        (22, 24, "Greeter.loud", "function_definition"),
        (27, 28, None, None),  # trailing comment + TAIL
    ]


@requires_tree_sitter
def test_decorators_are_part_of_the_chunk() -> None:
    by_symbol = {c.symbol: c for c in _chunk(SAMPLE)}
    assert by_symbol["top_func"].text.startswith("@decorator(arg=1)")
    assert by_symbol["Greeter.loud"].text.lstrip().startswith("@property")


@requires_tree_sitter
def test_every_nonblank_line_lands_in_exactly_one_chunk() -> None:
    lines = SAMPLE.splitlines()
    owners = [0] * (len(lines) + 1)
    for chunk in _chunk(SAMPLE):
        for i in range(chunk.start_line, chunk.end_line + 1):
            owners[i] += 1
    for i, line in enumerate(lines, start=1):
        expected = 1 if line.strip() else owners[i]  # blank lines may be nobody's
        assert owners[i] == expected, f"line {i}: {line!r}"


@requires_tree_sitter
def test_chunk_text_is_a_verbatim_slice() -> None:
    lines = SAMPLE.splitlines()
    for chunk in _chunk(SAMPLE):
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


@requires_tree_sitter
def test_metadata_flows_through() -> None:
    chunk = _chunk(SAMPLE)[0]
    expected = ("r", "sample.py", "python", Kind.CODE)
    assert (chunk.repo, chunk.path, chunk.lang, chunk.kind) == expected


@requires_tree_sitter
def test_async_def_is_a_function_chunk() -> None:
    chunks = _chunk("async def fetch(x):\n    return x\n")
    assert [(c.symbol, c.node_type) for c in chunks] == [("fetch", "function_definition")]


@requires_tree_sitter
def test_nested_function_stays_inside_its_parent() -> None:
    code = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    chunks = _chunk(code)
    assert [c.symbol for c in chunks] == ["outer"]
    assert "def inner" in chunks[0].text


@requires_tree_sitter
def test_blank_only_gaps_produce_no_chunks() -> None:
    code = "def a():\n    return 1\n\n\n\ndef b():\n    return 2\n"
    assert [c.symbol for c in _chunk(code)] == ["a", "b"]


@requires_tree_sitter
def test_empty_and_blank_files_yield_nothing() -> None:
    assert _chunk("") == []
    assert _chunk("\n\n\n") == []


@requires_tree_sitter
def test_broken_file_does_not_crash() -> None:
    chunks = _chunk("def oops(:\n    return 1\n")
    assert isinstance(chunks, list)  # error-tolerant parse, no exception


def test_missing_tree_sitter_is_rejected_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # No skipif: the guard must work precisely when the extra is absent.
    monkeypatch.setattr("wsindex.ingest.ast_chunker.HAS_TREE_SITTER", False)
    with pytest.raises(RuntimeError, match="--extra ast"):
        _chunk("def f():\n    return 1\n")
