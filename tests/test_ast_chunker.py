"""Tests for the code AST chunkers: per-language sample shapes, the
_extend_back prelude walk, and properties shared by every language
(coverage invariant, verbatim slices, metadata).

Parser-dependent tests skip themselves on a base install (no `ast` extra),
including the missing-grammar guard test: it deletes the "python" registry
entry, which must exist to be deleted.
"""

from typing import TYPE_CHECKING

import pytest

from wsindex.ingest.ast_chunker import (
    _CODE_EXTRACTORS,
    _CONFIG_EXTRACTORS,
    CODE_PARSERS,
    CONFIG_PARSERS,
    HAS_TREE_SITTER,
    _extend_back,
    chunk_code_ast,
)
from wsindex.model import Chunk, Kind

if TYPE_CHECKING:
    from tree_sitter import Node

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


def _chunk(text: str, lang: str = "python") -> list[Chunk]:
    if lang not in CODE_PARSERS:
        pytest.skip(f"no {lang} grammar installed")
    return chunk_code_ast(text, repo="r", path=f"sample.{lang}", lang=lang, kind=Kind.CODE)


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


@pytest.mark.skipif("python" not in CODE_PARSERS, reason="needs the ast extra")
def test_missing_grammar_is_rejected_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(CODE_PARSERS, "python")
    # Call chunk_code_ast directly: _chunk would pytest.skip on the deleted
    # registry entry, silently bypassing the guard under test.
    with pytest.raises(RuntimeError, match="--extra ast"):
        chunk_code_ast(
            "def f():\n    return 1\n", repo="r", path="s.py", lang="python", kind=Kind.CODE
        )


def test_every_parser_has_an_extractor() -> None:
    assert set(CODE_PARSERS) <= set(_CODE_EXTRACTORS)
    assert set(CONFIG_PARSERS) <= set(_CONFIG_EXTRACTORS)


# --- _extend_back (rust prelude look-behind) --------------------------------


def _rust_children(source: str) -> "list[Node]":
    """Module-level named children of a parsed rust snippet; skips without the grammar."""
    ts_rust = pytest.importorskip("tree_sitter_rust")
    from tree_sitter import Language, Parser

    parser = Parser(Language(ts_rust.language()))
    return parser.parse(source.encode()).root_node.named_children


def test_extend_back_attaches_contiguous_prelude_chain() -> None:
    children = _rust_children("/// Doc.\n#[derive(Debug)]\nstruct S;\n")
    # struct at index 2 starts on line 3; the chain reaches back to line 1.
    assert _extend_back(children, 2, 3) == 1


def test_extend_back_stops_at_a_blank_line() -> None:
    children = _rust_children("/// Far doc.\n\n#[derive(Debug)]\nstruct S;\n")
    # The attribute (line 3) is adjacent, the doc comment (line 1) is not.
    assert _extend_back(children, 2, 4) == 3


def test_extend_back_ignores_non_prelude_siblings() -> None:
    children = _rust_children("use std::fmt;\nstruct S;\n")
    assert _extend_back(children, 1, 2) == 2


def test_extend_back_with_nothing_before_the_definition() -> None:
    children = _rust_children("struct S;\n")
    assert _extend_back(children, 0, 1) == 1


RUST = """\
//! Module doc comment.

use std::fmt;

const MAX: usize = 100;

#[derive(Debug, Clone)]
pub struct Greeter {
    name: String,
}

impl Greeter {
    /// Doc comment of new.
    pub fn new(name: String) -> Self {
        Self { name }
    }

    fn greet(&self) -> String {
        format!("hello {}", self.name)
    }
}

pub fn top_level(x: usize) -> usize {
    x + MAX
}

#[test]
fn it_works() {
    assert_eq!(top_level(1), 101);
}
"""


def test_rust_sample_spans_symbols_and_node_types() -> None:
    got = [(c.start_line, c.end_line, c.symbol, c.node_type) for c in _chunk(RUST, lang="rust")]
    assert got == [
        (1, 5, None, None),  # //! + use + const
        (7, 10, "Greeter", "struct_item"),  # #[derive] inside
        (12, 12, "Greeter", "impl_item"),  # header
        (13, 16, "Greeter::new", "function_item"),  # /// inside
        (18, 20, "Greeter::greet", "function_item"),
        (21, 21, "Greeter", "impl_item"),  # closing }
        (23, 25, "top_level", "function_item"),
        (27, 30, "it_works", "function_item"),  # #[test] inside
    ]


# --- properties shared by every code language -------------------------------

CODE_SAMPLES = [("python", SAMPLE), ("rust", RUST)]


@pytest.mark.parametrize(("lang", "sample"), CODE_SAMPLES)
def test_every_nonblank_line_lands_in_exactly_one_chunk(lang: str, sample: str) -> None:
    lines = sample.splitlines()
    owners = [0] * (len(lines) + 1)
    for chunk in _chunk(sample, lang=lang):
        for i in range(chunk.start_line, chunk.end_line + 1):
            owners[i] += 1
    for i, line in enumerate(lines, start=1):
        expected = 1 if line.strip() else owners[i]  # blank lines may be nobody's
        assert owners[i] == expected, f"{lang} line {i}: {line!r}"


@pytest.mark.parametrize(("lang", "sample"), CODE_SAMPLES)
def test_chunk_text_is_a_verbatim_slice(lang: str, sample: str) -> None:
    lines = sample.splitlines()
    for chunk in _chunk(sample, lang=lang):
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


@pytest.mark.parametrize(("lang", "sample"), CODE_SAMPLES)
def test_metadata_flows_through(lang: str, sample: str) -> None:
    chunk = _chunk(sample, lang=lang)[0]
    expected = ("r", f"sample.{lang}", lang, Kind.CODE)
    assert (chunk.repo, chunk.path, chunk.lang, chunk.kind) == expected
