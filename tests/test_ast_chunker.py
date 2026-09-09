"""Tests for the code AST chunkers: per-language sample shapes, the
_extend_back prelude walk, and properties shared by every language
(coverage invariant, verbatim slices, metadata).

Parser-dependent tests skip themselves on a base install (no `ast` extra),
including the missing-grammar guard test: it deletes the "python" registry
entry, which must exist to be deleted.
"""

from typing import TYPE_CHECKING

import pytest

from wsindex.ingest.ast import HAS_TREE_SITTER
from wsindex.ingest.ast.rust import _extend_back
from wsindex.ingest.chunker import chunk_file
from wsindex.ingest.languages import REGISTRY
from wsindex.model import Chunk, Kind, SourceFile

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


def _installed() -> set[str]:
    """Languages whose grammar is actually present in this environment."""
    return {spec.name for spec in REGISTRY.specs if REGISTRY.parser(spec.name) is not None}


def _chunk(text: str, lang: str = "python") -> list[Chunk]:
    if REGISTRY.parser(lang) is None:
        pytest.skip(f"no {lang} grammar installed")
    return chunk_file(text, SourceFile(repo="r", path=f"sample.{lang}", lang=lang, kind=Kind.CODE))


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


def test_every_parser_has_an_extractor_or_a_splitter() -> None:
    # This used to be a hand-checked invariant across four parallel
    # tables. `LanguageSpec` carries the grammar and the extractor as one
    # value and `register` refuses one without the other, so the two can
    # no longer drift — this asserts the property still holds end to end.
    #
    # A grammar may instead belong to a *container*, whose
    # parse is used to split the file rather than to chunk it. Those have
    # a splitter and no extractor, by construction.
    for spec in REGISTRY.specs:
        if REGISTRY.parser(spec.name) is None:
            continue
        has_path = REGISTRY.extractor(spec.name) is not None or spec.sections is not None
        assert has_path, spec.name


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
    assert _extend_back(children, index=2, start_line=3) == 1


def test_extend_back_stops_at_a_blank_line() -> None:
    children = _rust_children("/// Far doc.\n\n#[derive(Debug)]\nstruct S;\n")
    # The attribute (line 3) is adjacent, the doc comment (line 1) is not.
    assert _extend_back(children, index=2, start_line=4) == 3


def test_extend_back_ignores_non_prelude_siblings() -> None:
    children = _rust_children("use std::fmt;\nstruct S;\n")
    assert _extend_back(children, index=1, start_line=2) == 2


def test_extend_back_with_nothing_before_the_definition() -> None:
    children = _rust_children("struct S;\n")
    assert _extend_back(children, index=0, start_line=1) == 1


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


TS = """\
// header comment
import { x } from "./x";

const MAX = 100;

export function topLevel(a: number): number {
  return a + MAX;
}

export class Greeter {
  private name: string;

  constructor(name: string) {
    this.name = name;
  }

  greet(): string {
    return `hello ${this.name}`;
  }
}

interface Options {
  verbose: boolean;
}

export type Alias = string | number;

export const arrow = (n: number): number => n * 2;
"""


def test_ts_sample_spans_symbols_and_node_types() -> None:
    got = [(c.start_line, c.end_line, c.symbol, c.node_type) for c in _chunk(TS, lang="typescript")]
    assert got == [
        (1, 4, None, None),  # comment + import + plain const
        (6, 8, "topLevel", "function_declaration"),  # export inside
        (10, 11, "Greeter", "class_declaration"),  # header + field
        (13, 15, "Greeter.constructor", "method_definition"),
        (17, 19, "Greeter.greet", "method_definition"),
        (20, 20, "Greeter", "class_declaration"),  # closing }
        (22, 24, "Options", "interface_declaration"),
        (26, 26, "Alias", "type_alias_declaration"),
        (28, 28, "arrow", "lexical_declaration"),
    ]


def test_ts_only_function_valued_consts_become_chunks() -> None:
    code = "const MAX = 100;\nexport const arrow = (n: number): number => n * 2;\n"
    got = [(c.symbol, c.node_type) for c in _chunk(code, lang="typescript")]
    assert got == [(None, None), ("arrow", "lexical_declaration")]


JAVA = """\
package com.example;

import java.util.List;

/** Javadoc of Greeter. */
@Deprecated
public class Greeter {
    private String name;

    public Greeter(String name) {
        this.name = name;
    }

    @Override
    public String greet() {
        return "hello " + name;
    }
}

interface Options {
    boolean verbose();
}
"""


def test_java_sample_spans_symbols_and_node_types() -> None:
    got = [(c.start_line, c.end_line, c.symbol, c.node_type) for c in _chunk(JAVA, lang="java")]
    assert got == [
        (1, 5, None, None),  # package + import + javadoc (sibling comment)
        (6, 8, "Greeter", "class_declaration"),  # @Deprecated + header + field
        (10, 12, "Greeter.Greeter", "constructor_declaration"),
        (14, 17, "Greeter.greet", "method_declaration"),  # @Override inside
        (18, 18, "Greeter", "class_declaration"),  # closing }
        (20, 22, "Options", "interface_declaration"),
    ]


# --- properties shared by every code language -------------------------------

CODE_SAMPLES = [("python", SAMPLE), ("rust", RUST), ("typescript", TS), ("java", JAVA)]


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


# --- JavaScript / JSX / TSX, Go, C, C++ -----------------------------------
#
# The batch is "cheap" in the plan's sense: JS, JSX and TSX reuse the
# TypeScript policy object outright, and Go arrives from the example
# plugin unchanged. Only C and C++ needed new extractors — and only
# because C hides names in declarator chains and C++ nests whole files
# in namespaces.


JAVASCRIPT = """\
import x from "y";

function alpha(a) {
	return a;
}

const beta = (b) => b * 2;

const PLAIN = 42;

class Widget {
	render() {
		return null;
	}
}
"""


@pytest.mark.skipif("javascript" not in _installed(), reason="needs the ast extra")
def test_javascript_sample_spans_symbols_and_node_types() -> None:
    got = [(c.symbol, c.node_type) for c in _chunk(JAVASCRIPT, lang="javascript")]
    assert got == [
        (None, None),  # the import
        ("alpha", "function_declaration"),
        ("beta", "lexical_declaration"),  # arrow const counts as a function
        (None, None),  # a plain const is a legitimate gap
        ("Widget", "class_declaration"),
        ("Widget.render", "method_definition"),
        ("Widget", "class_declaration"),  # the closing brace
    ]


@pytest.mark.skipif("javascript" not in _installed(), reason="needs the ast extra")
def test_jsx_needs_no_second_language() -> None:
    # The JavaScript grammar parses JSX, so `.jsx` is the same language.
    code = "function Button({label}) {\n\treturn <button>{label}</button>;\n}\n"
    assert [c.symbol for c in _chunk(code, lang="javascript")] == ["Button"]


@pytest.mark.skipif("tsx" not in _installed(), reason="needs the ast extra")
def test_tsx_gets_typescript_policy_with_jsx_syntax() -> None:
    code = (
        "interface Props { label: string }\n\n"
        "export function Button({label}: Props) {\n"
        "\treturn <button>{label}</button>;\n}\n"
    )
    got = [(c.symbol, c.node_type) for c in _chunk(code, lang="tsx")]
    assert got == [
        ("Props", "interface_declaration"),
        ("Button", "function_declaration"),  # the `export` keyword is inside
    ]


GO = """\
package main

// Server owns the socket.
type Server struct {
	Host string
}

func New(host string) *Server {
	return &Server{Host: host}
}

func (s *Server) Serve(path string) error {
	return nil
}

func (s Server) String() string {
	return s.Host
}
"""


@pytest.mark.skipif("go" not in _installed(), reason="needs the ast extra")
def test_go_sample_spans_symbols_and_node_types() -> None:
    got = [(c.symbol, c.node_type) for c in _chunk(GO, lang="go")]
    assert got == [
        (None, None),  # package clause
        ("Server", "type_declaration"),
        ("New", "function_declaration"),
        ("Server.Serve", "method_declaration"),
        ("Server.String", "method_declaration"),
    ]


@pytest.mark.skipif("go" not in _installed(), reason="needs the ast extra")
def test_go_doc_comment_joins_its_declaration() -> None:
    # Arrived with the extractor from the example plugin; pointer and
    # value receivers qualify identically.
    by_symbol = {c.symbol: c for c in _chunk(GO, lang="go")}
    assert by_symbol["Server"].text.startswith("// Server owns the socket.")
    assert {"Server.Serve", "Server.String"} <= set(by_symbol)


C = """\
#include <stdio.h>

struct Node {
	int value;
};

enum Color { RED, GREEN };

static int helper(int a) {
	return a;
}

int *make(void) {
	return 0;
}
"""


@pytest.mark.skipif("c" not in _installed(), reason="needs the ast extra")
def test_c_sample_spans_symbols_and_node_types() -> None:
    got = [(c.symbol, c.node_type) for c in _chunk(C, lang="c")]
    assert got == [
        (None, None),  # the include
        ("Node", "struct_specifier"),
        ("Color", "enum_specifier"),
        ("helper", "function_definition"),
        ("make", "function_definition"),
    ]


@pytest.mark.skipif("c" not in _installed(), reason="needs the ast extra")
def test_c_pointer_return_still_yields_the_name() -> None:
    # `int *make(void)` nests the function_declarator inside a
    # pointer_declarator; a one-level lookup would miss the name.
    by_symbol = {c.symbol: c for c in _chunk(C, lang="c")}
    assert by_symbol["make"].text.startswith("int *make(void)")


CPP = """\
#include <string>

namespace app {

class Server {
public:
	void serve(const std::string& p) { host_ = p; }
	void stop();

private:
	std::string host_;
};

void Server::stop() {}

template <typename T>
T identity(T v) {
	return v;
}

}  // namespace app
"""


@pytest.mark.skipif("cpp" not in _installed(), reason="needs the ast extra")
def test_cpp_sample_spans_symbols_and_node_types() -> None:
    got = [(c.symbol, c.node_type) for c in _chunk(CPP, lang="cpp")]
    assert got == [
        (None, None),  # the include and the namespace header
        ("Server", "class_specifier"),
        ("Server::serve", "function_definition"),  # inline method
        ("Server", "class_specifier"),  # the class remainder
        ("Server::stop", "function_definition"),  # out-of-line definition
        ("identity", "function_definition"),
        (None, None),  # the namespace's closing brace
    ]


@pytest.mark.skipif("cpp" not in _installed(), reason="needs the ast extra")
def test_cpp_finds_definitions_inside_namespaces() -> None:
    # The whole point of the recursion: a file wrapped in a namespace has
    # one top-level node, so a flat walk would find nothing at all.
    assert any(c.symbol == "identity" for c in _chunk(CPP, lang="cpp"))


@pytest.mark.skipif("cpp" not in _installed(), reason="needs the ast extra")
def test_cpp_qualifies_methods_with_the_native_separator() -> None:
    # `Server::serve`, not `Server.serve` — the same choice Rust makes.
    symbols = {c.symbol for c in _chunk(CPP, lang="cpp")}
    assert {"Server::serve", "Server::stop"} <= symbols


@pytest.mark.skipif("cpp" not in _installed(), reason="needs the ast extra")
def test_cpp_template_keeps_its_parameter_list() -> None:
    by_symbol = {c.symbol: c for c in _chunk(CPP, lang="cpp")}
    assert by_symbol["identity"].text.startswith("template <typename T>")


@pytest.mark.skipif("cpp" not in _installed(), reason="needs the ast extra")
def test_cpp_nested_namespaces_are_reached() -> None:
    code = "namespace a {\nnamespace b {\nint f() { return 1; }\n}\n}\n"
    assert any(c.symbol == "f" for c in _chunk(code, lang="cpp"))


@pytest.mark.skipif("cpp" not in _installed(), reason="needs the ast extra")
def test_extern_c_block_is_descended_into() -> None:
    # `extern "C" {}` nests exactly like a namespace does.
    code = 'extern "C" {\nint legacy(void) { return 1; }\n}\n'
    assert any(c.symbol == "legacy" for c in _chunk(code, lang="cpp"))


@pytest.mark.parametrize("lang", ["javascript", "tsx", "go", "c", "cpp"])
def test_batch_one_languages_cover_every_line(lang: str) -> None:
    # The invariant the plan names for every new language: each non-blank
    # line lands in exactly one chunk, no matter how the extractor works.
    if lang not in _installed():
        pytest.skip(f"no {lang} grammar installed")
    source = {"javascript": JAVASCRIPT, "tsx": JAVASCRIPT, "go": GO, "c": C, "cpp": CPP}[lang]
    chunks = _chunk(source, lang=lang)
    covered = [line for c in chunks for line in range(c.start_line, c.end_line + 1)]
    non_blank = {i for i, line in enumerate(source.splitlines(), 1) if line.strip()}
    assert non_blank <= set(covered)
    assert len(covered) == len(set(covered))


# --- C#, Kotlin, PHP, Ruby ------------------------------------------------
#
# Four languages, one extractor. They differ in almost everything except
# the shape that matters — something optional wraps the file, types hold
# members, some declarations stand alone — so the walk lives once in
# `ast.nested` and each language supplies a `NestedPolicy`. What is
# pinned below is that the policies describe their languages correctly.


CSHARP = """\
using System;

namespace App {
	public class Server {
		public void Serve(string p) { }
		public int Port { get; set; }
	}

	public enum Color { Red, Green }
}
"""


@pytest.mark.skipif("csharp" not in _installed(), reason="needs the ast extra")
def test_csharp_sample_spans_symbols_and_node_types() -> None:
    got = [(c.symbol, c.node_type) for c in _chunk(CSHARP, lang="csharp")]
    assert got == [
        (None, None),  # using directive + the namespace header
        ("Server", "class_declaration"),
        ("Server.Serve", "method_declaration"),
        ("Server.Port", "property_declaration"),
        ("Server", "class_declaration"),  # the class remainder
        ("Color", "enum_declaration"),  # claimed whole
        (None, None),  # the namespace's closing brace
    ]


@pytest.mark.skipif("csharp" not in _installed(), reason="needs the ast extra")
def test_csharp_finds_declarations_inside_a_namespace() -> None:
    # Everything in C# nests under a namespace, so without the recursion
    # a whole file would yield exactly one chunk.
    assert any(c.symbol == "Server.Serve" for c in _chunk(CSHARP, lang="csharp"))


@pytest.mark.skipif("csharp" not in _installed(), reason="needs the ast extra")
def test_csharp_file_scoped_namespace_is_a_container_too() -> None:
    code = "namespace App;\n\nclass S {\n\tvoid M() { }\n}\n"
    assert any(c.symbol == "S.M" for c in _chunk(code, lang="csharp"))


KOTLIN = """\
package app

class Server(val host: String) {
	fun serve(path: String): Boolean {
		return true
	}
}

object Registry {
	fun get(): Int = 1
}

fun topLevel(a: Int) = a
"""


@pytest.mark.skipif("kotlin" not in _installed(), reason="needs the ast extra")
def test_kotlin_sample_spans_symbols_and_node_types() -> None:
    got = [(c.symbol, c.node_type) for c in _chunk(KOTLIN, lang="kotlin")]
    assert got == [
        (None, None),  # the package header — not a container, just a line
        ("Server", "class_declaration"),
        ("Server.serve", "function_declaration"),
        ("Server", "class_declaration"),
        ("Registry", "object_declaration"),
        ("Registry.get", "function_declaration"),
        ("Registry", "object_declaration"),
        ("topLevel", "function_declaration"),  # the same node type, standalone
    ]


@pytest.mark.skipif("kotlin" not in _installed(), reason="needs the ast extra")
def test_kotlin_members_live_in_a_child_not_a_field() -> None:
    # Kotlin keeps members in a `class_body` child rather than a `body`
    # field — the case `NestedPolicy.body_types` exists for. If the
    # fallback were missing, classes would yield one chunk and no methods.
    symbols = {c.symbol for c in _chunk(KOTLIN, lang="kotlin")}
    assert {"Server.serve", "Registry.get"} <= symbols


PHP = """\
<?php
namespace App;

class Server {
	public function serve(string $p): void {}
	private $host;
}

function helper(int $a): int { return $a; }
"""


@pytest.mark.skipif("php" not in _installed(), reason="needs the ast extra")
def test_php_sample_spans_symbols_and_node_types() -> None:
    got = [(c.symbol, c.node_type) for c in _chunk(PHP, lang="php")]
    assert got == [
        (None, None),  # the php tag and the file-scoped namespace
        ("Server", "class_declaration"),
        ("Server.serve", "method_declaration"),
        ("Server", "class_declaration"),  # the private field and brace
        ("helper", "function_definition"),
    ]


@pytest.mark.skipif("php" not in _installed(), reason="needs the ast extra")
def test_php_braced_namespace_is_descended_into() -> None:
    code = "<?php\nnamespace App {\nfunction f(): int { return 1; }\n}\n"
    assert any(c.symbol == "f" for c in _chunk(code, lang="php"))


RUBY = """\
require "json"

module App
  class Server
    def serve(path)
      true
    end

    def self.build
      new
    end
  end
end

def top_level(x)
  x
end
"""


@pytest.mark.skipif("ruby" not in _installed(), reason="needs the ast extra")
def test_ruby_sample_spans_symbols_and_node_types() -> None:
    got = [(c.symbol, c.node_type) for c in _chunk(RUBY, lang="ruby")]
    assert got == [
        (None, None),  # the require and the module header
        ("Server", "class"),
        ("Server.serve", "method"),
        ("Server.build", "singleton_method"),  # `def self.build`
        ("Server", "class"),
        (None, None),  # the module's `end`
        ("top_level", "method"),  # the same node type, standalone
    ]


@pytest.mark.skipif("ruby" not in _installed(), reason="needs the ast extra")
def test_ruby_def_is_a_method_inside_a_class_and_a_function_outside() -> None:
    # The case that makes the standalone/member split earn its keep: one
    # node type, two meanings, decided by where it sits.
    by_symbol = {c.symbol for c in _chunk(RUBY, lang="ruby")}
    assert "Server.serve" in by_symbol  # qualified inside the class
    assert "top_level" in by_symbol  # bare at the top level


@pytest.mark.skipif("ruby" not in _installed(), reason="needs the ast extra")
def test_ruby_class_level_methods_answer_the_same_symbol_query() -> None:
    # `def self.build` is Ruby's class method; qualifying it the same way
    # is what makes `--symbol Server` return a type's whole surface.
    symbols = {c.symbol for c in _chunk(RUBY, lang="ruby")}
    assert {"Server.serve", "Server.build"} <= symbols


@pytest.mark.parametrize("lang", ["csharp", "kotlin", "php", "ruby"])
def test_batch_two_languages_cover_every_line(lang: str) -> None:
    if lang not in _installed():
        pytest.skip(f"no {lang} grammar installed")
    source = {"csharp": CSHARP, "kotlin": KOTLIN, "php": PHP, "ruby": RUBY}[lang]
    chunks = _chunk(source, lang=lang)
    covered = [line for c in chunks for line in range(c.start_line, c.end_line + 1)]
    non_blank = {i for i, line in enumerate(source.splitlines(), 1) if line.strip()}
    assert non_blank <= set(covered)
    assert len(covered) == len(set(covered))


@pytest.mark.parametrize("lang", ["csharp", "kotlin", "php", "ruby"])
def test_batch_two_broken_files_do_not_crash(lang: str) -> None:
    if lang not in _installed():
        pytest.skip(f"no {lang} grammar installed")
    assert isinstance(_chunk("class Broken {\n", lang=lang), list)
