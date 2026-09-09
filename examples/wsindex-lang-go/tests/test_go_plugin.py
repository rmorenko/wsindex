"""Tests for the Go plugin — and a template for testing your own.

A plugin is testable the same way the built-in languages are: feed source
through `chunk_file` and assert the shape that comes out. Nothing here
needs a repository, a store or an index; the plugin's whole contribution
is "which spans does this tree yield", and that is what gets pinned.

Run with:

    uv pip install -e examples/wsindex-lang-go
    uv run pytest examples/wsindex-lang-go/tests
"""

from textwrap import dedent

import pytest
from wsindex.ingest import REGISTRY, chunk_file
from wsindex.model import Chunk, Kind

# `importorskip`, not a plain import: without the package installed a
# plain import fails during collection, which is an error rather than a
# skip — a confusing way to meet a test suite whose README tells you to
# run it. This turns "you forgot to install it" into a readable message.
plugin = pytest.importorskip(
    "wsindex_lang_go",
    reason="run `uv pip install -e examples/wsindex-lang-go` first",
)
GO = plugin.GO
LANGUAGES = plugin.LANGUAGES

pytestmark = pytest.mark.skipif(
    REGISTRY.parser("go") is None,
    reason="the tree-sitter-go grammar is not installed",
)

SOURCE = dedent("""\
    package main

    import "fmt"

    // Server handles requests.
    type Server struct {
    \tHost string
    }

    type Handler interface {
    \tServe(path string) error
    }

    func New(host string) *Server {
    \treturn &Server{Host: host}
    }

    func (s *Server) Serve(path string) error {
    \tfmt.Println(path)
    \treturn nil
    }

    func (s Server) String() string {
    \treturn s.Host
    }
""")


def chunks(text: str) -> list[Chunk]:
    return chunk_file(text, repo="r", path="server.go", lang="go", kind=Kind.CODE)


def shape(text: str) -> list[tuple[str | None, str | None]]:
    return [(c.symbol, c.node_type) for c in chunks(text)]


# --- the plugin is discovered at all -------------------------------------


def test_the_entry_point_registers_go() -> None:
    # If this fails, the package is not installed — the plugin machinery
    # is fine, `pip install -e` just has not run.
    spec = REGISTRY.get("go")
    assert spec is not None
    assert spec.suffixes == (".go",)
    assert spec.kind is Kind.CODE


def test_the_exported_tuple_holds_the_spec() -> None:
    # What the entry point resolves to; a tuple so the plugin can grow.
    assert LANGUAGES == (GO,)


def test_go_files_are_selected_by_the_walker() -> None:
    from pathlib import Path

    matched = REGISTRY.match(Path("cmd/server/main.go"))
    assert matched is not None
    assert matched.name == "go"


# --- what Go gets chunked into -------------------------------------------


def test_functions_methods_and_types_each_get_a_chunk() -> None:
    assert shape(SOURCE) == [
        (None, None),  # package clause, import, and the doc comment
        ("Server", "type_declaration"),
        ("Handler", "type_declaration"),
        ("New", "function_declaration"),
        ("Server.Serve", "method_declaration"),
        ("Server.String", "method_declaration"),
    ]


def test_pointer_and_value_receivers_qualify_the_same() -> None:
    # `(s *Server)` and `(s Server)` are the same type to a reader, so
    # `--symbol Server` has to find both.
    symbols = {c.symbol for c in chunks(SOURCE)}
    assert {"Server.Serve", "Server.String"} <= symbols


def test_a_grouped_type_block_yields_one_chunk_per_type() -> None:
    source = "package main\n\ntype (\n\tID    string\n\tCount int\n)\n"
    assert [c.symbol for c in chunks(source) if c.symbol] == ["ID", "Count"]


def test_a_lone_type_declaration_includes_the_keyword() -> None:
    source = "package main\n\ntype ID string\n"
    typed = [c for c in chunks(source) if c.symbol == "ID"]
    assert typed[0].text.startswith("type ID")


def test_package_and_imports_become_gap_chunks() -> None:
    # Not claimed by the extractor, so the gap pass takes them: still
    # indexed, just without a symbol.
    first = chunks(SOURCE)[0]
    assert first.symbol is None
    assert first.text.startswith("package main")


def test_every_non_blank_line_lands_in_exactly_one_chunk() -> None:
    # The invariant the gap pass exists to guarantee. Worth asserting in
    # any plugin: an extractor that claims overlapping spans breaks it.
    produced = chunks(SOURCE)
    covered = [line for c in produced for line in range(c.start_line, c.end_line + 1)]
    non_blank = {i for i, line in enumerate(SOURCE.splitlines(), 1) if line.strip()}
    assert non_blank <= set(covered)
    assert len(covered) == len(set(covered))  # no line claimed twice


def test_chunk_text_is_a_verbatim_slice() -> None:
    lines = SOURCE.splitlines()
    for chunk in chunks(SOURCE):
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


def test_metadata_flows_through() -> None:
    chunk = chunks(SOURCE)[0]
    assert (chunk.repo, chunk.path, chunk.lang, chunk.kind) == ("r", "server.go", "go", Kind.CODE)


# --- damaged input must not raise ----------------------------------------


def test_a_syntax_error_is_chunked_from_what_parsed() -> None:
    # The parser is error tolerant and so must the extractor be: a file
    # mid-edit is the normal state of a file being searched for.
    produced = chunks("package main\n\nfunc broken( {\n")
    assert isinstance(produced, list)


def test_an_empty_file_yields_nothing() -> None:
    assert chunks("") == []


def test_a_file_with_only_a_package_clause_still_indexes() -> None:
    produced = chunks("package main\n")
    assert len(produced) == 1
    assert produced[0].symbol is None


# --- doc comments belong to what they document ---------------------------


def test_a_doc_comment_joins_the_declaration_below_it() -> None:
    # Go's convention: `// Serve ...` directly above the method. That
    # sentence is usually the most searchable thing about a declaration,
    # so filing it as a chunk of its own would waste it.
    source = "package main\n\n// Serve handles one request.\nfunc Serve() {}\n"
    documented = [c for c in chunks(source) if c.symbol == "Serve"]
    assert len(documented) == 1
    assert documented[0].text.startswith("// Serve handles one request.")
    assert "func Serve()" in documented[0].text


def test_several_comment_lines_all_join() -> None:
    source = "package main\n\n// One.\n// Two.\n// Three.\nfunc Serve() {}\n"
    documented = next(c for c in chunks(source) if c.symbol == "Serve")
    assert documented.text.count("//") == 3


def test_a_comment_separated_by_a_blank_line_stays_apart() -> None:
    # godoc's own rule: a blank line between comment and declaration
    # means the comment is not documentation. The chunker follows it.
    source = "package main\n\n// Not documentation.\n\nfunc Serve() {}\n"
    produced = chunks(source)
    served = next(c for c in produced if c.symbol == "Serve")
    assert served.text == "func Serve() {}"
    # It is still indexed — the gap pass sweeps it up with the preceding
    # uncovered lines rather than giving it a chunk of its own.
    assert any("// Not documentation." in c.text for c in produced if c.symbol is None)


def test_a_documented_type_keeps_its_comment() -> None:
    source = "package main\n\n// Server owns the socket.\ntype Server struct {\n\tHost string\n}\n"
    documented = next(c for c in chunks(source) if c.symbol == "Server")
    assert documented.text.startswith("// Server owns the socket.")
