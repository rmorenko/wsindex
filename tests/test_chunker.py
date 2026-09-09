"""Tests for the kind dispatcher: behavior-level, no mocks.

The tests pin the observable contract per route — section chunks for
markdown docs, AST chunks for code and configs with a grammar, plain
windows for every fallback — rather than which function was called.
"""

from textwrap import dedent

import pytest

from wsindex.ingest import chunk_file
from wsindex.ingest.languages import REGISTRY
from wsindex.model import Kind, SourceFile


def has_grammar(lang: str) -> bool:
    return REGISTRY.parser(lang) is not None


def drop_grammar(monkeypatch: pytest.MonkeyPatch, lang: str) -> None:
    """Make the registry report no parser for one language.

    The dispatcher asks the registry per call, so hiding the parser is
    all it takes to exercise the fallback — no need to reach into a
    shared dict the way the old parser tables required.
    """
    real = REGISTRY.parser
    monkeypatch.setattr(REGISTRY, "parser", lambda name: None if name == lang else real(name))


REPO = "test"

MARKDOWN = dedent("""\
    intro
    # One
    body
    """)


def test_doc_markdown_produces_sections() -> None:
    chunks = chunk_file(
        MARKDOWN, SourceFile(repo=REPO, path="README.md", lang="markdown", kind=Kind.DOC)
    )
    assert len(chunks) == 2
    assert all(c.node_type == "section" for c in chunks)
    assert chunks[1].symbol == "One"


def test_code_falls_back_to_plain_windows() -> None:
    # An invented name, not a real language: this used to say "go", which
    # stopped being true the moment examples/wsindex-lang-go was installed.
    # A core test must not depend on which plugins the machine happens to
    # have, so the precondition is asserted rather than assumed.
    lang = "nolang-for-this-test"
    assert REGISTRY.parser(lang) is None
    code = "func f() int {\n\treturn 1\n}\n"
    chunks = chunk_file(code, SourceFile(repo=REPO, path="src/m.nolang", lang=lang, kind=Kind.CODE))
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].symbol is None


@pytest.mark.skipif(not has_grammar("python"), reason="needs the ast extra")
def test_python_code_gets_ast_chunks() -> None:
    code = "def f():\n    return 1\n"
    chunks = chunk_file(code, SourceFile(repo=REPO, path="src/m.py", lang="python", kind=Kind.CODE))
    assert len(chunks) == 1
    assert chunks[0].node_type == "function_definition"
    assert chunks[0].symbol == "f"
    assert chunks[0].text == "def f():\n    return 1"


@pytest.mark.skipif(not has_grammar("rust"), reason="needs the ast extra")
def test_rust_code_gets_ast_chunks() -> None:
    code = "impl S {\n    fn m(&self) -> u8 {\n        1\n    }\n}\n"
    chunks = chunk_file(code, SourceFile(repo=REPO, path="src/s.rs", lang="rust", kind=Kind.CODE))
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["S::m"].node_type == "function_item"


@pytest.mark.skipif(not has_grammar("java"), reason="needs the ast extra")
def test_java_code_gets_ast_chunks() -> None:
    code = "class App {\n    void run() {\n    }\n}\n"
    chunks = chunk_file(
        code, SourceFile(repo=REPO, path="src/App.java", lang="java", kind=Kind.CODE)
    )
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["App.run"].node_type == "method_declaration"


@pytest.mark.skipif(not has_grammar("typescript"), reason="needs the ast extra")
def test_typescript_code_gets_ast_chunks() -> None:
    code = "export function f(): number {\n  return 1;\n}\n"
    chunks = chunk_file(
        code, SourceFile(repo=REPO, path="src/f.ts", lang="typescript", kind=Kind.CODE)
    )
    assert len(chunks) == 1
    assert chunks[0].symbol == "f"
    assert chunks[0].node_type == "function_declaration"
    assert chunks[0].text.startswith("export function f")


@pytest.mark.skipif(not has_grammar("python"), reason="needs the ast extra")
def test_python_without_tree_sitter_falls_back_to_plain_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drop_grammar(monkeypatch, "python")
    code = "def f():\n    return 1\n"
    chunks = chunk_file(code, SourceFile(repo=REPO, path="src/m.py", lang="python", kind=Kind.CODE))
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].symbol is None


@pytest.mark.skipif(not has_grammar("toml"), reason="needs the ast extra")
def test_toml_config_gets_table_chunks() -> None:
    cfg = "[table]\nkey = 1\n"
    chunks = chunk_file(
        cfg, SourceFile(repo=REPO, path="pyproject.toml", lang="toml", kind=Kind.CONFIG)
    )
    assert len(chunks) == 1
    assert chunks[0].node_type == "table"
    assert chunks[0].symbol == "table"
    assert chunks[0].text == "[table]\nkey = 1"


def test_config_without_grammar_falls_back_to_plain_windows() -> None:
    cfg = "key = 1\nother = 2\n"
    chunks = chunk_file(
        cfg, SourceFile(repo=REPO, path="settings.ini", lang="ini", kind=Kind.CONFIG)
    )
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].symbol is None


@pytest.mark.skipif(not has_grammar("toml"), reason="needs the ast extra")
def test_removed_grammar_falls_back_to_plain_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    drop_grammar(monkeypatch, "toml")
    cfg = "[table]\nkey = 1\n"
    chunks = chunk_file(
        cfg, SourceFile(repo=REPO, path="pyproject.toml", lang="toml", kind=Kind.CONFIG)
    )
    assert chunks[0].node_type is None


def test_metadata_flows_through() -> None:
    chunks = chunk_file(
        MARKDOWN, SourceFile(repo="wsx", path="docs/a.md", lang="markdown", kind=Kind.DOC)
    )
    assert chunks
    for chunk in chunks:
        assert chunk.repo == "wsx"
        assert chunk.path == "docs/a.md"
        assert chunk.lang == "markdown"
