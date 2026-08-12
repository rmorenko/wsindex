"""Tests for the kind dispatcher: behavior-level, no mocks.

The tests pin the observable contract per route — section chunks for
markdown docs, AST chunks for code and configs with a grammar, plain
windows for every fallback — rather than which function was called.
"""

from textwrap import dedent

import pytest

from wsindex.ingest.ast_chunker import CODE_PARSERS, CONFIG_PARSERS
from wsindex.ingest.chunker import chunk_file
from wsindex.model import Kind

REPO = "test"

MARKDOWN = dedent("""\
    intro
    # One
    body
    """)


def test_doc_markdown_produces_sections() -> None:
    chunks = chunk_file(MARKDOWN, repo=REPO, path="README.md", lang="markdown", kind=Kind.DOC)
    assert len(chunks) == 2
    assert all(c.node_type == "section" for c in chunks)
    assert chunks[1].symbol == "One"


def test_code_falls_back_to_plain_windows() -> None:
    code = "def f():\n    return 1\n"
    chunks = chunk_file(code, repo=REPO, path="src/m.py", lang="java", kind=Kind.CODE)
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].symbol is None
    assert chunks[0].text == "def f():\n    return 1"


def test_python_code_gets_ast_chunks() -> None:
    code = "def f():\n    return 1\n"
    chunks = chunk_file(code, repo=REPO, path="src/m.py", lang="python", kind=Kind.CODE)
    assert len(chunks) == 1
    assert chunks[0].node_type == "function_definition"
    assert chunks[0].symbol == "f"
    assert chunks[0].text == "def f():\n    return 1"


@pytest.mark.skipif("python" not in CODE_PARSERS, reason="needs the ast extra")
def test_python_without_tree_sitter_falls_back_to_plain_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(CODE_PARSERS, "python")
    code = "def f():\n    return 1\n"
    chunks = chunk_file(code, repo=REPO, path="src/m.py", lang="python", kind=Kind.CODE)
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].symbol is None


@pytest.mark.skipif("toml" not in CONFIG_PARSERS, reason="needs the ast extra")
def test_toml_config_gets_table_chunks() -> None:
    cfg = "[table]\nkey = 1\n"
    chunks = chunk_file(cfg, repo=REPO, path="pyproject.toml", lang="toml", kind=Kind.CONFIG)
    assert len(chunks) == 1
    assert chunks[0].node_type == "table"
    assert chunks[0].symbol == "table"
    assert chunks[0].text == "[table]\nkey = 1"


def test_config_without_grammar_falls_back_to_plain_windows() -> None:
    cfg = "key = 1\nother = 2\n"
    chunks = chunk_file(cfg, repo=REPO, path="settings.ini", lang="ini", kind=Kind.CONFIG)
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].symbol is None


@pytest.mark.skipif("toml" not in CONFIG_PARSERS, reason="needs the ast extra")
def test_removed_grammar_falls_back_to_plain_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    # CONFIG_PARSERS is one shared dict object: the dispatcher imported a
    # reference to it, so deleting the entry is visible through both names
    # (unlike rebinding a flag, which needs patching per namespace).
    monkeypatch.delitem(CONFIG_PARSERS, "toml")
    cfg = "[table]\nkey = 1\n"
    chunks = chunk_file(cfg, repo=REPO, path="pyproject.toml", lang="toml", kind=Kind.CONFIG)
    assert chunks[0].node_type is None


def test_metadata_flows_through() -> None:
    chunks = chunk_file(MARKDOWN, repo="wsx", path="docs/a.md", lang="markdown", kind=Kind.DOC)
    assert chunks
    for chunk in chunks:
        assert chunk.repo == "wsx"
        assert chunk.path == "docs/a.md"
        assert chunk.lang == "markdown"
