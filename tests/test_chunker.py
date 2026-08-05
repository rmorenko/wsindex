"""Tests for the kind dispatcher: behavior-level, no mocks.

The three kinds currently share the text_chunker fallback, so the tests pin
the observable contract (section chunks for markdown docs, plain windows for
code/config) rather than which function was called — they must keep passing
unchanged when the code/config branches move to the AST chunker.
"""

from textwrap import dedent

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
    chunks = chunk_file(code, repo=REPO, path="src/m.py", lang="python", kind=Kind.CODE)
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].symbol is None
    assert chunks[0].text == "def f():\n    return 1"


def test_config_falls_back_to_plain_windows() -> None:
    cfg = "[table]\nkey = 1\n"
    chunks = chunk_file(cfg, repo=REPO, path="pyproject.toml", lang="toml", kind=Kind.CONFIG)
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].text == "[table]\nkey = 1"


def test_metadata_flows_through() -> None:
    chunks = chunk_file(MARKDOWN, repo="wsx", path="docs/a.md", lang="markdown", kind=Kind.DOC)
    assert chunks
    for chunk in chunks:
        assert chunk.repo == "wsx"
        assert chunk.path == "docs/a.md"
        assert chunk.lang == "markdown"
