from textwrap import dedent

import pytest

from wsindex.ingest.text_chunker import chunk_markdown, chunk_plain, chunk_text
from wsindex.model import Kind

REPO = "test"
PATH = "/path/to/file"


def test_markdown_sections() -> None:
    example = dedent(
        """\
        intro line
        # One
        body a
        ## Two
        body b
        """
    )
    chunks = chunk_markdown(example, repo=REPO, path=PATH, lang="markdown", kind=Kind.DOC)
    assert len(chunks) == 3
    assert chunks[0].text == "intro line"
    assert chunks[0].start_line == 1
    assert chunks[0].kind == Kind.DOC
    assert chunks[1].text.startswith("# One")
    assert chunks[1].start_line == 2
    assert chunks[1].end_line == 3
    assert chunks[1].symbol == "One"
    assert chunks[1].text.endswith("body a")
    assert chunks[1].kind == Kind.DOC
    assert chunks[2].text.startswith("## Two")
    assert chunks[2].start_line == 4
    assert chunks[2].end_line == 5
    assert chunks[2].text.endswith("body b")
    assert chunks[2].kind == Kind.DOC


def test_plain_short_text_single_chunk() -> None:
    example = dedent("""\
         first line
         second line

         fourth line
     """)
    chunks = chunk_plain(example, repo=REPO, path=PATH, lang="text", kind=Kind.DOC)
    assert len(chunks) == 1
    assert chunks[0].text == "first line\nsecond line\n\nfourth line"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 4
    assert chunks[0].symbol is None
    assert chunks[0].kind == Kind.DOC


def test_plain_window_positions() -> None:
    example = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = chunk_plain(example, repo=REPO, path=PATH, lang="text", kind=Kind.DOC)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40), (31, 70), (61, 100)]


def test_empty_text_gives_no_chunks() -> None:
    example = ""
    chunks = chunk_plain(example, repo=REPO, path=PATH, lang="text", kind=Kind.DOC)
    assert len(chunks) == 0
    example = "\n\n"
    chunks = chunk_plain(example, repo=REPO, path=PATH, lang="text", kind=Kind.DOC)
    assert len(chunks) == 0


def test_markdown_fence_hides_headers() -> None:
    example = dedent("""\
        # Real
        ```
        # not a header
        ```
        tail
        """)
    chunks = chunk_markdown(example, repo=REPO, path=PATH, lang="markdown", kind=Kind.DOC)
    assert len(chunks) == 1
    assert chunks[0].symbol == "Real"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 5
    assert "# not a header" in chunks[0].text


def test_markdown_blank_sections_skipped() -> None:
    assert chunk_markdown("", repo=REPO, path=PATH, lang="markdown", kind=Kind.DOC) == []
    # Blank preamble before the first header must not become an empty chunk.
    chunks = chunk_markdown("\n\n# One\nbody", repo=REPO, path=PATH, lang="markdown", kind=Kind.DOC)
    assert len(chunks) == 1
    assert chunks[0].symbol == "One"
    assert chunks[0].start_line == 3
    assert chunks[0].end_line == 4


def test_overlap_must_be_less_than_window() -> None:
    with pytest.raises(ValueError):
        chunk_plain("x", repo="r", path="p", lang="text", window=10, overlap=10, kind=Kind.DOC)


def test_metadata_flows_through() -> None:
    example = dedent("""\
             first line
             second line

             fourth line
         """)
    chunks = chunk_text(example, repo=REPO, path=PATH, lang="rst", kind=Kind.CODE)
    for chunk in chunks:
        assert chunk.repo == REPO
        assert chunk.path == PATH
        assert chunk.lang == "rst"
        assert chunk.kind == Kind.CODE
    chunks = chunk_text(example, repo=REPO, path=PATH, lang="markdown", kind=Kind.DOC)
    for chunk in chunks:
        assert chunk.kind == Kind.DOC
        assert chunk.repo == REPO
        assert chunk.path == PATH
        assert chunk.lang == "markdown"
