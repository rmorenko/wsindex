"""Tests for the kind dispatcher: behavior-level, no mocks.

The tests pin the observable contract per route — section chunks for
markdown docs, AST chunks for code and configs with a grammar, plain
windows for every fallback — rather than which function was called.
"""

import itertools
from textwrap import dedent

import pytest

from helpers import needs_grammar
from wsindex.ingest import chunk_file
from wsindex.ingest.languages import REGISTRY
from wsindex.ingest.text_chunker import chunk_plain
from wsindex.model import Kind, SourceFile


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


@needs_grammar("python")
def test_python_code_gets_ast_chunks() -> None:
    code = "def f():\n    return 1\n"
    chunks = chunk_file(code, SourceFile(repo=REPO, path="src/m.py", lang="python", kind=Kind.CODE))
    assert len(chunks) == 1
    assert chunks[0].node_type == "function_definition"
    assert chunks[0].symbol == "f"
    assert chunks[0].text == "def f():\n    return 1"


@needs_grammar("rust")
def test_rust_code_gets_ast_chunks() -> None:
    code = "impl S {\n    fn m(&self) -> u8 {\n        1\n    }\n}\n"
    chunks = chunk_file(code, SourceFile(repo=REPO, path="src/s.rs", lang="rust", kind=Kind.CODE))
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["S::m"].node_type == "function_item"


@needs_grammar("java")
def test_java_code_gets_ast_chunks() -> None:
    code = "class App {\n    void run() {\n    }\n}\n"
    chunks = chunk_file(
        code, SourceFile(repo=REPO, path="src/App.java", lang="java", kind=Kind.CODE)
    )
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["App.run"].node_type == "method_declaration"


@needs_grammar("typescript")
def test_typescript_code_gets_ast_chunks() -> None:
    code = "export function f(): number {\n  return 1;\n}\n"
    chunks = chunk_file(
        code, SourceFile(repo=REPO, path="src/f.ts", lang="typescript", kind=Kind.CODE)
    )
    assert len(chunks) == 1
    assert chunks[0].symbol == "f"
    assert chunks[0].node_type == "function_declaration"
    assert chunks[0].text.startswith("export function f")


@needs_grammar("python")
def test_python_without_tree_sitter_falls_back_to_plain_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drop_grammar(monkeypatch, "python")
    code = "def f():\n    return 1\n"
    chunks = chunk_file(code, SourceFile(repo=REPO, path="src/m.py", lang="python", kind=Kind.CODE))
    assert len(chunks) == 1
    assert chunks[0].node_type is None
    assert chunks[0].symbol is None


@needs_grammar("toml")
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


@needs_grammar("toml")
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


# --- the character cap ----------------------------------------------------
#
# A window measured only in lines is measured in the wrong unit: the model
# reads 256 tokens and stops. Measured on a real corpus, a line the model
# read is found by its own words 50% of the time at median depth 2, and a
# line past the cut 18% of the time at median depth 33.


def source_file() -> SourceFile:
    return SourceFile(repo="r", path="notes.txt", lang="text", kind=Kind.DOC)


def test_a_long_window_is_cut_at_the_character_cap() -> None:
    # Forty lines of eighty characters is 3 200, well past what the model
    # reads, and the old chunker made it one chunk.
    text = "\n".join("x" * 79 for _ in range(40))

    chunks = chunk_plain(text, source_file(), max_chars=300)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_a_short_file_is_untouched_by_the_cap() -> None:
    text = "\n".join(f"line {n}" for n in range(10))

    assert len(chunk_plain(text, source_file())) == 1


def test_the_ordinary_window_is_unchanged() -> None:
    """Forty lines, ten over, no line long enough to hit the cap.

    The cap has to be invisible where it does not apply, or every corpus
    without long lines gets re-chunked for nothing — and every chunk id
    in every existing index changes with it.
    """
    text = "\n".join(f"line {n}" for n in range(100))

    chunks = chunk_plain(text, source_file())

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40), (31, 70), (61, 100)]


def test_a_single_line_longer_than_the_cap_is_kept_whole() -> None:
    # It cannot be split without breaking the promise that a chunk's text
    # is a verbatim slice of its line range. Better one over-long chunk
    # than a chunk whose lines do not say where it came from.
    text = "y" * 5000

    chunks = chunk_plain(text, source_file(), max_chars=300)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 1)


def test_every_chunk_is_still_a_verbatim_slice_of_its_lines() -> None:
    # The invariant the whole module exists to keep: a hit points at a
    # real location. Asserted against the cap, since the cap is what
    # moves the boundaries.
    lines = [f"{'z' * (n * 17 % 200)} line {n}" for n in range(120)]
    text = "\n".join(lines)

    for chunk in chunk_plain(text, source_file(), max_chars=400):
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


def test_a_capped_window_still_overlaps_its_neighbour() -> None:
    # Overlap is why a passage straddling a boundary stays retrievable.
    # A cap that dropped it would trade one loss for another.
    text = "\n".join("w" * 95 for _ in range(60))

    chunks = chunk_plain(text, source_file(), max_chars=400)

    assert len(chunks) > 2
    for earlier, later in itertools.pairwise(chunks):
        assert later.start_line <= earlier.end_line, "consecutive chunks must share lines"


def test_overlap_stays_proportional_rather_than_fixed() -> None:
    """A five-line window with ten lines of overlap would be 80% repetition.

    So the overlap scales with what the cap actually admitted. The test
    is that a capped run does not explode: fifty long lines must not
    produce far more chunks than there are lines.
    """
    text = "\n".join("q" * 200 for _ in range(50))

    chunks = chunk_plain(text, source_file(), max_chars=400)

    assert len(chunks) <= 50


def test_the_cap_terminates_on_pathological_input() -> None:
    # Every line longer than the cap: the window can never take two, and
    # a step of zero would hang the indexer on one file.
    text = "\n".join("p" * 500 for _ in range(20))

    chunks = chunk_plain(text, source_file(), max_chars=100)

    assert len(chunks) == 20
