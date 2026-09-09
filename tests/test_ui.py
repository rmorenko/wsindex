"""Output tests: the two shapes, and the rule that picks between them.

The rule is the point — rich on a terminal, the old greppable lines
everywhere else — so the tests drive a `Console` told what it is rather
than a mock of one. `force_terminal` is how rich itself is asked to
pretend, and `width` pins the wrapping so an assertion means the same
thing on any machine.
"""

from __future__ import annotations

import io
import re

import pytest
from rich.console import Console

from wsindex.model import Hit
from wsindex.ui import PLAIN_ENV, plain_hit, progress, render_hits, wants_rich


def hit(score: float = 0.5, **overrides: object) -> Hit:
    meta: dict[str, object] = {
        "repo": "app",
        "path": "src/main.py",
        "start_line": 41,
        "end_line": 43,
        "lang": "python",
        "kind": "code",
        "symbol": "greet",
        "text": "def greet(name):\n    return name\n",
    }
    meta.update(overrides)
    return Hit(score=score, metadata=meta, native_id="id")


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def rendered(*, terminal: bool, hits: list[Hit] | None = None) -> str:
    """What the reader sees, with the styling taken back off.

    `no_color=True` drops colours but keeps bold, and bold sits *inside*
    the repo name — so a test asserting on text has to strip the codes or
    it is asserting on the styling instead.
    """
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=terminal, width=100, no_color=True)
    render_hits(hits if hits is not None else [hit()], target=console)
    return _ANSI.sub("", stream.getvalue())


# --- which shape ---------------------------------------------------------


def test_a_pipe_gets_the_greppable_line(monkeypatch: pytest.MonkeyPatch) -> None:
    # `wsindex search x | grep foo` has to keep working, and a table
    # drawn with box characters is not something to grep.
    monkeypatch.delenv(PLAIN_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert rendered(terminal=False).strip() == "app/src/main.py:41-43  0.500  def greet(name):"


def test_a_terminal_gets_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PLAIN_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)

    out = rendered(terminal=True)

    assert "score" in out and "where" in out
    assert "app/src/main.py" in out
    assert "0.500" in out
    # Real line numbers, not 1..n: a snippet whose numbers do not match
    # the file is worse than one with none.
    assert "41" in out


def test_no_color_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PLAIN_ENV, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert rendered(terminal=True).strip().startswith("app/src/main.py:41-43")


def test_the_escape_hatch_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    # For a demo recording, a screenshot, or a terminal that lies.
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv(PLAIN_ENV, "1")
    assert rendered(terminal=True).strip().startswith("app/src/main.py:41-43")


def test_wants_rich_asks_the_console_it_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PLAIN_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert wants_rich(Console(file=io.StringIO(), force_terminal=True))
    assert not wants_rich(Console(file=io.StringIO(), force_terminal=False))


# --- what the shapes hold -------------------------------------------------


def test_the_plain_line_is_one_line_even_for_a_long_chunk() -> None:
    # It is a line-oriented format; a chunk's first line is what fits.
    long_chunk = hit(text="def f():\n" + "    x = 1\n" * 200)
    assert "\n" not in plain_hit(long_chunk)


def test_a_language_without_a_lexer_still_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    # `lang` comes from the chunker and may be a name rich has never
    # heard of — a plugin's language, or one marked up per repo (17z).
    monkeypatch.delenv(PLAIN_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)

    out = rendered(terminal=True, hits=[hit(lang="invented-lang")])

    assert "def greet(name):" in out


def test_the_symbol_is_shown_when_there_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PLAIN_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert "greet" in rendered(terminal=True)
    assert "41-43" in rendered(terminal=True, hits=[hit(symbol=None)])


# --- progress --------------------------------------------------------------


def test_progress_is_silent_off_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PLAIN_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, width=100)

    with progress("indexing", target=console) as report:
        assert report is None  # the pipeline tolerates None, so no branch

    assert stream.getvalue() == ""


def test_progress_names_the_repo_being_read(monkeypatch: pytest.MonkeyPatch) -> None:
    # Indexing is seconds of nothing, which reads as a hang.
    monkeypatch.delenv(PLAIN_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=100, no_color=True)

    with progress("indexing", target=console) as report:
        assert report is not None
        report("repo1")

    assert "indexing" in stream.getvalue()
    assert "repo1" in stream.getvalue()
