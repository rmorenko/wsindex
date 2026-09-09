"""Shell tests: the parser, the editor argv, and the loop's stateful part.

The loop itself is driven with a scripted `PromptSession`, because what
is worth testing is what the loop *does* with a line — a terminal
emulator would be testing prompt_toolkit.

Everything else here is a pure function on purpose: parsing a line and
deciding how to invoke an editor are the two places this command can be
subtly wrong, and neither needs a terminal to be wrong in.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from wsindex.config import Config
from wsindex.model import Hit, Kind
from wsindex.shell import ShellError, open_in_editor, parse, run


def hit(**overrides: object) -> Hit:
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
    return Hit(score=0.5, metadata=meta, native_id="id")


# --- parsing ---------------------------------------------------------------


def test_a_plain_question_is_the_query() -> None:
    query = parse("how are chunks deduplicated")
    assert query.text == "how are chunks deduplicated"
    assert query.k == 10
    assert query.filters is None


def test_flags_may_come_before_or_after_the_words() -> None:
    # People type a flag where they think of it, and both orders are the
    # same question.
    before = parse("--lang python how does it chunk")
    after = parse("how does it chunk --lang python")
    assert before == after
    assert before.text == "how does it chunk"


def test_every_search_flag_is_understood() -> None:
    query = parse(
        "port --repo app --lang toml --lang yaml --kind config --path 'src/*' --symbol f -k 3"
    )
    assert query.repo == "app"
    assert query.k == 3
    assert query.filters is not None
    assert query.filters.lang == ("toml", "yaml")
    assert query.filters.kind == (Kind.CONFIG,)
    assert query.filters.path == "src/*"
    assert query.filters.symbol == "f"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("--lang", "needs a value"),
        ("x -k many", "wants a number"),
        ("x --kind prose", "--kind wants one of"),
        ("x --nope 1", "unknown flag"),
        ("--lang python", "nothing to search for"),
        ("unbalanced 'quote", "quotation"),
    ],
)
def test_a_line_that_makes_no_sense_says_why(line: str, message: str) -> None:
    with pytest.raises(ShellError, match=message):
        parse(line)


# --- opening an editor -----------------------------------------------------


@pytest.mark.parametrize(
    ("editor", "expected_tail"),
    [
        ("vim", ["+41", "/checkouts/app/src/main.py"]),
        ("nvim", ["+41", "/checkouts/app/src/main.py"]),
        ("code", ["--goto", "/checkouts/app/src/main.py:41"]),
        ("subl", ["/checkouts/app/src/main.py:41"]),
        ("emacs", ["/checkouts/app/src/main.py"]),
    ],
)
def test_the_editor_is_told_the_line_the_way_it_expects(
    editor: str, expected_tail: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Getting this wrong means opening at the top of a two-thousand-line
    # file, which looks like the search was wrong.
    monkeypatch.setenv("EDITOR", editor)
    Config.reset()
    config = Config.default("ws")
    config.add_repo("app", path="/checkouts/app")

    assert open_in_editor(hit())[1:] == expected_tail


def test_an_editor_with_arguments_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITOR", "code --wait")
    Config.reset()
    Config.default("ws").add_repo("app", path="/checkouts/app")

    assert open_in_editor(hit())[:3] == ["code", "--wait", "--goto"]


def test_a_repo_the_config_forgot_still_opens_something(monkeypatch: pytest.MonkeyPatch) -> None:
    # A hit can outlive its config entry; a relative path beats a crash.
    monkeypatch.setenv("EDITOR", "vi")
    Config.reset()
    Config.default("ws")

    assert open_in_editor(hit())[-1].endswith("src/main.py")


# --- the loop --------------------------------------------------------------


class _ScriptedSession:
    """A PromptSession that reads from a list instead of a terminal."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)

    def prompt(self, _: str) -> str:
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


def drive(
    lines: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    hits: list[Hit] | None = None,
    searched: list[tuple[str, Any]] | None = None,
) -> str:
    """Run the loop over a script and return what it printed."""
    stream = io.StringIO()
    monkeypatch.setattr(
        "wsindex.shell.console",
        lambda **_: Console(file=stream, force_terminal=False, width=100, no_color=True),
    )
    # Patched where `run` looks it up — inside the function, so the
    # patch has to land on prompt_toolkit itself.
    monkeypatch.setattr("prompt_toolkit.PromptSession", lambda **_: _ScriptedSession(lines))

    class _Pipeline:
        def search(self, query: str, **kwargs: Any) -> list[Hit]:
            if searched is not None:
                searched.append((query, kwargs))
            if hits is None:
                return [hit()]
            return hits

    run(_Pipeline(), history_dir=tmp_path / "state")  # type: ignore[arg-type]
    return stream.getvalue()


def test_a_question_is_searched_and_shown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[tuple[str, Any]] = []
    out = drive(["how does it chunk --lang python"], monkeypatch, tmp_path, searched=seen)

    assert seen == [
        ("how does it chunk", {"k": 10, "repo": None, "filters": seen[0][1]["filters"]})
    ]
    assert seen[0][1]["filters"].lang == ("python",)
    assert "app/src/main.py:41-43" in out


def test_an_empty_result_says_so(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert "no results" in drive(["nothing here"], monkeypatch, tmp_path, hits=[])


def test_a_number_shows_that_hit_in_full(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = drive(["question", "1"], monkeypatch, tmp_path)
    # The whole chunk, not the one-line summary the table shows.
    assert "return name" in out


def test_picking_before_searching_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert "search first" in drive(["1"], monkeypatch, tmp_path)


def test_picking_a_hit_that_is_not_there(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = drive(["question", "9"], monkeypatch, tmp_path)
    assert "between 1 and 1" in out


def test_a_bad_line_does_not_end_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The loop's whole job is to keep going; an unknown flag is a
    # sentence, not an exit.
    out = drive(["--nope 1", "question"], monkeypatch, tmp_path)
    assert "unknown flag" in out
    assert "app/src/main.py:41-43" in out


def test_quit_ends_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[tuple[str, Any]] = []
    drive([":quit", "never searched"], monkeypatch, tmp_path, searched=seen)
    assert seen == []


def test_help_and_repos_answer_without_searching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Config.reset()
    Config.default("ws").add_repo("app", path="/checkouts/app")
    out = drive([":help", ":repos"], monkeypatch, tmp_path)
    assert ":open <number>" in out
    assert "/checkouts/app" in out


def test_the_history_file_lands_in_the_index_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A per-machine note about this workspace, beside state.json.
    drive([], monkeypatch, tmp_path)
    assert (tmp_path / "state").is_dir()


def test_ctrl_c_abandons_the_line_and_ctrl_d_ends_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Every REPL agrees on this pair, and getting it backwards means
    # losing a session to a mistyped character.
    class _Interrupting:
        def __init__(self) -> None:
            self.calls = 0

        def prompt(self, _: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            if self.calls == 2:
                return ""  # a blank line is also just skipped
            raise EOFError

    session = _Interrupting()
    stream = io.StringIO()
    monkeypatch.setattr(
        "wsindex.shell.console",
        lambda **_: Console(file=stream, force_terminal=False, width=100, no_color=True),
    )
    monkeypatch.setattr("prompt_toolkit.PromptSession", lambda **_: session)

    run(_NeverSearched(), history_dir=tmp_path / "state")  # type: ignore[arg-type]

    assert session.calls == 3  # it kept going after both


class _NeverSearched:
    def search(self, *args: Any, **kwargs: Any) -> list[Hit]:
        raise AssertionError("nothing should have been searched")


def test_an_unknown_repo_is_a_sentence_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(
        "wsindex.shell.console",
        lambda **_: Console(file=stream, force_terminal=False, width=100, no_color=True),
    )
    monkeypatch.setattr(
        "prompt_toolkit.PromptSession", lambda **_: _ScriptedSession(["x --repo nope"])
    )

    class _Strict:
        def search(self, *args: Any, **kwargs: Any) -> list[Hit]:
            raise ValueError("unknown repo id: 'nope'")

    run(_Strict(), history_dir=tmp_path / "state")  # type: ignore[arg-type]

    assert "unknown repo id" in stream.getvalue()


def test_open_runs_the_editor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EDITOR", "vi")
    Config.reset()
    Config.default("ws").add_repo("app", path="/checkouts/app")
    ran: list[list[str]] = []
    monkeypatch.setattr("subprocess.run", lambda argv, **_: ran.append(argv))

    out = drive(["question", ":open 1"], monkeypatch, tmp_path)

    assert ran == [["vi", "+41", "/checkouts/app/src/main.py"]]
    assert "vi +41" in out


def test_an_editor_that_will_not_run_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EDITOR", "no-such-editor")
    Config.reset()
    Config.default("ws").add_repo("app", path="/checkouts/app")

    def explode(argv: list[str], **_: Any) -> None:
        raise OSError("No such file or directory")

    monkeypatch.setattr("subprocess.run", explode)

    assert "could not run the editor" in drive(["question", ":open 1"], monkeypatch, tmp_path)


def test_open_without_a_number_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert "between 1 and 1" in drive(["question", ":open"], monkeypatch, tmp_path)


def test_hit_zero_is_not_the_last_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Python would read `hits[-1]` and show the wrong chunk, silently.
    assert "between 1 and 1" in drive(["question", "0"], monkeypatch, tmp_path)
