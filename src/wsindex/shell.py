"""`wsindex shell`: ask many questions of one loaded model.

A `wsindex search` spends most of its seconds before it searches
anything: loading a model, opening the store. Ask three questions and you
pay for that three times. Here it is paid once.

History on disk, completion over flags and repo ids, a hit shown in full
by number, `:open` into `$EDITOR` at the right line.

Deliberately not a framework: there is no command language beyond a few
words starting with `:`, and a query is anything else. The query flags
are `wsindex search`'s, because this is another adapter over the same
library — `--lang python` means the same thing in both.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wsindex.config import Config
from wsindex.model import Hit, Kind, SearchFilter
from wsindex.ui import console, render_hits

if TYPE_CHECKING:  # pragma: no cover - import-time only, for annotations
    from prompt_toolkit.completion import Completer
    from rich.console import Console

    from wsindex.pipeline import Pipeline

HISTORY_FILE = "shell_history"
"""Kept in the index directory, beside `state.json`: it is a per-machine
note about this workspace, which is exactly what that directory holds."""

COMMANDS = (":help", ":repos", ":open", ":show", ":quit")

_HELP = """\
Type a question to search. Flags are the ones `wsindex search` takes:

  how are chunks deduplicated --lang python -k 5
  --repo app --kind config where is the port set

Then:
  <number>          show that hit in full
  :open <number>    open it in $EDITOR at its line
  :repos            list the repositories of this workspace
  :help  :quit      this, and out (Ctrl-D also works)
"""


@dataclass(frozen=True, kw_only=True)
class Query:
    """A parsed line: the words to search for, and how to narrow it."""

    text: str
    k: int = 10
    repo: str | None = None
    filters: SearchFilter | None = None


class ShellError(ValueError):
    """A line the shell could not make sense of. Says so and loops on."""


def parse(line: str) -> Query:
    """Turn one input line into a query, flags and all.

    Flags may appear anywhere, because people type them where they think
    of them — `--lang python how does X work` and the reverse are the
    same question.

    Args:
        line: What the user typed.

    Returns:
        The query.

    Raises:
        ShellError: A flag is unknown, missing its value, or unusable.
    """
    try:
        words = shlex.split(line)
    except ValueError as exc:  # an unbalanced quote
        raise ShellError(str(exc)) from exc

    terms: list[str] = []
    k = 10
    repo: str | None = None
    langs: list[str] = []
    kinds: list[Kind] = []
    path: str | None = None
    symbol: str | None = None

    index = 0
    while index < len(words):
        word = words[index]
        if not word.startswith("--") and word not in ("-k",):
            terms.append(word)
            index += 1
            continue
        if index + 1 >= len(words):
            raise ShellError(f"{word} needs a value")
        value = words[index + 1]
        index += 2
        match word:
            case "-k" | "--top":
                try:
                    k = int(value)
                except ValueError as exc:
                    raise ShellError(f"-k wants a number, got {value!r}") from exc
            case "--repo":
                repo = value
            case "--lang":
                langs.append(value)
            case "--kind":
                try:
                    kinds.append(Kind(value))
                except ValueError as exc:
                    raise ShellError(
                        f"--kind wants one of {', '.join(k.value for k in Kind)}, got {value!r}"
                    ) from exc
            case "--path":
                path = value
            case "--symbol":
                symbol = value
            case _:
                raise ShellError(f"unknown flag {word}")

    if not terms:
        raise ShellError("nothing to search for")
    candidate = SearchFilter(lang=tuple(langs), kind=tuple(kinds), path=path, symbol=symbol)
    return Query(
        text=" ".join(terms),
        k=k,
        repo=repo,
        filters=None if candidate.is_empty else candidate,
    )


def open_in_editor(hit: Hit) -> list[str]:
    """The command that opens a hit where its author would want it.

    Editors disagree about how to be told a line, and getting it wrong
    means opening at the top of a two-thousand-line file. The three
    spellings below cover what people actually have set; anything else
    gets the file, which is still better than nothing.

    Args:
        hit: The hit to open.

    Returns:
        argv for the editor, unexecuted — so this stays testable and the
        caller decides whether to run it.
    """
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    # A hit knows its repo id and its repo-relative path. Where that repo
    # sits on this machine is the config's business, so that is where the
    # root comes from.
    roots = {repo.id: repo.path for repo in Config().repos}
    path = str(Path(roots.get(hit.repo, ".")) / hit.path)
    line = hit.start_line
    argv = shlex.split(editor)
    name = Path(argv[0]).name
    if name in ("vi", "vim", "nvim", "nano", "kak"):
        return [*argv, f"+{line}", path]
    if name in ("code", "code-insiders", "cursor", "windsurf"):
        return [*argv, "--goto", f"{path}:{line}"]
    if name in ("subl", "idea", "pycharm", "zed"):
        return [*argv, f"{path}:{line}"]
    return [*argv, path]


def _completer() -> Completer:
    """Completion over the shell's words and this workspace's repo ids."""
    from prompt_toolkit.completion import WordCompleter

    repos = [repo.id for repo in Config().repos]
    return WordCompleter(
        [*COMMANDS, "--repo", "--lang", "--kind", "--path", "--symbol", "-k", *repos],
        ignore_case=True,
    )


class _Session:
    """One shell session: the console, the pipeline, and the last answer.

    A class rather than a closure over locals because the loop *has*
    state — which hits were shown last — and `:open 2` is a question
    about it. Naming that state is cheaper than threading it through.
    """

    def __init__(self, pipeline: Pipeline, out: Console) -> None:
        """Bind a session to its engine and its output.

        Args:
            pipeline: Already built — that is the point of this command.
            out: Where answers go.
        """
        self.pipeline = pipeline
        self.out = out
        self.hits: list[Hit] = []
        self.last_query = ""

    def handle(self, line: str) -> bool:
        """Answer one line. False means the session should end.

        The dispatcher, and nothing else: each branch is one word and one
        call, so what the shell understands can be read in one place.
        """
        if line in (":quit", ":q", ":exit"):
            return False
        if line in (":help", ":h", "?"):
            self.out.print(_HELP)
        elif line == ":repos":
            self.show_repos()
        elif not self.pick(line):
            self.search(line)
        return True

    def show_repos(self) -> None:
        """The workspace's repositories, id and path."""
        for repo in Config().repos:
            self.out.print(f"  [bold]{repo.id}[/]  [dim]{repo.path}[/]")

    def search(self, line: str) -> None:
        """Parse a line as a query, run it, show what came back."""
        try:
            query = parse(line)
            self.last_query = query.text
            self.hits = list(
                self.pipeline.search(query.text, k=query.k, repo=query.repo, filters=query.filters)
            )
            skipped = self.pipeline.unsearched(query.repo)
        except (ShellError, ValueError) as exc:
            # A bad line is a sentence, not an exit: the loop's whole job
            # is to keep going.
            self.out.print(f"[red]{exc}[/]")
            return
        if skipped:
            # The CLI, the API and the MCP tools all say this; the shell
            # was the one adapter over the same search that did not. A
            # repo nobody indexed takes no part in any answer, and an
            # answer that skipped half the workspace must not look like
            # one that did not.
            self.out.print(f"[yellow]not searched (never indexed): {', '.join(skipped)}[/]")
        if not self.hits:
            self.out.print("[dim]no results[/]")
            return
        render_hits(self.hits, target=self.out)

    def pick(self, line: str) -> bool:
        """Handle a line that names a hit; False when it names none.

        The only stateful reading in the shell — it answers about the
        previous result — and where every "which hit did you mean"
        mistake lives.
        """
        words = line.split()
        verb, argument = words[0], (words[1] if len(words) > 1 else "")
        if verb.isdigit():
            verb, argument = ":show", verb
        elif verb not in (":open", ":show"):
            return False

        chosen = self._chosen(argument)
        if chosen is None:
            return True
        # The only signal in this project that anybody found what they
        # were looking for. A search tells you what came back; a pick
        # tells you which of it was right, and there is nowhere else to
        # learn that. It is why the shell was worth building before the
        # analytics were.
        if self.pipeline.stats is not None:
            self.pipeline.stats.picked(
                self.last_query, rank=self.hits.index(chosen) + 1, chunk_id=chosen.native_id
            )
        if verb == ":open":
            self.open(chosen)
        else:
            self.show(chosen)
        return True

    def _chosen(self, argument: str) -> Hit | None:
        """The hit a number names, or None after saying why not."""
        if not self.hits:
            self.out.print("[red]nothing to pick from — search first[/]")
            return None
        try:
            position = int(argument)
            if position < 1:
                # Python would read hits[-1] and show the wrong chunk.
                raise IndexError
            return self.hits[position - 1]
        except (ValueError, IndexError):
            self.out.print(f"[red]pick a hit between 1 and {len(self.hits)}[/]")
            return None

    def open(self, hit: Hit) -> None:
        """Hand one hit to `$EDITOR` at its line."""
        argv = open_in_editor(hit)
        self.out.print(f"[dim]{' '.join(argv)}[/]")
        try:
            subprocess.run(argv, check=False)
        except OSError as exc:
            self.out.print(f"[red]could not run the editor — {exc}[/]")

    def show(self, hit: Hit) -> None:
        """Print one hit whole, highlighted as its own language."""
        from rich.syntax import Syntax

        self.out.print(f"[bold]{hit.repo}[/]/{hit.path}:{hit.start_line}-{hit.end_line}")
        self.out.print(
            Syntax(
                hit.text,
                hit.lang or "text",
                theme="ansi_dark",
                line_numbers=True,
                start_line=hit.start_line,
                word_wrap=True,
                background_color="default",
            )
        )


def run(pipeline: Pipeline, *, history_dir: Path) -> None:
    """Read a line, answer it, repeat.

    The loop and nothing else: what a line *means* is `_Session.handle`.

    Args:
        pipeline: Already built — that is the point of this command.
        history_dir: Where to keep the input history.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    history_dir.mkdir(parents=True, exist_ok=True)
    prompt: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / HISTORY_FILE)),
        completer=_completer(),
    )
    session = _Session(pipeline, console())
    session.out.print(
        f"wsindex [bold]{Config().name}[/] — model loaded once, ask away. :help for the rest."
    )
    while True:
        try:
            line = prompt.prompt("wsindex> ").strip()
        except KeyboardInterrupt:
            # Ctrl-C abandons the line, like every other REPL; Ctrl-D
            # ends the session, which is the EOFError below.
            continue
        except EOFError:
            return
        if line and not session.handle(line):
            return


__all__ = ["HISTORY_FILE", "Query", "ShellError", "open_in_editor", "parse", "run"]
