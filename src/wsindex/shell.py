"""`wsindex shell`: ask many questions of one loaded model.

The command that fixes the CLI's real cost. A
`wsindex search` spends most of its seconds before it searches anything:
building the embedder means loading a model, and opening the store means
reading a manifest. Ask three questions and you pay for all of that three
times. Here it is paid once and the loop is as fast as the search.

What the loop offers, and nothing more:

- **History**, on disk, so yesterday's query is one arrow-up away.
- **Completion** of the flags and of the repo ids, from the config — a
  repo id is the one thing a person is guaranteed not to remember
  exactly.
- **Picking a hit by number**, then reading the whole chunk with its
  syntax highlighted, or opening it in `$EDITOR` at the right line.

Deliberately not a framework. There is no command language beyond a few
words starting with `:`; a query is anything else. Growing a parser here
would be building a second CLI inside the first one, and `typer` already
has that job.

The query flags are the search command's, because the shell is another
adapter over the same library — the same rule the HTTP server keeps
(ADR-10). `--lang python` means in here exactly what it means out there.
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
    meta = hit.metadata
    # The store keeps a repo id and a repo-relative path — that pairing is
    # the chunk's identity (ARCH §4). Where that repo sits on this machine
    # is the config's business, so that is where the root comes from.
    roots = {repo.id: repo.path for repo in Config().repos}
    path = str(Path(roots.get(str(meta["repo"]), ".")) / str(meta["path"]))
    line = int(meta["start_line"])
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


def run(pipeline: Pipeline, *, history_dir: Path) -> None:
    """The loop itself: read a line, answer it, repeat.

    Args:
        pipeline: Already built — that is the whole point of this command.
        history_dir: Where to keep the input history.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    out = console()
    history_dir.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / HISTORY_FILE)),
        completer=_completer(),
    )
    name = Config().name
    out.print(f"wsindex [bold]{name}[/] — model loaded once, ask away. :help for the rest.")
    hits: list[Hit] = []

    while True:
        try:
            line = session.prompt("wsindex> ").strip()
        except KeyboardInterrupt:
            # Ctrl-C abandons the line, like every other REPL; Ctrl-D
            # ends the session, which is what EOFError below means.
            continue
        except EOFError:
            return
        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            return
        if line in (":help", ":h", "?"):
            out.print(_HELP)
            continue
        if line == ":repos":
            for repo in Config().repos:
                out.print(f"  [bold]{repo.id}[/]  [dim]{repo.path}[/]")
            continue
        if _show(out, line, hits):
            continue
        try:
            query = parse(line)
        except ShellError as exc:
            out.print(f"[red]{exc}[/]")
            continue
        try:
            hits = list(
                pipeline.search(query.text, k=query.k, repo=query.repo, filters=query.filters)
            )
        except ValueError as exc:
            out.print(f"[red]{exc}[/]")
            continue
        if not hits:
            out.print("[dim]no results[/]")
            continue
        render_hits(hits, target=out)


def _show(console_out: Console, line: str, hits: list[Hit]) -> bool:
    """Handle the hit-picking words; True when the line was one of them.

    Split out because it is the only stateful part of the loop — it reads
    the previous answer — and because it is where every "which hit did
    you mean" mistake lives.
    """
    words = line.split()
    verb = words[0]
    if verb in (":open", ":show"):
        argument = words[1] if len(words) > 1 else ""
    elif verb.isdigit():
        verb, argument = ":show", verb
    else:
        return False

    if not hits:
        console_out.print("[red]nothing to pick from — search first[/]")
        return True
    try:
        chosen = hits[int(argument) - 1]
        if int(argument) < 1:
            raise IndexError
    except (ValueError, IndexError):
        console_out.print(f"[red]pick a hit between 1 and {len(hits)}[/]")
        return True

    if verb == ":open":
        argv = open_in_editor(chosen)
        console_out.print(f"[dim]{' '.join(argv)}[/]")
        try:
            subprocess.run(argv, check=False)
        except OSError as exc:
            console_out.print(f"[red]could not run the editor — {exc}[/]")
        return True

    meta = chosen.metadata
    from rich.syntax import Syntax

    console_out.print(
        f"[bold]{meta['repo']}[/]/{meta['path']}:{meta['start_line']}-{meta['end_line']}"
    )
    console_out.print(
        Syntax(
            str(meta["text"]),
            str(meta.get("lang") or "text"),
            theme="ansi_dark",
            line_numbers=True,
            start_line=int(meta["start_line"]),
            word_wrap=True,
            background_color="default",
        )
    )
    return True


__all__ = ["HISTORY_FILE", "Query", "ShellError", "open_in_editor", "parse", "run"]
