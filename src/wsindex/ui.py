"""How the CLI shows things, and when it is allowed to be pretty.

Этап 12's first step. One rule governs the whole module:

**Rich output only on a terminal.** `wsindex search x | grep foo` and
`wsindex search x > hits.txt` have to keep working, and a table drawn
with box-drawing characters is not something to grep. So a pipe, a file,
a CI log and `NO_COLOR` all get exactly the output this CLI has always
produced — one line per hit, `repo/path:start-end  score  first line` —
and a human at a terminal gets a table with a syntax-highlighted
snippet. Neither is an approximation of the other; they are for
different readers.

That rule is also what kept this step from touching a single existing
test: `CliRunner` is not a terminal, so every assertion about output
still describes what those callers see.

`lang` is already on every chunk, so highlighting costs nothing but
asking for it — the chunker recorded the language when it read the file
and the store carried it through.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only, for annotations
    from rich.console import Console

    from wsindex.model import Hit

PLAIN_ENV = "WSINDEX_PLAIN"
"""Set to any non-empty value to force plain output on a terminal too.
For a demo recording, a screenshot with a fixed width, or a terminal that
lies about what it supports."""

_SNIPPET_LINES = 6
"""How much of a chunk the table shows. A chunk can be a hundred lines;
the question a hit answers is "is this the one", and the first few lines
answer it. `wsindex shell` is where the whole thing gets read."""


def console(*, stderr: bool = False) -> Console:
    """A rich console for this process."""
    from rich.console import Console

    return Console(stderr=stderr, soft_wrap=False)


def wants_rich(target: Console | None = None) -> bool:
    """True when the caller is a human at a terminal.

    Three ways to say no, and all of them are somebody's real setup: the
    output is not a terminal (a pipe, a file, a CI log), `NO_COLOR` is
    set (the informal standard), or `WSINDEX_PLAIN` is set (ours, for
    when the terminal is lying).
    """
    if os.environ.get(PLAIN_ENV) or os.environ.get("NO_COLOR"):
        return False
    return (target or console()).is_terminal


def plain_hit(hit: Hit) -> str:
    """One hit as one greppable line — the format this CLI has always had."""
    meta = hit.metadata
    first_line = str(meta["text"]).splitlines()[0]
    return (
        f"{meta['repo']}/{meta['path']}:{meta['start_line']}-{meta['end_line']}"
        f"  {hit.score:.3f}  {first_line}"
    )


def _snippet(hit: Hit) -> Any:
    """The chunk's opening lines, highlighted as its own language."""
    from rich.syntax import Syntax

    meta = hit.metadata
    lines = str(meta["text"]).splitlines()[:_SNIPPET_LINES]
    return Syntax(
        "\n".join(lines),
        str(meta.get("lang") or "text"),
        theme="ansi_dark",
        # The real line numbers, not 1..n: a snippet whose numbers do not
        # match the file is worse than one with no numbers at all.
        line_numbers=True,
        start_line=int(meta["start_line"]),
        word_wrap=True,
        background_color="default",
    )


def render_hits(hits: Sequence[Hit], *, target: Console | None = None) -> None:
    """Print search results, in the shape the reader can use.

    Args:
        hits: What the pipeline returned, best first.
        target: Console to write to; built if not given.
    """
    out = target or console()
    if not wants_rich(out):
        for hit in hits:
            out.print(plain_hit(hit), highlight=False, markup=False, soft_wrap=True)
        return

    from rich.table import Table

    table = Table(box=None, padding=(0, 1, 1, 0), show_header=True, header_style="dim")
    table.add_column("#", justify="right", style="dim", width=2)
    table.add_column("score", justify="right", width=5)
    table.add_column("where", overflow="fold")
    table.add_column("what", overflow="fold")
    for position, hit in enumerate(hits, start=1):
        meta = hit.metadata
        where = (
            f"[bold]{meta['repo']}[/]/{meta['path']}\n"
            f"[dim]{meta['start_line']}-{meta['end_line']}"
            f"{'  ' + str(meta['symbol']) if meta.get('symbol') else ''}[/]"
        )
        table.add_row(str(position), f"{hit.score:.3f}", where, _snippet(hit))
    out.print(table)


@contextmanager
def progress(label: str, *, target: Console | None = None) -> Iterator[Any]:
    """Show that something long is happening, or stay silent off a terminal.

    Indexing is seconds of nothing, which reads as a hang. What this
    reports is which repository is being read, because that is what the
    pipeline knows — turning it into a sentence is this module's job, not
    the engine's.

    Yields:
        A callable taking the current repo id, or None when plain. The
        caller passes it straight to `Pipeline.index`, which tolerates
        None, so there is no branch on the calling side.
    """
    out = target or console(stderr=True)
    if not wants_rich(out):
        yield None
        return

    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=out,
        transient=True,  # leave no bar behind: the report is the output
    ) as bar:
        task = bar.add_task(label)

        def report(repo_id: str) -> None:
            bar.update(task, description=f"{label} — {repo_id}")

        yield report


__all__ = ["PLAIN_ENV", "console", "plain_hit", "progress", "render_hits", "wants_rich"]
