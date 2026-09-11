"""The command line: one table of contents, and the commands themselves.

Every invocation is a fresh process. Two kinds of state outlive it: the
human-edited `wsindex.toml`, read on every command, and the index
database, which the tool owns and which cannot be safely hand-edited.

Each command asks the config for the workspace, does one thing, saves if
it mutated anything, and speaks human: expected failures go to stderr
with exit code 1, and a traceback in the output is always a bug.

That rule is about the default, not about forbidding the truth.
`WSINDEX_DEBUG=1` opens a door: the traceback comes through, the
library's own log records reach stderr, and the parts that run in
threads run in one. It is a variable rather than a flag because a
command has usually already failed by the time you want it — a variable
can be set and the same line repeated.

Commands live in modules beside this one, grouped by what a person came
to do, and are registered here rather than decorated in place — so the
whole surface is one list, and no command module imports this one.
"""

import logging
import os
import sys

import typer

from wsindex.cli.composition import build_pipeline, build_store
from wsindex.cli.external import fetch
from wsindex.cli.indexing import compact, index, sync
from wsindex.cli.interfaces import mcp, serve, shell
from wsindex.cli.searching import refs, search, why
from wsindex.cli.workspace import (
    add_repo,
    domains,
    dupes,
    explain,
    init,
    stats,
    status,
)
from wsindex.config import Config

DEBUG_ENV = "WSINDEX_DEBUG"
"""Set to any non-empty value to trade the tidy output for the whole
truth. Named like `WSINDEX_PLAIN`, and for the same reason: the one
thing a person needs when the default behaviour is in their way."""


def debugging() -> bool:
    """Whether the user asked for everything this run knows."""
    return bool(os.environ.get(DEBUG_ENV))


app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """wsindex — semantic search across the repositories of a workspace."""
    # One wsindex process serves exactly one command, so `Config()` loads
    # once and stays cached on the class. The reset matters only when
    # several commands share a process (the CliRunner in the test suite):
    # it keeps every command starting from disk state, as it does in
    # production.
    Config.reset()


for command in (
    init,
    add_repo,
    status,
    explain,
    domains,
    dupes,
    stats,
    index,
    search,
    sync,
    fetch,
    refs,
    why,
    shell,
    mcp,
    serve,
    compact,
):
    app.command()(command)


def run() -> None:
    """Console-script entry point: a library error is a message, not a trace.

    Commands catch what they expect — an unknown repo id, a directory
    that is not a checkout. What is left are the failures the engine
    raises from wherever it happens to notice them: a missing optional
    dependency, a model whose vectors are the wrong width. Those used to
    reach the terminal as a traceback, which this CLI has always called a
    bug in itself.

    `typer.Exit` and `click`'s own exits are `SystemExit`, so they pass
    through untouched.

    With `WSINDEX_DEBUG` set, nothing is caught at all: a RuntimeError
    may be a bug rather than a user's mistake, and one line is then
    exactly the wrong amount of information. `GitCommandError`,
    `ConnectorError` and everything LanceDB or torch raise are all
    RuntimeErrors, so this was every last frame anyone had.
    """
    if debugging():
        _open_the_door()
        app()
        return
    try:
        app()
    except RuntimeError as exc:
        typer.echo(f"error: {exc}{_advice(exc)}", err=True)
        raise SystemExit(1) from exc


def _open_the_door() -> None:
    """Turn on everything a person debugging this would want.

    Three things, because they are the three that were missing: the
    library's log records (silent by default — a log line is not an
    interface for somebody at a terminal), the level that makes them
    detailed, and single-threaded blame, since a breakpoint in a worker
    thread is a breakpoint in the wrong place.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("wsindex").setLevel(logging.DEBUG)

    from wsindex.ingest import commits

    commits.BLAME_WORKERS = 1


def _advice(exc: RuntimeError) -> str:
    """One more sentence, when the error alone leaves nowhere to go.

    A damaged store answers in its backend's own words — `lance error:
    LanceError(IO): Generic memory error: Invalid range 0..642` — which
    says what broke inside and nothing about what to do. There *is*
    something to do, and it is cheap: an index is derived data, so
    throwing it away costs one re-index. Same sentence the pre-ADR-7
    config check uses, for the same reason.
    """
    if "lance error" not in str(exc).lower():
        return ""
    from wsindex.config import Config

    return (
        f"\nhint: the index looks damaged — remove {Config().index_dir} and run "
        "`wsindex index` (ids are deterministic, re-indexing is cheap)"
    )


__all__ = ["DEBUG_ENV", "app", "build_pipeline", "build_store", "debugging", "run"]
