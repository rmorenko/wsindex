"""The command line: one table of contents, and the commands themselves.

Every invocation is a fresh process. State shared between commands has
two very different owners:

- `wsindex.toml` — the workspace declaration (repo list, backend
  choice). Human-edited, read on every command, small. Found by the
  four-mode resolver in `wsindex.paths` (workspace-first, XDG-fallback).
- The index database — tool-managed, accumulates chunks and embeddings
  across `index` runs; deduplication reads it before the (expensive)
  embedding step, which is why re-indexing is incremental. Cannot be
  safely hand-edited or naively copied. Its location follows the config.

Each command asks the config for the workspace, does one thing, saves if
it mutated anything, and speaks human: expected failures go to stderr
and exit with code 1, a traceback in the output is always a bug.

The commands live in modules beside this one, grouped by what a person
came to do, and are registered here rather than decorated in place. That
keeps the whole surface visible in one list — and keeps every module
free of an import back to this one.
"""

import typer

from wsindex.cli.composition import build_pipeline, build_store
from wsindex.cli.external import fetch
from wsindex.cli.indexing import compact, index, sync
from wsindex.cli.interfaces import mcp, serve, shell
from wsindex.cli.searching import refs, search, why
from wsindex.cli.workspace import add_repo, init, status
from wsindex.config import Config

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

__all__ = ["app", "build_pipeline", "build_store"]
