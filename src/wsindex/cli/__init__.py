"""The command line: one table of contents, and the commands themselves.

Every invocation is a fresh process. Two kinds of state outlive it: the
human-edited `wsindex.toml`, read on every command, and the index
database, which the tool owns and which cannot be safely hand-edited.

Each command asks the config for the workspace, does one thing, saves if
it mutated anything, and speaks human: expected failures go to stderr
with exit code 1, and a traceback in the output is always a bug.

Commands live in modules beside this one, grouped by what a person came
to do, and are registered here rather than decorated in place — so the
whole surface is one list, and no command module imports this one.
"""

import typer

from wsindex.cli.composition import build_pipeline, build_store
from wsindex.cli.external import fetch
from wsindex.cli.indexing import compact, index, sync
from wsindex.cli.interfaces import mcp, serve, shell
from wsindex.cli.searching import refs, search, why
from wsindex.cli.workspace import add_repo, explain, init, status
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
    explain,
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
    """
    try:
        app()
    except RuntimeError as exc:
        typer.echo(f"error: {exc}{_advice(exc)}", err=True)
        raise SystemExit(1) from exc


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


__all__ = ["app", "build_pipeline", "build_store", "run"]
