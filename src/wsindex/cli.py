"""Command-line interface and the composition root.

Every invocation is a fresh process. State shared between commands has
two very different owners:

- `wsindex.toml` — the workspace declaration (repo list, backend
  choice). Human-edited, read on every command, small. Found by the
  four-mode resolver in `wsindex.paths` (workspace-first, XDG-fallback).
- The index database — tool-managed, accumulates chunks and embeddings
  across `index` runs; deduplication reads it before the (expensive)
  embedding step, which is why re-indexing is incremental. Cannot be
  safely hand-edited or naively copied. Its location follows the
  config (`.wsindex/` next to a workspace config; `$XDG_DATA_HOME/
  wsindex/<name>/` for a user or system config).

Each command asks `Config()` for the workspace, does one thing, saves
if it mutated anything, and speaks human: expected failures go to
stderr and exit with code 1, a traceback in the output is always a bug.

`Config` never refuses to exist: with no file anywhere it serves built-in
defaults and says so through `is_default`. It stays silent about it —
having a user to talk to is a property of this module, not of the config
— so telling them is `_config`'s job, and refusing to go on without a
real file is `_require_config_file`'s.
"""

import os
from typing import Annotated, assert_never

import typer

from wsindex.config import Backend, Config, Provider
from wsindex.embed import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from wsindex.paths import (
    ConfigLocation,
    resolve_cache_dir,
    searched_paths,
    user_config_file,
    workspace_config_path,
)
from wsindex.pipeline import Pipeline
from wsindex.store import LocalStore, TensorusStore, VectorStore

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


def _config() -> Config:
    """Load the workspace config, telling the user when there is none.

    `Config()` falls back to built-in defaults instead of failing, which
    is the right call for a library but a silent one — the user asked
    about *their* workspace and would be reading numbers about a
    workspace that does not exist. So the message lives here, where
    there is a terminal to write it to.
    """
    config = Config()
    if config.is_default:
        typer.echo("warning: no wsindex config found. Checked:", err=True)
        for line in searched_paths():
            typer.echo(f"  - {line}", err=True)
        typer.echo("Showing built-in defaults instead.", err=True)
    return config


def _require_config_file(config: Config) -> ConfigLocation:
    """Abort a command that cannot work without a config file on disk.

    `Config()` never fails — a missing file means a warning and the
    built-in defaults (see `wsindex.config`). That is enough for `status`,
    which only reports what it sees, but `index` and `search` need
    somewhere to keep the index and `add-repo` needs somewhere to save.
    The warning already listed where wsindex looked; this only adds the
    way out.
    """
    location = config.location
    if location is None:
        typer.echo(
            "error: this command needs a config file — run `wsindex init <name>` "
            "(workspace) or `wsindex init --user <name>` (user).",
            err=True,
        )
        raise typer.Exit(code=1)
    return location


def _build_pipeline() -> Pipeline:
    """Composition root: decides *which* objects exist, not what they hold.

    Every value that lives in the config — the model, the dimensionality,
    the server URL, the index dir, the repo list, the metric — each object
    now reads for itself from `Config()`. What is left here is the part a
    config cannot do: choosing a class per `backend`/`provider`, reading
    the API key out of the environment (it must never reach a file), and
    turning a bad combination into a human error instead of a traceback.
    """
    config = _config()
    _require_config_file(config)
    store: VectorStore
    embedder: Embedder
    match config.backend:
        case Backend.LOCAL:
            match config.provider:
                case Provider.SENTENCE_TRANSFORMERS:
                    # cache_folder is a paths concern, not a config field:
                    # the model cache is shared by every workspace.
                    embedder = SentenceTransformerEmbedder(
                        cache_folder=resolve_cache_dir() / "models"
                    )
                    if embedder.dim != config.dim:
                        typer.echo(
                            f"error: embedder dim mismatch — config expects {config.dim}, "
                            f"model '{config.model}' produces {embedder.dim}",
                            err=True,
                        )
                        raise typer.Exit(code=1)
                case Provider.FAKE:
                    embedder = FakeEmbedder(dim=config.dim)
                case _:  # pragma: no cover - mypy proves this branch unreachable
                    assert_never(config.provider)
            store = LocalStore(embedder=embedder)
        case Backend.TENSORUS:
            api_key = os.environ.get("TENSORUS_API_KEY")
            if not api_key:
                typer.echo(
                    "error: TENSORUS_API_KEY is not set — export it before using "
                    "the tensorus backend",
                    err=True,
                )
                raise typer.Exit(code=1)
            store = TensorusStore(api_key=api_key)
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(config.backend)
    return Pipeline(store=store)


@app.command()
def init(
    name: str,
    backend: Annotated[Backend, typer.Option(help="Vector store backend")] = Backend.LOCAL,
    provider: Annotated[
        Provider, typer.Option(help="Embeddings provider")
    ] = Provider.SENTENCE_TRANSFORMERS,
    user: Annotated[
        bool,
        typer.Option("--user", help="Install into $XDG_CONFIG_HOME/wsindex/ instead of the CWD"),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing config")] = False,
) -> None:
    """Create a wsindex config.

    Two modes: workspace (default) writes `./wsindex.toml`, so the tool
    lives with the project (like `git init`); `--user` writes
    `$XDG_CONFIG_HOME/wsindex/config.toml`, for `pipx install` and other
    from-anywhere invocations. Both refuse to overwrite unless `--force`
    is given. Defaults to the local backend so a fresh workspace works
    offline; the tensorus backend needs a running server (docker compose
    up) and TENSORUS_API_KEY in the env.
    """
    target = user_config_file() if user else workspace_config_path()
    if target.exists() and not force:
        typer.echo(
            f"error: config already exists at {target} (use --force to overwrite)",
            err=True,
        )
        raise typer.Exit(code=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    config = Config.default(name, backend=backend, provider=provider)
    config.save(target)
    typer.echo(f"created {target}: workspace '{name}', backend '{backend.value}'")


@app.command()
def add_repo(repo_id: str, path: str) -> None:
    """Register a repository; its id becomes the dataset name."""
    config = _config()
    location = _require_config_file(config)
    try:
        config.add_repo(repo_id, path=path)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    config.save(location.path)
    typer.echo(f"added repo '{repo_id}' -> {path}")


@app.command()
def index() -> None:
    """Walk, chunk and embed every configured repo into the store."""
    pipeline = _build_pipeline()
    report = pipeline.index()
    typer.echo(f"files: {report.files}  chunks: {report.chunks}  written: {report.written}")
    if report.missing_repos:
        typer.echo("warning: missing repos: " + ", ".join(report.missing_repos), err=True)


@app.command()
def search(
    query: str,
    top: Annotated[int, typer.Option("--top", "-k", help="How many hits")] = 10,
) -> None:
    """Search all indexed repos, best hits first."""
    pipeline = _build_pipeline()
    hits = pipeline.search(query, k=top)
    if not hits:
        typer.echo("no results")
        return
    for hit in hits:
        meta = hit.metadata
        first_line = str(meta["text"]).splitlines()[0]
        typer.echo(
            f"{meta['repo']}/{meta['path']}:{meta['start_line']}-{meta['end_line']}"
            f"  {hit.score:.3f}  {first_line}"
        )


@app.command()
def status() -> None:
    """Show the workspace: name, backend, registered repos."""
    config = _config()
    location = config.location
    if location is None:
        typer.echo("config: none — showing built-in defaults")
    else:
        typer.echo(f"config: {location.path} ({location.mode.value})")
    typer.echo(f"workspace: {config.name}")
    typer.echo(f"backend: {config.backend.value}")
    if not config.repos:
        typer.echo("repos: none — add one with `wsindex add-repo <id> <path>`")
        return
    typer.echo("repos:")
    for repo in config.repos:
        typer.echo(f"  {repo.id} -> {repo.path}")
