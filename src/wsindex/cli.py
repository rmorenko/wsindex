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

Each command loads the config, does one thing, saves if it mutated
anything, and speaks human: expected failures go to stderr and exit
with code 1, a traceback in the output is always a bug.
"""

import os
from typing import Annotated, assert_never

import typer

from wsindex.config import Backend, Config, Provider, load_config, save_config
from wsindex.embed.embedder import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from wsindex.paths import (
    ConfigLocation,
    find_config,
    resolve_cache_dir,
    resolve_index_dir,
    searched_paths,
    user_config_file,
    workspace_config_path,
)
from wsindex.pipeline import Pipeline
from wsindex.store.base import VectorStore
from wsindex.store.local import LocalStore
from wsindex.store.tensorus import TensorusStore

app = typer.Typer(no_args_is_help=True)


def _load_config() -> tuple[Config, ConfigLocation]:
    """Locate and load a wsindex config, or abort with exit code 1.

    Returns:
        The parsed Config and the ConfigLocation that found it — the
        location is needed downstream to resolve the index directory
        and to save mutations back to the same file.
    """
    location = find_config()
    if location is None:
        typer.echo("error: no wsindex config found. Checked:", err=True)
        for line in searched_paths():
            typer.echo(f"  - {line}", err=True)
        typer.echo(
            "Run `wsindex init <name>` (workspace) " "or `wsindex init --user <name>` (user).",
            err=True,
        )
        raise typer.Exit(code=1)
    return load_config(location.path), location


def _build_pipeline(config: Config, location: ConfigLocation) -> Pipeline:
    """Composition root: the only place that turns config strings into objects."""
    store: VectorStore
    embedder: Embedder
    match config.backend:
        case Backend.LOCAL:
            match config.provider:
                case Provider.SENTENCE_TRANSFORMERS:
                    embedder = SentenceTransformerEmbedder(
                        model_name=config.model,
                        cache_folder=resolve_cache_dir() / "models",
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
            store = LocalStore(root=resolve_index_dir(location, config.name), embedder=embedder)
        case Backend.TENSORUS:
            api_key = os.environ.get("TENSORUS_API_KEY")
            if not api_key:
                typer.echo(
                    "error: TENSORUS_API_KEY is not set — export it before using "
                    "the tensorus backend",
                    err=True,
                )
                raise typer.Exit(code=1)
            store = TensorusStore(
                base_url=config.base_url, api_key=api_key, model_name=config.model
            )
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(config.backend)
    return Pipeline(config=config, store=store)


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
    config = Config.default_config(name)
    config.backend = backend
    config.provider = provider
    save_config(config=config, path=target)
    typer.echo(f"created {target}: workspace '{name}', backend '{backend.value}'")


@app.command()
def add_repo(repo_id: str, path: str) -> None:
    """Register a repository; its id becomes the dataset name."""
    config, location = _load_config()
    try:
        config.add_repo(repo_id, path=path)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    save_config(config=config, path=location.path)
    typer.echo(f"added repo '{repo_id}' -> {path}")


@app.command()
def index() -> None:
    """Walk, chunk and embed every configured repo into the store."""
    config, location = _load_config()
    pipeline = _build_pipeline(config, location)
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
    config, location = _load_config()
    pipeline = _build_pipeline(config, location)
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
    config, location = _load_config()
    typer.echo(f"config: {location.path} ({location.mode.value})")
    typer.echo(f"workspace: {config.name}")
    typer.echo(f"backend: {config.backend.value}")
    if not config.repos:
        typer.echo("repos: none — add one with `wsindex add-repo <id> <path>`")
        return
    typer.echo("repos:")
    for repo in config.repos:
        typer.echo(f"  {repo.id} -> {repo.path}")
