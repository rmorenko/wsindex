"""Command-line interface and the composition root.

Every invocation is a fresh process. State shared between commands has
two very different owners:

- `wsindex.toml` in the CWD — the workspace declaration (repo list,
  backend choice). Human-edited, read on every command, small.
- `.wsindex/` — the index database. Tool-managed, accumulates chunks
  and embeddings across `index` runs; deduplication reads it before
  the (expensive) embedding step, which is why re-indexing is
  incremental. Cannot be safely hand-edited or naively copied.

Each command loads the config, does one thing, saves if it mutated
anything, and speaks human: expected failures go to stderr and exit
with code 1, a traceback in the output is always a bug.
"""

import os
from pathlib import Path
from typing import Annotated, assert_never

import typer

from wsindex.config import Backend, Config, Provider, load_config, save_config
from wsindex.embed.embedder import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from wsindex.pipeline import Pipeline
from wsindex.store.base import VectorStore
from wsindex.store.local import LocalStore
from wsindex.store.tensorus import TensorusStore

WSINDEX_TOML = "wsindex.toml"
INDEX_DIR = ".wsindex"

app = typer.Typer(no_args_is_help=True)


def _load_config() -> Config:
    """Load wsindex.toml from the CWD or abort the command with exit code 1."""
    path = Path(WSINDEX_TOML)
    if not path.exists():
        typer.echo("error: no wsindex.toml here — run `wsindex init <name>` first", err=True)
        raise typer.Exit(code=1)
    return load_config(path)


def _build_pipeline(config: Config) -> Pipeline:
    """Composition root: the only place that turns config strings into objects."""
    store: VectorStore
    embedder: Embedder
    match config.backend:
        case Backend.LOCAL:
            match config.provider:
                case Provider.SENTENCE_TRANSFORMERS:
                    embedder = SentenceTransformerEmbedder(model_name=config.model)
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
            store = LocalStore(root=Path(INDEX_DIR), embedder=embedder)
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
) -> None:
    """Create wsindex.toml in the current directory.

    Refuses to overwrite an existing config. Defaults to the local backend
    so a fresh workspace works offline; the tensorus backend needs a
    running server (docker compose up) and TENSORUS_API_KEY in the env.
    """
    path = Path(WSINDEX_TOML)
    if path.exists():
        typer.echo("error: wsindex.toml already exists here", err=True)
        raise typer.Exit(code=1)
    config = Config.default_config(name)
    config.backend = backend
    config.provider = provider
    save_config(config=config, path=path)
    typer.echo(f"created {WSINDEX_TOML}: workspace '{name}', backend '{backend.value}'")


@app.command()
def add_repo(repo_id: str, path: str) -> None:
    """Register a repository; its id becomes the dataset name."""
    config = _load_config()
    try:
        config.add_repo(repo_id, path=path)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    save_config(config=config, path=Path(WSINDEX_TOML))
    typer.echo(f"added repo '{repo_id}' -> {path}")


@app.command()
def index() -> None:
    """Walk, chunk and embed every configured repo into the store."""
    pipeline = _build_pipeline(_load_config())
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
    pipeline = _build_pipeline(_load_config())
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
    config = _load_config()
    typer.echo(f"workspace: {config.name}")
    typer.echo(f"backend: {config.backend.value}")
    if not config.repos:
        typer.echo("repos: none — add one with `wsindex add-repo <id> <path>`")
        return
    typer.echo("repos:")
    for repo in config.repos:
        typer.echo(f"  {repo.id} -> {repo.path}")
