"""Building the objects a command needs, and reading the workspace.

The composition root: it decides *which* objects exist, not what they
hold. The values that describe work — the repo list, the metric — each
object reads for itself from `Config`. What is left here is the part a
config cannot do: choosing a class per `backend`/`provider`, handing each
class the one location it cannot derive, and turning a bad combination
into a human error instead of a traceback.

Separate from the commands because it is a different job: a command
answers a question, this decides what will answer it. The server and the
MCP adapter build their pipelines through here too, so no interface can
quietly acquire an engine of its own.
"""

from typing import assert_never

import typer

from wsindex.config import Backend, Config, Provider
from wsindex.embed import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from wsindex.links import LinkStore
from wsindex.paths import ConfigLocation, resolve_cache_dir, searched_paths
from wsindex.pipeline import Pipeline
from wsindex.rank.reranker import CrossEncoderReranker
from wsindex.store import LanceDBStore, VectorStore


def config_or_default() -> Config:
    """Load the workspace config, telling the user when there is none.

    `Config()` falls back to built-in defaults instead of failing, which
    is the right call for a library but a silent one — the user asked
    about *their* workspace and would be reading numbers about a
    workspace that does not exist. So the message lives here, where
    there is a terminal to write it to.
    """
    try:
        config = Config()
    except (KeyError, ValueError) as exc:
        # A malformed file is the user's to fix, not a crash to report.
        # `TOMLDecodeError` is a ValueError, so this covers unparsable
        # TOML as well as a missing section or an unusable value — and a
        # traceback in the output is always a bug (see the module
        # docstring), including this one.
        typer.echo(f"error: cannot read the wsindex config — {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if config.is_default:
        typer.echo("warning: no wsindex config found. Checked:", err=True)
        for line in searched_paths():
            typer.echo(f"  - {line}", err=True)
        typer.echo("Showing built-in defaults instead.", err=True)
    return config


def require_config_file(config: Config) -> ConfigLocation:
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


def build_pipeline() -> Pipeline:
    """Composition root: decides *which* objects exist, not what they hold.

    The values that describe *work* — the repo list, the metric — each
    object now reads for itself from `Config()`. What is left here is the
    part a config cannot do: choosing a class per `backend`/`provider`,
    handing each class the one location it cannot derive (the store uri,
    the model cache), and turning a bad combination into a human error
    instead of a traceback.
    """
    config = config_or_default()
    require_config_file(config)
    store = build_store(config)
    reranker = CrossEncoderReranker(model_name=config.rank_model) if config.rank_enabled else None
    # index_dir, not store_uri: the incremental state and the link database
    # are local, per-machine notes about this host, even when the vectors
    # live in S3.
    return Pipeline(
        store=store,
        state_dir=config.index_dir,
        reranker=reranker,
        links=LinkStore(config.index_dir),
    )


def require_extra(name: str, packages: str) -> None:
    """Turn a missing optional dependency into a sentence, not a traceback.

    Called after the failed import rather than before it: asking whether a
    package is installed and then importing it are two chances to
    disagree, and the import is the one that decides.

    Args:
        name: The extra as it is spelled in `pyproject.toml`.
        packages: What it brings, so the message names what is missing.

    Raises:
        typer.Exit: Always — this is the end of the command.
    """
    typer.echo(
        f"error: this needs the `{name}` extra — `uv sync --extra {name}` ({packages})",
        err=True,
    )
    raise typer.Exit(code=1)


def build_store(config: Config) -> VectorStore:
    """Open the workspace store described by the config.

    Split out of `_build_pipeline` because `compact` needs a store and
    nothing else. It still pays for the embedder: the vector column's
    width comes from `embedder.dim`, so opening the table without one
    would mean a second, subtly different way to describe the same
    schema — the kind of duplication that goes wrong quietly.
    """
    store: VectorStore
    embedder: Embedder
    match config.backend:
        case Backend.LOCAL:
            match config.provider:
                case Provider.SENTENCE_TRANSFORMERS:
                    # cache_folder is a paths concern, not a config field:
                    # the model cache is shared by every workspace.
                    embedder = SentenceTransformerEmbedder(
                        config.model, cache_folder=resolve_cache_dir() / "models"
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
            store = LanceDBStore(uri=config.store_uri, embedder=embedder)
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(config.backend)
    return store
