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

import os
from typing import assert_never

import typer

from wsindex.config import Backend, Config, LinksBackend, Provider
from wsindex.embed import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from wsindex.links import LinkStore
from wsindex.paths import ConfigLocation, make_index_dir, resolve_cache_dir, searched_paths
from wsindex.pipeline import Pipeline
from wsindex.rank.reranker import CrossEncoderReranker
from wsindex.stats import SearchLog
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
    # Before the store, not after: `LanceDBStore` connects eagerly, and
    # connecting creates the directory at the process umask. Whoever
    # makes it first decides its permissions, and the two callers that
    # care (`LinkStore`, `IndexState.save`) both arrive second.
    make_index_dir(config.index_dir)
    store = build_store(config)
    reranker = CrossEncoderReranker(model_name=config.rank_model) if config.rank_enabled else None
    # `state_dir` is index_dir and always will be: the commit each repo
    # was last indexed at is genuinely a note about *this* host, since
    # two machines sit on different branches. Links are not — every
    # field of one is derived from content — which is why they get a
    # backend of their own rather than sharing that assumption.
    return Pipeline(
        store=store,
        state_dir=config.index_dir,
        reranker=reranker,
        links=build_links(config),
        stats=SearchLog(config.index_dir) if config.stats_enabled else None,
    )


def build_links(config: Config) -> LinkStore:
    """Open the link store this workspace asks for.

    Args:
        config: The workspace.

    Returns:
        A store; SQLite unless `[links] backend = "postgres"`.

    Raises:
        typer.Exit: Postgres was asked for and cannot be reached — no
            `dsn_env`, an unset variable, or a refused connection. All
            three are the config's to fix, and starting on SQLite
            instead would silently answer `refs` from a different set of
            links than the one the workspace shares.
    """
    if config.links_backend is not LinksBackend.POSTGRES:
        return LinkStore(config.index_dir)
    name = config.links_dsn_env
    if not name:
        typer.echo(
            'error: [links] backend = "postgres" needs `dsn_env` naming the variable '
            "that holds the connection string (the name, never the string)",
            err=True,
        )
        raise typer.Exit(code=1)
    dsn = os.environ.get(name)
    if not dsn:
        typer.echo(f"error: ${name} is not set, and [links] dsn_env names it", err=True)
        raise typer.Exit(code=1)
    try:
        return LinkStore.postgres(dsn)
    except Exception as exc:
        # Deliberately broad: psycopg raises a family of its own, and
        # every one of them means the same thing to somebody reading
        # this — the database named in the config did not answer.
        typer.echo(f"error: cannot reach the links database — {exc}", err=True)
        raise typer.Exit(code=1) from exc


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

    Split out of `build_pipeline` because `compact` needs a store and
    nothing else — and, now that the embedder loads its model lazily,
    gets one without paying six seconds for a network it never uses.

    The vector column's width comes from the config rather than from the
    model, so opening a store asks nothing of it. The two must agree, and
    the embedder checks that the first time it actually loads: a model
    swapped without changing `dim` fails then, with a sentence naming
    both numbers.
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
                        config.model,
                        cache_folder=resolve_cache_dir() / "models",
                        dim=config.dim,
                    )
                case Provider.FAKE:
                    embedder = FakeEmbedder(dim=config.dim)
                case _:  # pragma: no cover - mypy proves this branch unreachable
                    assert_never(config.provider)
            store = LanceDBStore(uri=config.store_uri, embedder=embedder)
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(config.backend)
    return store
