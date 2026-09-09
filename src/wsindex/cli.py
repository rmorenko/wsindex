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

from datetime import timedelta
from pathlib import Path
from typing import Annotated, assert_never

import typer

from wsindex.config import Backend, Config, Provider
from wsindex.embed import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from wsindex.ingest import GitCommandError, NotAGitRepositoryError, sync_repo
from wsindex.model import Kind, SearchFilter
from wsindex.paths import (
    ConfigLocation,
    resolve_cache_dir,
    searched_paths,
    user_config_file,
    workspace_config_path,
)
from wsindex.pipeline import Pipeline
from wsindex.rank.reranker import CrossEncoderReranker
from wsindex.store import LanceDBStore, VectorStore

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

    The values that describe *work* — the repo list, the metric — each
    object now reads for itself from `Config()`. What is left here is the
    part a config cannot do: choosing a class per `backend`/`provider`,
    handing each class the one location it cannot derive (the store uri,
    the model cache), and turning a bad combination into a human error
    instead of a traceback.
    """
    config = _config()
    _require_config_file(config)
    store = _build_store(config)
    reranker = CrossEncoderReranker(model_name=config.rank_model) if config.rank_enabled else None
    # index_dir, not store_uri: the incremental state is a local, per-machine
    # note about how far this host got, even when the vectors live in S3.
    return Pipeline(store=store, state_dir=config.index_dir, reranker=reranker)


def _build_store(config: Config) -> VectorStore:
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
            store = LanceDBStore(uri=config.store_uri, embedder=embedder)
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(config.backend)
    return store


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
    is given.

    The workspace is fully offline once the embedding model is
    downloaded, and its index lands next to the config. Point
    `[store] uri` at s3://... for shared storage instead (credentials
    come from the AWS_* env).
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
def add_repo(
    repo_id: str,
    path: str,
    remote: Annotated[
        str | None,
        typer.Option("--remote", help="Clone url; `wsindex sync` keeps <path> up to date from it"),
    ] = None,
) -> None:
    """Register a repository; its id becomes the dataset name.

    With `--remote` the working copy becomes the workspace's business:
    `wsindex sync` clones it if `path` does not exist yet and
    fast-forwards it afterwards. Without one, `path` is a checkout you
    maintain yourself and sync leaves it alone.
    """
    config = _config()
    location = _require_config_file(config)
    try:
        config.add_repo(repo_id, path=path, remote=remote)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    config.save(location.path)
    suffix = f" (remote: {remote})" if remote else ""
    typer.echo(f"added repo '{repo_id}' -> {path}{suffix}")


@app.command()
def index() -> None:
    """Chunk and embed what changed in every configured repo.

    Incremental against git: a repo that has been indexed before and has
    a clean working tree only pays for the files that changed since. A
    dirty tree costs a full pass, because a commit-to-commit diff cannot
    see uncommitted work.
    """
    pipeline = _build_pipeline()
    try:
        report = pipeline.index()
    except NotAGitRepositoryError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"files: {report.files}  chunks: {report.chunks}  "
        f"written: {report.written}  deleted: {report.deleted}"
    )
    if report.full_repos:
        # Told, not hidden: this is why the run took seconds instead of
        # milliseconds, and the user is the only one who can fix it.
        typer.echo(
            "note: full pass for " + ", ".join(report.full_repos) + " (first index, or a dirty "
            "working tree — commit or stash to go incremental)",
            err=True,
        )
    if report.missing_repos:
        typer.echo("warning: missing repos: " + ", ".join(report.missing_repos), err=True)


@app.command()
def search(
    query: str,
    top: Annotated[int, typer.Option("--top", "-k", help="How many hits")] = 10,
    repo: Annotated[str | None, typer.Option("--repo", help="Restrict to a single repo id")] = None,
    lang: Annotated[
        list[str] | None,
        typer.Option("--lang", help="Restrict to a language (repeat for OR)"),
    ] = None,
    kind: Annotated[
        list[Kind] | None,
        typer.Option("--kind", help="Restrict to a Kind (code/config/doc; repeat for OR)"),
    ] = None,
    path: Annotated[
        str | None,
        typer.Option("--path", help="Path glob (`*`, `?` wildcards)"),
    ] = None,
    symbol: Annotated[
        str | None, typer.Option("--symbol", help="Substring of the chunk symbol")
    ] = None,
) -> None:
    """Search all indexed repos, best hits first.

    Scope flags stack: --repo narrows the dataset list, structural
    filters (--lang/--kind/--path/--symbol) go down to the store as a
    prefilter (top-k over the filtered subset, not slashed out of it).
    """
    pipeline = _build_pipeline()
    candidate = SearchFilter(
        lang=tuple(lang or ()),
        kind=tuple(kind or ()),
        path=path,
        symbol=symbol,
    )
    filters: SearchFilter | None = None if candidate.is_empty else candidate
    try:
        hits = pipeline.search(query, k=top, repo=repo, filters=filters)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
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
def sync(
    no_index: Annotated[
        bool,
        typer.Option("--no-index", help="Update the working copies only, do not re-index"),
    ] = False,
) -> None:
    """Update every repo that has a `remote`, then re-index what changed.

    Clones a repo whose `path` does not exist yet, fast-forwards one that
    does. Repos without a `remote` are checkouts you maintain yourself
    and are left alone.

    Sync never touches work the remote does not have: uncommitted changes
    or local commits make it decline and say so, rather than choosing for
    you. That costs nothing but speed — `index` handles a dirty tree by
    re-reading the whole repo.
    """
    config = _config()
    _require_config_file(config)
    remotes = [repo for repo in config.repos if repo.remote is not None]
    if not remotes:
        typer.echo("no repos have a remote — add one with `wsindex add-repo <id> <path> --remote`")
        return
    skipped = False
    for repo in remotes:
        try:
            outcome = sync_repo(Path(repo.path), remote=str(repo.remote))
        except (NotAGitRepositoryError, GitCommandError) as exc:
            # One unreachable remote must not strand the repos after it,
            # so this is reported per repo instead of aborting the run.
            typer.echo(f"{repo.id}: failed — {exc}", err=True)
            skipped = True
            continue
        stream_err = outcome.is_skip
        typer.echo(f"{repo.id}: {outcome.value}", err=stream_err)
        skipped = skipped or stream_err
    if no_index:
        # Exit code reflects the syncing, which is all this run did.
        raise typer.Exit(code=1 if skipped else 0)
    index()
    if skipped:
        raise typer.Exit(code=1)


@app.command()
def compact(
    keep_days: Annotated[
        float,
        typer.Option(
            "--keep-days",
            help="Keep index history younger than this many days (default: keep none)",
        ),
    ] = 0.0,
) -> None:
    """Reclaim the disk that deleted and rewritten chunks still occupy.

    Deleting a chunk hides it at once but does not free its bytes, and an
    incremental `index` deletes on every run — so the index grows even
    when the workspace does not. This is the pass that shrinks it.

    Manual rather than part of `index`, because it is the one command
    that throws history away: until it runs the store can be rolled back
    to an earlier version, and afterwards it cannot. Use `--keep-days` if
    something else may be reading the same store — a search that started
    before this pass would be reading a version it removes.
    """
    config = _config()
    _require_config_file(config)
    report = _build_store(config).compact(older_than=timedelta(days=keep_days))
    versions = f"{report.versions_before} -> {report.versions_after} versions"
    if report.bytes_freed is None:
        # A remote store cannot be measured from here; saying so beats
        # printing a zero that reads like "nothing happened".
        typer.echo(f"compacted: {versions} (size not measurable for a remote store)")
        return
    typer.echo(
        f"reclaimed {_human_bytes(report.bytes_freed)} "
        f"({_human_bytes(report.bytes_before or 0)} -> "
        f"{_human_bytes(report.bytes_after or 0)}); {versions}"
    )


def _human_bytes(size: int) -> str:
    """Format a byte count the way a person reads it."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable: the loop returns at GB")  # pragma: no cover


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
