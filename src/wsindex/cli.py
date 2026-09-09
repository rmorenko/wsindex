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
import re
from datetime import timedelta
from pathlib import Path
from typing import Annotated, assert_never

import typer

from wsindex.config import Backend, Config, Provider, Repository, RepoSource
from wsindex.connectors import ConnectorError, ConnectorSpec, route
from wsindex.embed import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from wsindex.ingest import GitCommandError, NotAGitRepositoryError, sync_repo
from wsindex.links import Edge, LinkKind, LinkStore
from wsindex.model import Hit, Kind, SearchFilter
from wsindex.paths import (
    ConfigLocation,
    resolve_cache_dir,
    searched_paths,
    user_config_file,
    workspace_config_path,
)
from wsindex.pipeline import Pipeline
from wsindex.rank.reranker import CrossEncoderReranker
from wsindex.snapshot import materialize
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
    # index_dir, not store_uri: the incremental state and the link database
    # are local, per-machine notes about this host, even when the vectors
    # live in S3.
    return Pipeline(
        store=store,
        state_dir=config.index_dir,
        reranker=reranker,
        links=LinkStore(config.index_dir),
    )


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
    source: Annotated[
        RepoSource | None,
        typer.Option("--source", help="`connector` for a snapshot repo built from --url documents"),
    ] = None,
    url: Annotated[
        list[str] | None,
        typer.Option("--url", help="Document to materialize (repeat); needs --source connector"),
    ] = None,
    ignore: Annotated[
        list[str] | None,
        typer.Option("--ignore", help="Path glob this repo excludes (repeat)"),
    ] = None,
) -> None:
    """Register a repository; its id becomes the dataset name.

    With `--remote` the working copy becomes the workspace's business:
    `wsindex sync` clones it if `path` does not exist yet and
    fast-forwards it afterwards. Without one, `path` is a checkout you
    maintain yourself and sync leaves it alone.

    With `--source connector` the repo is a snapshot instead: sync
    fetches each `--url` through the configured connectors, writes it as
    markdown and commits. The two are mutually exclusive — a working copy
    has one owner.

    `--ignore` takes a path glob this repo excludes, on top of what the
    walker prunes everywhere. The other per-repo key, `formats`, is a
    suffix table and lives in `wsindex.toml` rather than on a command
    line.
    """
    config = _config()
    location = _require_config_file(config)
    try:
        config.add_repo(
            repo_id,
            path=path,
            remote=remote,
            source=source,
            urls=url or (),
            ignore=ignore or (),
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    config.save(location.path)
    if source is RepoSource.CONNECTOR:
        suffix = f" (snapshot of {len(url or ())} document(s))"
    else:
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
        f"written: {report.written}  deleted: {report.deleted}  "
        f"commits: {report.commits}"
    )
    if report.full_repos:
        # Told, not hidden: this is why the run took seconds instead of
        # milliseconds, and the user is the only one who can fix it.
        typer.echo(
            "note: full pass for "
            + ", ".join(report.full_repos)
            + " — a first index, edited per-repo markup, or uncommitted work "
            "(a commit-to-commit diff cannot see the last one; commit or stash it)",
            err=True,
        )
    if report.missing_repos:
        typer.echo("warning: missing repos: " + ", ".join(report.missing_repos), err=True)
    if pipeline.links is not None:
        drift = pipeline.links.dangling()
        if drift:
            # Reported, not listed: this is the evidence that code and
            # configuration have drifted apart (ADR-9), and a count is
            # enough to send someone looking. Listing them is `wsindex
            # refs`/`why` territory (step 28).
            first = drift[0]
            typer.echo(
                f"drift: {len(drift)} unresolved config reference(s), "
                f"first at {first.repo}/{first.path}:{first.line} -> {first.name}",
                err=True,
            )


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

    A repo with `source = "connector"` is filled the other way: its
    documents are fetched and committed into a snapshot repository, which
    `index` then reads like any other checkout.
    """
    config = _config()
    _require_config_file(config)
    pulled = [repo for repo in config.repos if repo.remote is not None]
    snapshots = [repo for repo in config.repos if repo.is_snapshot]
    if not pulled and not snapshots:
        typer.echo(
            "nothing to sync — give a repo a remote (`wsindex add-repo <id> <path> --remote "
            "<url>`) or make one a snapshot (`--source connector --url <url>`)"
        )
        return
    skipped = False
    for repo in pulled:
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
    for repo in snapshots:
        skipped = _sync_snapshot(repo, config.connectors) or skipped
    if no_index:
        # Exit code reflects the syncing, which is all this run did.
        raise typer.Exit(code=1 if skipped else 0)
    index()
    if skipped:
        raise typer.Exit(code=1)


def _sync_snapshot(repo: Repository, specs: list[ConnectorSpec]) -> bool:
    """Materialize one snapshot repo, reporting per document.

    Returns:
        True when something needs the user's attention — a document that
        could not be fetched, or a snapshot that could not be written.
        Sync's exit code is built from these, so a scheduled run fails
        loudly rather than leaving a silently stale snapshot.
    """
    try:
        report = materialize(Path(repo.path), urls=list(repo.urls), specs=specs)
    except (ValueError, GitCommandError) as exc:
        typer.echo(f"{repo.id}: failed — {exc}", err=True)
        return True
    typer.echo(f"{repo.id}: {report.summary()}", err=bool(report.failed))
    for url, reason in report.failed:
        # Listed, not counted: which document is missing decides whether
        # this is a typo in the config or a source that is down.
        typer.echo(f"  {url}: {reason}", err=True)
    return bool(report.failed)


_KIND_LABELS = {
    LinkKind.READS_KEY: "read by",
    LinkKind.DECLARES: "declared by",
    LinkKind.REFERENCES: "mentioned in",
    LinkKind.BLAMED_BY: "wrote",
}
"""How each edge kind reads in a report. Phrased from the *named thing's*
point of view, since that is what the user asked about — so `BLAMED_BY`
reads "wrote" here, not "written by": ask `refs` about a commit and the
answer is what that commit wrote."""


def _definitions(pipeline: Pipeline, symbol: str, *, limit: int = 20) -> list[Hit]:
    """Chunks whose symbol contains `symbol`, nearest match first.

    Goes through `search` rather than a dedicated store lookup: the
    `symbol` filter is a prefilter (step 19g), so the store narrows to
    exactly the matching chunks and the ranking is what breaks ties among
    them. Adding an exact-lookup method to `VectorStore` for this would
    grow the contract for one caller.
    """
    # No `repo` scope, so no ValueError to guard against: `Pipeline.search`
    # raises only for an unknown repo id, and a dataset that was never
    # indexed is skipped silently.
    return pipeline.search(symbol, k=limit, filters=SearchFilter(symbol=symbol))


@app.command()
def refs(name: str) -> None:
    """Everything that names something: a port, a ticket, a commit, a url.

    The inverted index over the links `index` recorded. Ask it about a
    port and it answers who reads it and who publishes it; about a
    ticket, which commits mention it.

    Not "who calls this function": code-to-code edges are deferred until
    they can be shown to pay for their noise (ADR-9 measured 11% of
    resolvable call names as ambiguous), so a function name has no
    callers to list yet — only its definition.
    """
    config = _config()
    _require_config_file(config)
    with LinkStore(config.index_dir) as links:
        edges = links.by_name(name)
    if not edges:
        typer.echo(f"no links named {name!r}")
        return
    typer.echo(name)
    for kind, label in _KIND_LABELS.items():
        group = [edge for edge in edges if edge.kind is kind]
        if not group:
            continue
        typer.echo(f"  {label}:")
        for edge in group:
            suffix = f"  -> {edge.url}" if edge.url else ""
            typer.echo(f"    {edge.repo}/{edge.path}:{edge.line}{suffix}")
    if any(edge.kind is LinkKind.READS_KEY for edge in edges) and not any(
        edge.kind is LinkKind.DECLARES for edge in edges
    ):
        # The drift report, narrowed to one name. Worth saying here too:
        # someone asking about a port is exactly who needs to know.
        typer.echo("  (nothing declares it — code and configuration have drifted)")


@app.command()
def why(symbol: str) -> None:
    """Why a definition looks the way it does: the commits that wrote it.

    Definition -> blame edges -> commit messages, plus whatever those
    commits pointed at outside the repository. The reasoning behind a
    design decision usually lives in a commit message and nowhere else;
    this is the path to it.
    """
    config = _config()
    _require_config_file(config)
    pipeline = _build_pipeline()
    found = _definitions(pipeline, symbol)
    if not found:
        typer.echo(f"no definition found for {symbol!r}")
        raise typer.Exit(code=1)
    with LinkStore(config.index_dir) as links:
        for hit in found[:3]:
            meta = hit.metadata
            typer.echo(
                f"{meta['symbol']}  {meta['repo']}/{meta['path']}:"
                f"{meta['start_line']}-{meta['end_line']}"
            )
            blame = links.out_of([str(hit.native_id)], kind=LinkKind.BLAMED_BY)
            if not blame:
                typer.echo("  (no commit recorded — run `wsindex index` to build blame edges)")
                continue
            typer.echo("  written by:")
            for edge in blame:
                _echo_commit(pipeline, links, edge)
            typer.echo("")


_TRAILER = re.compile(r"^[A-Z][A-Za-z-]+:\s")
"""A git trailer — `Co-Authored-By:`, `Signed-off-by:`, `Reviewed-by:`.
Metadata about the commit, not the reasoning behind the code, and `why`
asks about the latter."""


def _reasoning(message: str) -> list[str]:
    """A commit message with its trailers cut off, blank lines dropped.

    The body is the answer `why` exists to give, so it is kept whole —
    but a wall of `Co-Authored-By` at the end is noise between the reader
    and the next commit.
    """
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    while len(lines) > 1 and _TRAILER.match(lines[-1]):
        lines.pop()
    return lines


def _echo_commit(pipeline: Pipeline, links: LinkStore, edge: "Edge") -> None:
    """One commit behind a definition: its subject, then what it points at."""
    if edge.dst_chunk_id is None:
        # Blamed to a commit an earlier run indexed, or one outside the
        # window. Knowing which commit still answers "when did this
        # change" — see `blame_links`.
        typer.echo(f"    {edge.name}  (message not indexed)")
        return
    texts = pipeline.store.chunk_text(edge.repo, ids=[edge.dst_chunk_id])
    message = texts.get(edge.dst_chunk_id)
    if message is None:
        typer.echo(f"    {edge.name}  (message not indexed)")
        return
    lines = _reasoning(message)
    typer.echo(f"    {edge.name}  {lines[0]}")
    for line in lines[1:]:
        typer.echo(f"        {line}")
    for reference in links.out_of([edge.dst_chunk_id], kind=LinkKind.REFERENCES):
        typer.echo(f"        see {reference.name} -> {reference.url}")


@app.command()
def fetch(url: str) -> None:
    """Fetch one external document through the configured connectors.

    The way to check a `[[connectors]]` entry does what you meant, and
    the seam step 29b materializes through. A pointed pull: this brings
    back the document at the url and nothing around it.
    """
    config = _config()
    _require_config_file(config)
    connector = route(url, config.connectors)
    if connector is None:
        typer.echo(
            f"no connector claims {url} — add a [[connectors]] entry whose url_pattern matches it",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        document = connector.fetch(url)
    except ConnectorError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{document.url}")
    if document.title:
        typer.echo(f"title: {document.title}")
    for key, value in sorted(document.metadata.items()):
        typer.echo(f"{key}: {value}")
    typer.echo("")
    typer.echo(document.text)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Address to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind")] = 8000,
) -> None:
    """Serve the workspace over HTTP: search, index, and an admin page.

    The same engine this CLI uses, behind a thin HTTP layer (ADR-10):
    `/search` is this command's `search`, `/index` is its `index`, and
    `/admin` is a page with the repo list and two buttons. `[server]
    interval` turns on automatic syncing; `[server] token_env` names the
    variable holding the bearer token every request must carry.

    Binds to localhost by default. A search index over private
    repositories reaching the network is a decision, not a default —
    pass `--host 0.0.0.0` to make it, preferably with a token set.
    """
    config = _config()
    _require_config_file(config)
    try:
        import uvicorn

        from wsindex.server import create_app
    except ImportError as exc:  # pragma: no cover - depends on the install
        typer.echo(
            "error: the server needs the `server` extra — "
            "`uv sync --extra server` (fastapi, uvicorn)",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    token: str | None = None
    name = config.server_token_env
    if name is not None:
        token = os.environ.get(name)
        if not token:
            # The same refusal a connector makes: starting anyway would
            # open the index to anyone who can reach the port, and the
            # config says that is not what was wanted.
            typer.echo(
                f"error: ${name} is not set, and [server] token_env names it — "
                "set it, or remove token_env to serve without authentication",
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        typer.echo(
            "warning: serving without authentication ([server] token_env is unset)", err=True
        )

    typer.echo(f"wsindex '{config.name}' on http://{host}:{port}  (admin at /admin)")
    uvicorn.run(create_app(token=token), host=host, port=port, log_level="warning")


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
