"""Commands that fill and tidy the index: index, sync, compact."""

from datetime import timedelta
from pathlib import Path
from typing import Annotated

import typer

from wsindex.cli.composition import (
    build_pipeline,
    build_store,
    config_or_default,
    require_config_file,
)
from wsindex.config import Repository
from wsindex.connectors import ConnectorSpec
from wsindex.ingest import GitCommandError, NotAGitRepositoryError, sync_repo
from wsindex.snapshot import materialize
from wsindex.ui import progress as ui_progress


def index() -> None:
    """Chunk and embed what changed in every configured repo.

    Incremental against git: a repo that has been indexed before and has
    a clean working tree only pays for the files that changed since. A
    dirty tree costs a full pass, because a commit-to-commit diff cannot
    see uncommitted work.
    """
    pipeline = build_pipeline()
    try:
        with ui_progress("indexing") as report_repo:
            report = pipeline.index(progress=report_repo)
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
            # refs`/`why` territory.
            first = drift[0]
            typer.echo(
                f"drift: {len(drift)} unresolved config reference(s), "
                f"first at {first.repo}/{first.path}:{first.line} -> {first.name}",
                err=True,
            )


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
    config = config_or_default()
    require_config_file(config)
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
    config = config_or_default()
    require_config_file(config)
    report = build_store(config).compact(older_than=timedelta(days=keep_days))
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
