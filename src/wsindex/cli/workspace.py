"""Commands that describe the workspace: create it, add to it, look at it."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from wsindex.cli.composition import build_pipeline, config_or_default, require_config_file
from wsindex.config import Backend, Config, Provider, Repository, RepoSource
from wsindex.ingest import SKIP_REASONS
from wsindex.ingest.git_state import STATE_FILE, IndexState
from wsindex.paths import user_config_file, workspace_config_path


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
    config = config_or_default()
    location = require_config_file(config)
    try:
        config.add_repo(
            Repository(
                id=repo_id,
                path=path,
                remote=remote,
                source=source,
                urls=tuple(url or ()),
                ignore=tuple(ignore or ()),
            )
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
    if remote is None and source is None and not Path(path).expanduser().is_dir():
        # Said now, while the typo is still on screen. It used to surface
        # at the next `index` as `missing repos:`, possibly days later
        # and from a different command. A warning rather than a refusal:
        # with a remote the path is *supposed* not to exist yet, which is
        # why the two flags above exclude themselves from the check.
        typer.echo(
            f"warning: {path} does not exist — index will skip it; "
            "pass --remote to have sync clone it there",
            err=True,
        )


def status() -> None:
    """Show the workspace: name, backend, registered repos, index state."""
    config = config_or_default()
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
        typer.echo(f"  {repo.id} -> {repo.path}{_indexed_at(config, repo.id)}")


def _indexed_at(config: Config, repo_id: str) -> str:
    """What the index holds for one repo, as a suffix for the status line.

    `status` used to recite the config back — every line of it already
    visible in `wsindex.toml`. Nothing said whether the thing the tool
    exists for had ever run. Both facts are already on disk: the commit
    in `state.json`, the file's mtime for when.

    Never raises. Status is the command people run *because* something
    is wrong, so it must survive an index that is missing or damaged and
    say which.
    """
    state_file = config.index_dir / STATE_FILE
    try:
        state = IndexState.load(config.index_dir)
        commit = state.commits.get(repo_id)
        if commit is None:
            return "  (not indexed)"
        when = datetime.fromtimestamp(state_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"  (indexed {when} at {commit[:7]})"
    except OSError:
        return "  (index state unreadable)"


def explain(
    path: Annotated[str, typer.Argument(help="File to ask about, as you would type it")],
) -> None:
    """Say whether a file is indexed, and if not, which rule left it out.

    The question this tool gets asked most: "why does search not find my
    file?" Six rules can exclude one, and until this existed every one of
    them looked identical from outside — the file was simply absent. Each
    answer points at a different fix, so guessing between them is the
    difference between editing `formats`, moving the file, and looking at
    permissions.

    The path may be absolute or relative to the current directory; the
    repo it belongs to is worked out from the config.
    """
    require_config_file(config_or_default())
    report = build_pipeline().describe(Path(path))
    if report is None:
        typer.echo(
            f"error: {Path(path).expanduser().resolve()} is not inside any configured repo — "
            "`wsindex status` lists them",
            err=True,
        )
        raise typer.Exit(code=1)
    where = f"{report.repo}/{report.rel_path}"
    if report.skipped is not None:
        typer.echo(f"{where}: not indexed — {SKIP_REASONS[report.skipped]}")
        return
    typer.echo(f"{where}: indexed as {report.lang} ({report.kind.value if report.kind else ''})")
    if report.parsed_cleanly:
        typer.echo(f"  {report.chunks} chunk(s), {report.symbols} with a symbol")
    else:
        typer.echo(
            f"  {report.chunks} chunk(s), but the {report.lang} grammar reported errors — "
            "the parts it could not read are indexed as text, not definitions"
        )
