"""Commands that describe the workspace: create it, add to it, look at it."""

from typing import Annotated

import typer

from wsindex.cli.composition import config_or_default, require_config_file
from wsindex.config import Backend, Config, Provider, Repository, RepoSource
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


def status() -> None:
    """Show the workspace: name, backend, registered repos."""
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
        typer.echo(f"  {repo.id} -> {repo.path}")
