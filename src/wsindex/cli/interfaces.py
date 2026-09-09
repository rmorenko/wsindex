"""The other ways in: a shell, an agent protocol, an HTTP server.

Three adapters over the same library, each a command that hands the
pipeline to something that speaks a different language.
"""

import os
from typing import Annotated

import typer

from wsindex.cli.composition import build_pipeline, config_or_default, require_config_file


def shell() -> None:
    """Ask many questions without reloading the model each time.

    A `wsindex search` spends most of its seconds before it searches
    anything — loading the embedding model, opening the store. Here that
    is paid once. Arrow keys walk the history, Tab completes repo ids and
    flags, a number opens a hit in full, and `:open <n>` sends it to
    `$EDITOR` at the right line. `:help` for the rest.
    """
    config = config_or_default()
    require_config_file(config)
    try:
        from wsindex import shell as shell_module
    except ImportError as exc:  # pragma: no cover - depends on the install
        typer.echo(
            "error: the shell needs the `shell` extra — `uv sync --extra shell` (prompt-toolkit)",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    shell_module.run(build_pipeline(), history_dir=config.index_dir)


def mcp() -> None:
    """Serve the workspace to an agent client over MCP (stdio).

    The answer to "is there an IDE plugin": an agent client speaks MCP
    already, so pointing it at this command gives it `search`, `refs` and
    `why` over the workspace with no plugin to install. Configure it as
    the command to run; the protocol is on stdin and stdout, so nothing
    else may be printed there.

    A workspace already running `wsindex serve` can offer the same tools
    over HTTP instead — same tool code, other transport.
    """
    config = config_or_default()
    require_config_file(config)
    try:
        from wsindex.mcp_server import build
    except ImportError as exc:  # pragma: no cover - depends on the install
        typer.echo(
            "error: MCP needs the `mcp` extra — `uv sync --extra mcp`",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    # Nothing on stdout before this: the transport owns that stream, and
    # a friendly banner would be a protocol error.
    build(build_pipeline()).run(transport="stdio")


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
    config = config_or_default()
    require_config_file(config)
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
