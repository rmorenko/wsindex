"""The other ways in: a shell, an agent protocol, an HTTP server.

Three adapters over the same library, each a command that hands the
pipeline to something that speaks a different language.
"""

import ipaddress
import os
from typing import Annotated

import typer

from wsindex.cli.composition import (
    build_pipeline,
    config_or_default,
    require_config_file,
    require_extra,
)


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
    except ImportError:  # pragma: no cover - depends on the install
        require_extra("shell", "prompt-toolkit")
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
    except ImportError:  # pragma: no cover - depends on the install
        require_extra("mcp", "the official MCP SDK")
    # Nothing on stdout before this: the transport owns that stream, and
    # a friendly banner would be a protocol error.
    build(build_pipeline()).run(transport="stdio")


def is_loopback(host: str) -> bool:
    """True when only this machine can reach `host`.

    The question `serve` asks before it agrees to run without a token.
    `localhost` by name as well as by address, because that is what
    people type; anything it cannot parse — a hostname, `::`, `0.0.0.0` —
    is not loopback, which is the safe way to be wrong.
    """
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(
    host: Annotated[str, typer.Option("--host", help="Address to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind")] = 8000,
    insecure: Annotated[
        bool,
        typer.Option("--insecure", help="Allow a public address with no token"),
    ] = False,
) -> None:
    """Serve the workspace over HTTP: search, index, and an admin page.

    The same engine this CLI uses, behind a thin HTTP layer (ADR-10):
    `/search` is this command's `search`, `/index` is its `index`, and
    `/admin` is a page with the repo list and two buttons. `[server]
    interval` turns on automatic syncing; `[server] token_env` names the
    variable holding the bearer token every request must carry.

    Binds to localhost by default. A search index over private
    repositories reaching the network is a decision, not a default —
    pass `--host 0.0.0.0` to make it, and set a token. Without one this
    refuses to start on a public address; `--insecure` says you meant it.
    """
    config = config_or_default()
    require_config_file(config)
    try:
        import uvicorn

        from wsindex.server import create_app
    except ImportError:  # pragma: no cover - depends on the install
        require_extra("server", "fastapi, uvicorn")

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
    elif not is_loopback(host) and not insecure:
        # The symmetric half of the refusal above. A token named but
        # unset stops the server; an address the whole network can reach
        # with no token at all used to be a line of stderr somebody
        # scrolls past. Both are the config saying one thing and the
        # process doing another.
        typer.echo(
            f"error: {host} is reachable from the network and no token is set — "
            "set [server] token_env, bind to 127.0.0.1, or pass --insecure",
            err=True,
        )
        raise typer.Exit(code=1)
    else:
        typer.echo(
            "warning: serving without authentication ([server] token_env is unset)", err=True
        )

    typer.echo(f"wsindex '{config.name}' on http://{host}:{port}  (admin at /admin)")
    uvicorn.run(create_app(token=token), host=host, port=port, log_level="warning")
