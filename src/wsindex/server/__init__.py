"""The server: HTTP, a scheduler and an admin page over the same Pipeline.

Everything here is behind the `server` extra, so importing `wsindex` on a
base install never touches FastAPI. `create_app` is the only entry point
worth knowing; see ADR-10 for what may and may not live in this package.
"""

from wsindex.server.api import create_app

__all__ = ["create_app"]
