"""HTTP over the same Pipeline: search, index, status.

Deliberately the smallest thing that can be called a server. Every
endpoint is a call into the library and a rendering of what it returned
— there is no indexing code here, no chunking, no store access that
`Pipeline` does not already do. ADR-10
puts it plainly: an endpoint that cannot be expressed as a library call
means the library is missing something, not that the server should grow
it.

Which is why the contract mirrors the CLI's. `GET /search?q=...&k=&repo=`
is `wsindex search --repo`; `POST /index` is `wsindex index`. Two
interfaces over one engine stay honest only while they say the same
things, and the cheapest way to keep them saying the same things is to
give them the same words.

Authentication
--------------
A bearer token, named by the config as an environment variable and never
written in it — the rule connectors keep and the S3 store keeps (ADR-7).
With `token_env` set and the variable empty the server refuses to start:
a search index over private repositories is not a thing to begin serving
by accident. With no `token_env` at all the
server is open, which is a decision a person has to write down.

State
-----
One `Pipeline`, built once at startup and shared. That is safe for reads
because `Pipeline.search` refreshes the store first (ADR-10), and it is
what makes a request cheap: building a pipeline means loading an
embedding model.

Writes are serialized by a lock. Not for correctness — concurrent writers
were measured losing nothing (ADR-10) — but because two indexing
runs of the same repo do the same work twice and neither shortens the
other's next pass. One at a time, and the second caller is told the
first is running rather than made to wait.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from wsindex.config import Config
from wsindex.ingest import NotAGitRepositoryError
from wsindex.model import Kind, SearchFilter
from wsindex.server.admin import mount_admin
from wsindex.server.scheduler import mount_scheduler

if TYPE_CHECKING:  # pragma: no cover - import-time only, for annotations
    from wsindex.pipeline import Pipeline

# FastAPI at module scope, not inside the factory. It has to be: with
# postponed annotations a route's `request: Request` is resolved against
# the *module* globals, and a name imported inside a function is not
# there — FastAPI then reads it as a missing query parameter and answers
# 422 to every call. Importing here costs a base install nothing, since
# nothing imports `wsindex.server` except `wsindex serve`, which checks
# for the extra first.


@dataclass
class RunLog:
    """The last few indexing runs, for `status` and the admin page.

    In memory and bounded: a server that kept every run would be a
    logging system, and the question this answers is "did the last sync
    work", which needs the last few.
    """

    limit: int = 20
    entries: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, kind: str, started: float, detail: dict[str, Any]) -> None:
        """Append one finished run, dropping the oldest past `limit`."""
        with self._lock:
            self.entries.insert(
                0,
                {
                    "kind": kind,
                    "at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
                    "seconds": round(time.monotonic() - started, 2) if started < 1e9 else None,
                    **detail,
                },
            )
            del self.entries[self.limit :]


class Busy(RuntimeError):
    """An indexing run was asked for while one was already going."""


@dataclass
class Writer:
    """The one-writer rule of ADR-10, as a lock that refuses to queue.

    `try` rather than `acquire`: a caller who waits learns nothing and a
    scheduler that waits piles up. Told "already running", both do the
    right thing — the scheduler skips this tick, the human refreshes.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)

    @contextmanager
    def held(self) -> Iterator[None]:
        """Hold the write lock, or raise `Busy` immediately.

        Raises:
            Busy: Another indexing run is in progress.
        """
        if not self._lock.acquire(blocking=False):
            raise Busy("an indexing run is already in progress")
        try:
            yield
        finally:
            self._lock.release()

    @property
    def busy(self) -> bool:
        """True while a run holds the lock."""
        return self._lock.locked()


def create_app(
    *,
    pipeline_factory: Callable[[], Pipeline] | None = None,
    token: str | None = None,
) -> FastAPI:
    """Build the ASGI application.

    A factory rather than a module-level `app` so that the pipeline is
    built once, explicitly, and a test can pass its own. The import of
    FastAPI is inside for the same reason every optional dependency is
    imported late here: a base install must not pay for the `server`
    extra to import `wsindex`.

    Args:
        pipeline_factory: Builds the shared Pipeline. Defaults to the
            CLI's composition root, so the server and the CLI cannot
            drift into different engines.
        token: Bearer token every request must carry, or None for an
            open server.

    Returns:
        The application, with `state.pipeline`, `state.writer`,
        `state.runs` and `state.token` attached for the routers.
    """
    if pipeline_factory is None:
        from wsindex.cli import build_pipeline

        pipeline_factory = build_pipeline

    app = FastAPI(
        title="wsindex",
        summary="Semantic search across the repositories of a workspace.",
        version="0.1.0",
    )
    app.state.pipeline = pipeline_factory()
    app.state.writer = Writer()
    app.state.runs = RunLog()
    app.state.token = token

    def authorize(request: Request) -> None:
        """Reject a request without the configured bearer token."""
        if request.app.state.token is None:
            return
        header = request.headers.get("Authorization", "")
        offered = header.removeprefix("Bearer ").strip()
        if offered != request.app.state.token:
            # 401 with no hint about which half was wrong: a server that
            # says "unknown token" to one caller and "no token" to
            # another has told both something.
            raise HTTPException(status_code=401, detail="unauthorized")

    guarded = [Depends(authorize)]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Liveness, unauthenticated: a probe is not a reader.

        Deliberately says nothing about the workspace — a load balancer
        needs to know the process is up, not what it indexes.
        """
        return {"status": "ok"}

    @app.get("/search", dependencies=guarded)
    def search(
        q: Annotated[str, Query(description="Natural-language query")],
        k: Annotated[int, Query(ge=1, le=100, description="How many hits")] = 10,
        repo: Annotated[str | None, Query(description="Restrict to one repo id")] = None,
        lang: Annotated[list[str] | None, Query(description="Language (repeat for OR)")] = None,
        kind: Annotated[list[Kind] | None, Query(description="Kind (repeat for OR)")] = None,
        path: Annotated[str | None, Query(description="Path glob")] = None,
        symbol: Annotated[str | None, Query(description="Substring of the symbol")] = None,
    ) -> dict[str, Any]:
        """Search the workspace. The CLI's `search`, with its flags as query params."""
        candidate = SearchFilter(
            lang=tuple(lang or ()),
            kind=tuple(kind or ()),
            path=path,
            symbol=symbol,
        )
        try:
            hits = app.state.pipeline.search(
                q, k=k, repo=repo, filters=None if candidate.is_empty else candidate
            )
        except ValueError as exc:
            # An unknown repo id is the caller's mistake, not the
            # server's — the CLI exits 1 on it, and 400 is the same
            # sentence in HTTP.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"query": q, "count": len(hits), "hits": [hit.to_json() for hit in hits]}

    @app.post("/index", dependencies=guarded)
    def index() -> dict[str, Any]:
        """Re-index every configured repo. Incremental, exactly as the CLI is."""
        started = time.time()
        clock = time.monotonic()
        try:
            with app.state.writer.held():
                report = app.state.pipeline.index()
        except Busy as exc:
            # 409, not 429: nothing is rate-limiting the caller, the
            # resource is in a state that forbids the request.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NotAGitRepositoryError as exc:
            # A repo in the config that is not a checkout. The CLI prints
            # this and exits 1; letting it out as a 500 would say the
            # server broke, when the answer is in the config file.
            app.state.runs.record("index", started, {"error": str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        detail = {
            "files": report.files,
            "chunks": report.chunks,
            "written": report.written,
            "deleted": report.deleted,
            "commits": report.commits,
            "full_repos": list(report.full_repos),
            "missing_repos": list(report.missing_repos),
            "seconds": round(time.monotonic() - clock, 2),
        }
        app.state.runs.record("index", started, detail)
        return detail

    @app.get("/status", dependencies=guarded)
    def status() -> dict[str, Any]:
        """What this server is serving: workspace, repos, recent runs."""
        config = Config()
        return {
            "workspace": config.name,
            "backend": config.backend.value,
            "store": config.store_uri,
            "rank": config.rank_enabled,
            "indexing": app.state.writer.busy,
            "repos": [
                {
                    "id": repo.id,
                    "path": repo.path,
                    "remote": repo.remote,
                    "source": repo.source.value if repo.source else None,
                    "documents": len(repo.urls),
                }
                for repo in config.repos
            ],
            "runs": list(app.state.runs.entries),
        }

    mount_scheduler(app, guarded)
    mount_admin(app, guarded)
    _mount_mcp(app)
    return app


def _mount_mcp(app: FastAPI) -> None:
    """Offer the same tools over streamable HTTP, when the extra is here.

    The plan's own words for this step: two transports, one set of tool
    code. `wsindex mcp` runs the server over stdio for an editor that
    spawns it; a workspace that already has this one running gets `/mcp`
    for free, and neither transport has tool code of its own.

    Silently skipped without the `mcp` extra — a server missing an
    optional adapter should serve everything else, not refuse to start.
    """
    try:
        from wsindex.mcp_server import build
    except ImportError:  # pragma: no cover - depends on the install
        return
    tools = build(app.state.pipeline)
    # The sub-app routes `/mcp` of its own, so mounting it at `/mcp`
    # without this puts the endpoint at `/mcp/mcp` — and a client asking
    # the documented address gets "Session terminated", which reads as a
    # protocol fault rather than a 404. Measured against a real client.
    tools.settings.streamable_http_path = "/"
    # Mounted rather than re-routed: the SDK owns the session handling,
    # the event stream and the protocol version negotiation, and
    # re-implementing any of that here would be a second protocol.
    app.mount("/mcp", tools.streamable_http_app())
    app.router.lifespan_context = _with_session_manager(
        app.router.lifespan_context, tools.session_manager.run
    )


def _with_session_manager(outer: Any, inner: Any) -> Any:
    """Run the MCP session manager for the app's lifetime, inside `outer`.

    The streamable-HTTP transport keeps state per session and needs its
    task group running — mounting the app without this gives a 500 on
    the first call rather than at startup, which is the kind of failure
    that reaches production.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(scope: FastAPI) -> Any:
        async with inner(), outer(scope):
            yield

    return lifespan


__all__ = ["Busy", "RunLog", "Writer", "create_app"]
