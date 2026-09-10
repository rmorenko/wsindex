"""Keeping the index current without anybody asking: a tick, and a hook.

Two ways for work to start, because two things cause it: time passes, or
somebody pushes.

The tick does what `wsindex sync` does, every `[server] interval`
seconds. Off unless configured — a server that starts doing network work
on its own spends somebody's rate limit.

The hook, `POST /hooks/sync`, runs the same thing now. Its body is
ignored on purpose: parsing a provider's payload would claim support for
that provider's every event shape, and "something changed" is all an
incremental run needs.

Both decline rather than queue when a run is in progress (ADR-10), and
both run in threads: the work is blocking git and CPU-bound Python, and
a coroutine would stop the server answering searches.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from wsindex.ingest import GitCommandError, NotAGitRepositoryError, sync_repo
from wsindex.snapshot import materialize

log = logging.getLogger(__name__)


def sync_and_index(app: FastAPI) -> dict[str, Any]:
    """One full cycle: pull, materialize, index. What `wsindex sync` does.

    The CLI's `sync` command in library terms rather than in `typer`
    terms. Failures are per repo and reported, never raised: one
    unreachable remote must not stop the repos after it, and a server
    that died on a network blip would need a human to notice.

    Args:
        app: The application, for its pipeline, lock and run log.

    Returns:
        What happened, in the shape `POST /index` returns plus a `synced`
        map of repo id -> outcome.

    Raises:
        Busy: A run is already in progress.
        NotAGitRepositoryError: A configured path is not a checkout. The
            one failure that is not per repo: indexing reads them all in
            one pass, so it stops at the first. Recorded before it is
            re-raised, or a scheduled run would fail invisibly.
    """
    from wsindex.server.api import Busy  # noqa: F401  (documented in Raises)

    started = time.time()
    clock = time.monotonic()
    synced: dict[str, str] = {}
    with app.state.writer.held():
        config = app.state.config
        for repo in config.repos:
            if repo.is_snapshot:
                try:
                    report = materialize(
                        Path(repo.path), urls=list(repo.urls), specs=config.connectors
                    )
                    synced[repo.id] = report.summary()
                except (ValueError, GitCommandError) as exc:
                    log.warning("snapshot %s failed: %s", repo.id, exc)
                    synced[repo.id] = f"failed — {exc}"
            elif repo.remote is not None:
                try:
                    synced[repo.id] = sync_repo(Path(repo.path), remote=repo.remote).value
                except (NotAGitRepositoryError, GitCommandError) as exc:
                    log.warning("sync %s failed: %s", repo.id, exc)
                    synced[repo.id] = f"failed — {exc}"
        try:
            report = app.state.pipeline.index()
        except NotAGitRepositoryError as exc:
            app.state.runs.record("sync", started, {"synced": synced, "error": str(exc)})
            raise
    from wsindex.server.api import run_detail

    detail = {**run_detail(report, seconds=round(time.monotonic() - clock, 2)), "synced": synced}
    app.state.runs.record("sync", started, detail)
    return detail


class Ticker:
    """Runs `sync_and_index` on an interval, in a daemon thread.

    Stoppable by an `Event` rather than by a flag the loop checks: a
    server shutting down should not wait out the rest of an interval,
    and `Event.wait` returns the moment it is set.
    """

    def __init__(self, app: FastAPI, *, interval: float) -> None:
        """Prepare a ticker; `start` is separate so a test can skip it.

        Args:
            app: The application to run cycles against.
            interval: Seconds between cycles.
        """
        self.app = app
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin ticking, after one full interval."""
        self._thread = threading.Thread(target=self._loop, name="wsindex-ticker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the loop to end and wait briefly for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        # Waits first, so starting the server is not also a sync: the
        # operator who just started it is watching, and can press the
        # button themselves.
        while not self._stop.wait(self.interval):
            try:
                sync_and_index(self.app)
            except Exception as exc:
                # Deliberately broad and deliberately not fatal. This
                # thread is the only thing keeping the index current; if
                # it dies on one bad tick, the server keeps answering
                # from a corpus that quietly stops advancing.
                #
                # Logged as well as recorded, and with the traceback: the
                # run log holds twenty entries in memory and loses them
                # all on restart, which is the wrong place for the one
                # failure nobody was watching happen.
                log.exception("scheduled sync failed")
                self.app.state.runs.record(
                    "sync", time.time(), {"error": f"{type(exc).__name__}: {exc}"}
                )


def mount_scheduler(app: FastAPI, guarded: list[Any]) -> None:
    """Attach the hook endpoint and, when configured, the ticker.

    Args:
        app: Application to mount on; gains `state.ticker`.
        guarded: The auth dependency every other endpoint uses, passed in
            rather than rebuilt — two definitions of "who may call this"
            is how one of them ends up wrong.
    """
    from wsindex.server.api import Busy

    @app.post("/hooks/sync", dependencies=guarded)
    def hook() -> dict[str, Any]:
        """Sync and re-index now. The body is ignored — see the module docstring."""
        try:
            return sync_and_index(app)
        except Busy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NotAGitRepositoryError as exc:
            # The config names something that is not a checkout: the
            # caller's to fix, not a server fault.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    interval = app.state.config.server_interval
    app.state.ticker = Ticker(app, interval=interval) if interval else None
    if app.state.ticker is not None:
        app.state.ticker.start()


__all__ = ["Ticker", "mount_scheduler", "sync_and_index"]
