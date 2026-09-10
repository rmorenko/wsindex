"""What the server is doing, in the one format an ops team already scrapes.

Prometheus' text exposition format, emitted by hand. No dependency: the
format is a dozen lines of specification and `prometheus-client` would be
a runtime import in every install for something the tests can check
harder than a library can. And they do — `test_metrics.py` parses this
output with the *official* parser, so "valid exposition format" is
asserted rather than believed.

**Counted here, not in `stats.py`.** The search log is a personal note,
optional, and switched off by anyone who wants it off; metrics are how an
operator knows the process is alive. One must not depend on the other. It
also means no query text ever appears here: a label carrying what people
searched for would be both unbounded cardinality and a copy of the
private log shipped to whoever scrapes.

The 500 ms bucket is not an arbitrary power of ten. It is the SLO of step
39, so `rate(wsindex_http_request_seconds_bucket{le="0.5",route="/search"}[5m])
/ rate(wsindex_http_request_seconds_count{route="/search"}[5m])` is the
compliance ratio directly, with no recording rule in between.
"""

from __future__ import annotations

import bisect
import math
import threading
import time
from collections.abc import Iterator
from typing import Any

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
"""What a scraper expects to be told this is. Version 0.0.4 is the
classic text format — the one every Prometheus reads."""

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
"""Request-duration bucket bounds, in seconds.

Chosen around what this server measurably does rather than by decade: a
warm search is 8 ms (`poe bench`), so the interesting resolution is below
100 ms, and `0.5` is there because it is the SLO."""

INDEX_BUCKETS = (0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0)
"""Indexing-duration bounds. A different scale entirely: the warm path is
0.14 s and a cold index of a real repo 8 s, so seconds and minutes are
the units that matter."""


class _Histogram:
    """One histogram series: cumulative buckets, a sum and a count."""

    def __init__(self, bounds: tuple[float, ...]) -> None:
        self.bounds = bounds
        self.counts = [0] * len(bounds)
        self.total = 0.0
        self.observations = 0

    def observe(self, value: float) -> None:
        """Record one measurement."""
        # `bisect_left`, and the difference is the whole meaning of `le`:
        # buckets are "less than or **equal**", so a value sitting exactly
        # on a bound belongs to that bound's bucket. `bisect_right` starts
        # one past it and quietly under-counts every round number — which
        # is most of them, since bounds are chosen to be round.
        for index in range(bisect.bisect_left(self.bounds, value), len(self.bounds)):
            self.counts[index] += 1
        self.total += value
        self.observations += 1

    def cumulative(self) -> Iterator[tuple[float, int]]:
        """`(le, count)` pairs, ascending, infinity last.

        The bound as a number, not as the text `+Inf`: how Prometheus
        spells infinity is `_number`'s business, and writing it here too
        would be the same fact in two places — with a branch in `_number`
        that nothing could reach.
        """
        yield from zip(self.bounds, self.counts, strict=True)
        yield math.inf, self.observations


class Metrics:
    """Everything the server counts, and how to render it.

    Deliberately knows nothing about FastAPI. The live gauges — whether a
    run is in progress, how many repos there are — arrive as arguments to
    `render`, so this stays a value object a test can drive directly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = {}
        self._latency: dict[tuple[str, str], _Histogram] = {}
        self._index_runs: dict[str, int] = {}
        self._index_seconds = _Histogram(INDEX_BUCKETS)
        self._written = 0
        self._deleted = 0
        self._last_index_success = 0.0

    def observed(self, *, route: str, method: str, status: int, seconds: float) -> None:
        """Record one finished HTTP request.

        Args:
            route: The route *template* (`/search`), never the raw path —
                a label built from user input is an unbounded label.
            method: HTTP method.
            status: Response status code.
            seconds: Wall clock, first byte of request to last of response.
        """
        with self._lock:
            key = (route, method, str(status))
            self._requests[key] = self._requests.get(key, 0) + 1
            self._latency.setdefault((route, method), _Histogram(LATENCY_BUCKETS)).observe(seconds)

    def indexed(
        self, outcome: str, *, seconds: float = 0.0, written: int = 0, deleted: int = 0
    ) -> None:
        """Record one indexing run, however it ended.

        Args:
            outcome: `ok`, `busy` or `error`. `busy` is a refusal, not a
                failure — the one-writer rule working — and separating
                the two is the whole reason this is a label.
            seconds: How long it took; only meaningful for `ok`.
            written: Chunks written.
            deleted: Chunks deleted.
        """
        with self._lock:
            self._index_runs[outcome] = self._index_runs.get(outcome, 0) + 1
            if outcome != "ok":
                return
            self._index_seconds.observe(seconds)
            self._written += written
            self._deleted += deleted
            self._last_index_success = time.time()

    def render(self, *, version: str, indexing: bool, repos: int) -> str:
        """The whole exposition, as one body.

        Args:
            version: This server's version, for `build_info`.
            indexing: Whether a run holds the write lock right now.
            repos: How many repositories are configured.

        Returns:
            Text in Prometheus exposition format 0.0.4.
        """
        lines: list[str] = []
        with self._lock:
            _info(lines, version)
            _counter(
                lines,
                "wsindex_http_requests_total",
                "HTTP requests served, by route, method and status.",
                {
                    _labels(route=route, method=method, status=status): count
                    for (route, method, status), count in sorted(self._requests.items())
                },
            )
            _histograms(
                lines,
                "wsindex_http_request_seconds",
                "Time to serve an HTTP request, by route and method.",
                {
                    _labels(route=route, method=method): histogram
                    for (route, method), histogram in sorted(self._latency.items())
                },
            )
            _counter(
                lines,
                "wsindex_index_runs_total",
                "Indexing runs, by outcome; `busy` means the one-writer rule refused one.",
                {
                    _labels(outcome=outcome): count
                    for outcome, count in sorted(self._index_runs.items())
                },
            )
            _histograms(
                lines,
                "wsindex_index_seconds",
                "Time taken by an indexing run that completed.",
                {_labels(): self._index_seconds},
            )
            _counter(
                lines,
                "wsindex_chunks_written_total",
                "Chunks written to the vector store by indexing runs.",
                {_labels(): self._written},
            )
            _counter(
                lines,
                "wsindex_chunks_deleted_total",
                "Chunks deleted from the vector store by indexing runs.",
                {_labels(): self._deleted},
            )
            _gauge(
                lines,
                "wsindex_last_index_success_timestamp_seconds",
                "When the last successful indexing run finished; 0 if none has.",
                {_labels(): self._last_index_success},
            )
        _gauge(
            lines,
            "wsindex_indexing",
            "1 while an indexing run holds the write lock.",
            {_labels(): float(indexing)},
        )
        _gauge(
            lines,
            "wsindex_repos",
            "Repositories this server is configured to index.",
            {_labels(): float(repos)},
        )
        return "".join(lines)


def route_of(scope: dict[str, Any]) -> str:
    """The route template a request matched, or `other`.

    `other` rather than the raw path, and that is the point: an unmatched
    request is a 404 whose path came from the caller, and putting it in a
    label lets anyone with a URL bar grow this server's memory without
    bound.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if isinstance(path, str) else "other"


class Meter:
    """Time every request, including what is mounted.

    Pure ASGI, like `Guard`, and for the same reason: `BaseHTTPMiddleware`
    buffers responses and runs them in a task group, which is a lot of
    machinery to pay for a stopwatch. This one wraps `send` to catch the
    status and otherwise gets out of the way.

    A middleware rather than a line in each handler because the handlers
    are not the only things serving requests — `/mcp` is a mounted
    application with routes of its own, and a guard or a meter that has
    to be *listed* per endpoint is one mount away from being wrong. That
    exact bug is recorded on `Guard`.
    """

    def __init__(self, app: Any, metrics: Metrics) -> None:
        self._app = app
        self._metrics = metrics

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Pass the request through, recording what it cost."""
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        started = time.perf_counter()
        # 500 unless the app says otherwise: an exception that escapes
        # never sends a start message, and the request did fail.
        seen = 500

        async def watched(message: Any) -> None:
            nonlocal seen
            if message.get("type") == "http.response.start":
                seen = int(message["status"])
            await send(message)

        try:
            await self._app(scope, receive, watched)
        finally:
            self._metrics.observed(
                # Read after the call: routing is what sets `scope["route"]`.
                route=route_of(scope),
                method=str(scope.get("method", "")),
                status=seen,
                seconds=time.perf_counter() - started,
            )


# --- the exposition format, which is small enough to write out -----------


def _info(lines: list[str], version: str) -> None:
    """`build_info`: the version as a label on a constant 1, as is done."""
    _gauge(
        lines,
        "wsindex_build_info",
        "Always 1; the version is the label.",
        {_labels(version=version): 1.0},
    )


def _counter(lines: list[str], name: str, help_text: str, series: dict[str, int]) -> None:
    _family(lines, name, help_text, "counter", series)


def _gauge(lines: list[str], name: str, help_text: str, series: dict[str, float]) -> None:
    _family(lines, name, help_text, "gauge", series)


def _family(lines: list[str], name: str, help_text: str, kind: str, series: dict[str, Any]) -> None:
    """One metric family: HELP, TYPE, then its series.

    An empty family still gets its HELP and TYPE. A scraper that sees a
    counter go from absent to present cannot tell a restart from a first
    increment, so a zero-sample family is worth the two lines.
    """
    lines.append(f"# HELP {name} {_escape_help(help_text)}\n")
    lines.append(f"# TYPE {name} {kind}\n")
    for labels, value in series.items():
        lines.append(f"{name}{labels} {_number(value)}\n")


def _histograms(lines: list[str], name: str, help_text: str, series: dict[str, _Histogram]) -> None:
    """A histogram family: cumulative buckets, then `_sum` and `_count`.

    `le` joins whatever labels the series already has, and it goes last
    because that is how every other exporter writes it and diffs get read
    by people.
    """
    lines.append(f"# HELP {name} {_escape_help(help_text)}\n")
    lines.append(f"# TYPE {name} histogram\n")
    for labels, histogram in series.items():
        for le, count in histogram.cumulative():
            lines.append(f"{name}_bucket{_with(labels, le=_number(le))} {count}\n")
        lines.append(f"{name}_sum{labels} {_number(histogram.total)}\n")
        lines.append(f"{name}_count{labels} {histogram.observations}\n")


def _labels(**pairs: str) -> str:
    """`{a="1",b="2"}`, or an empty string when there are no labels."""
    if not pairs:
        return ""
    inside = ",".join(f'{key}="{_escape_label(value)}"' for key, value in pairs.items())
    return "{" + inside + "}"


def _with(labels: str, **extra: str) -> str:
    """`labels` plus more, whether or not it had any to begin with."""
    added = _labels(**extra)
    if not labels:
        return added
    return labels[:-1] + "," + added[1:]


def _escape_label(value: str) -> str:
    """Backslash, double quote and newline, as the format requires."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_help(text: str) -> str:
    """HELP escapes backslash and newline only — a quote is literal there."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _number(value: float | int) -> str:
    """A value as Prometheus writes it: `+Inf` rather than `inf`."""
    if isinstance(value, int):
        return str(value)
    if value == float("inf"):
        return "+Inf"
    # `repr` keeps 0.005 as `0.005` instead of `0.005000000000000001`,
    # and integral floats stay readable as `1.0`.
    return repr(float(value))


__all__ = [
    "CONTENT_TYPE",
    "INDEX_BUCKETS",
    "LATENCY_BUCKETS",
    "Meter",
    "Metrics",
    "route_of",
]
