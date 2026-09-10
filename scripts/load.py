"""Load against the HTTP server, judged against an SLO fixed beforehand.

The discipline is the acceptance harness's, not the benchmark's.
`bench.py` asks "did this get slower than last time" and needs a
baseline; this asks "is it fast enough", which needs a number written
down **before** the first run, or the number becomes whatever was
measured. The four below were fixed before this file could run, and the
one that came from outside — 500 ms — is the plan's own expectation for
step 39.

Three scenarios, in the order the plan named them:

`search`
    The everyday case: `WORKERS` clients asking real questions of an idle
    server. The floor everything else is compared against.

`search-during-index`
    The question actually worth asking, and the reason this file is not
    just `ab`. A full re-index writes thousands of chunks through the
    same process that is answering searches, under one GIL. Every search
    here overlaps a run by construction: the workers keep going until the
    indexing thread says it has finished, so the sample is the
    contention, not the average of contention and quiet.

`index-contention`
    ADR-10's one-writer rule under real concurrency rather than under a
    unit test: `WORKERS` simultaneous `POST /index` must produce exactly
    one run and `WORKERS - 1` refusals.

The server is a real `wsindex serve` subprocess with a real config, a
real token and its access log on — that last one deliberately, because
it is what shipping does, and a benchmark that quietly turns off the
product's defaults measures something nobody runs.

Usage:
    uv run poe load                  # run and judge
    uv run poe load -- --workers 16  # a different concurrency
    uv run poe load -- --json out.json

Environment:
    WSINDEX_BENCH_DIR  corpus clone cache, shared with `bench.py`
    WSINDEX_LOAD_FAST  set to "1" for the fake embedder — for checking
                       the harness itself, never for a real verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from acceptance import CRITERIA
from bench import bench_corpus, busy, environment

# --- the SLO, fixed before the first run ----------------------------------

P95_BUDGET_MS = 500.0
"""p95 search latency, with re-ranking off, in **both** search scenarios.

The plan's expectation for step 39, taken as written. The same budget
applies while an index run is going, and that is a choice rather than an
oversight: a person waiting on a search does not care that the server is
busy writing, so a separate, kinder budget for the busy case would be a
budget for the server's comfort instead of theirs.

For scale, `poe bench` measures the same search in-process at 7.7 ms p50
and 10.4 ms p95, so this leaves roughly fifty times over for HTTP,
JSON, the GIL and the write."""

P99_BUDGET_MS = 1000.0
"""p99, because a p95 alone hides a stall.

Twice the p95 budget: a tail that is merely twice the body is a queue
doing its job, and one that is ten times it is something blocking."""

FAILURES_ALLOWED = 0
"""Refused, dropped or timed-out searches. Correctness is not a
percentile — a server that answers 99% of searches and 500s the rest has
not met a latency target, it has hidden a bug under one."""

TIMEOUT_S = 30.0
"""When a request stops being slow and starts being a failure. Well past
any budget here, so a timeout means something is wedged rather than
loaded."""

# --- how much load ---------------------------------------------------------

WORKERS = 8
"""Concurrent clients. A team plus its agents, which is the population
this serves; it is not trying to be the internet. `--workers` sweeps it
when the question is the shape of the curve rather than the verdict."""

SEARCHES_PER_WORKER = 25
"""Requests each worker makes in the idle scenario: 200 in total, the
same sample size `poe bench` uses for its percentiles."""

STARTUP_TIMEOUT_S = 120.0
"""How long to wait for `/healthz`. Generous: the first start downloads
the embedding model if the cache is cold."""


@dataclass(frozen=True)
class Sample:
    """One request, as the client saw it."""

    ms: float
    status: int
    hits: int = 0
    error: str | None = None

    @property
    def failed(self) -> bool:
        """Whether this counts against `FAILURES_ALLOWED`.

        A 409 from `POST /index` is not a failure — it is the one-writer
        rule answering — so only transport errors and 5xx are.
        """
        return self.error is not None or self.status >= 500


@dataclass
class Result:
    """What one scenario measured."""

    scenario: str
    samples: list[Sample] = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""

    @property
    def latencies(self) -> list[float]:
        return sorted(sample.ms for sample in self.samples)

    @property
    def failures(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.failed]

    @property
    def statuses(self) -> dict[str, int]:
        return {str(code): n for code, n in sorted(Counter(s.status for s in self.samples).items())}

    def percentile(self, fraction: float) -> float:
        """Nearest-rank percentile in milliseconds; 0.0 with no samples."""
        values = self.latencies
        if not values:
            return 0.0
        return round(values[min(int(len(values) * fraction), len(values) - 1)], 1)

    def summary(self) -> dict[str, Any]:
        """The row a report prints and a JSON file keeps."""
        latencies = self.latencies
        return {
            "scenario": self.scenario,
            "requests": len(self.samples),
            "seconds": round(self.seconds, 2),
            "rps": round(len(self.samples) / self.seconds, 1) if self.seconds else 0.0,
            "p50_ms": self.percentile(0.50),
            "p95_ms": self.percentile(0.95),
            "p99_ms": self.percentile(0.99),
            "max_ms": round(latencies[-1], 1) if latencies else 0.0,
            "mean_hits": round(statistics.mean([s.hits for s in self.samples]), 1)
            if self.samples
            else 0.0,
            "failures": len(self.failures),
            "statuses": self.statuses,
            "note": self.note,
        }


# --- the client ------------------------------------------------------------


class Client:
    """One worker's connection to the server, with its own pool."""

    def __init__(self, base: str, token: str) -> None:
        self._http = httpx.Client(
            base_url=base,
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT_S,
        )

    def close(self) -> None:
        self._http.close()

    def timed(self, method: str, path: str, **kwargs: Any) -> Sample:
        """One request, measured; a transport failure is a sample too.

        A failure that raises out of a worker would end that worker and
        quietly shrink the load — the run would then report a smaller,
        faster population than it applied.
        """
        at = time.perf_counter()
        try:
            answer = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            return Sample(
                ms=(time.perf_counter() - at) * 1000,
                status=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        ms = (time.perf_counter() - at) * 1000
        hits = 0
        if answer.status_code == 200 and path == "/search":
            hits = int(answer.json().get("count", 0))
        return Sample(ms=ms, status=answer.status_code, hits=hits)


QUERIES = [query for query, _ in CRITERIA]
"""The acceptance criteria's questions. Real ones somebody wrote for this
corpus, rotated so no worker measures the same query over and over."""


def fanned(work: Callable[[int], list[Sample]], workers: int) -> tuple[list[Sample], float]:
    """Run `work(index)` in `workers` threads; collect every sample.

    Threads rather than processes or asyncio: the thing under test is one
    server process, and the client only has to be able to wait on sockets
    at the same time.
    """
    collected: list[list[Sample]] = [[] for _ in range(workers)]

    def run(index: int) -> None:
        collected[index] = work(index)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(workers)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [sample for batch in collected for sample in batch], time.perf_counter() - started


# --- the scenarios ---------------------------------------------------------


def scenario_search(server: Server, workers: int) -> Result:
    """`WORKERS` clients searching an idle server."""

    def work(index: int) -> list[Sample]:
        client = server.client()
        try:
            return [
                client.timed(
                    "GET", "/search", params={"q": QUERIES[(index + n) % len(QUERIES)], "k": 5}
                )
                for n in range(SEARCHES_PER_WORKER)
            ]
        finally:
            client.close()

    samples, seconds = fanned(work, workers)
    return Result(scenario="search", samples=samples, seconds=seconds)


def scenario_search_during_sync(server: Server, workers: int) -> Result:
    """The everyday case the plan named: search while an ordinary sync runs.

    One file changed and committed, then the incremental pass that a
    ticking server does every `[server] interval`. This is what "search
    during sync" means on a running server nearly all of the time, and it
    is a different measurement from the one below — not a milder version
    of it.
    """
    server.commit_one_change()
    return _while_indexing(server, workers, "search-during-sync", "an incremental sync")


def scenario_search_during_index(server: Server, workers: int) -> Result:
    """The same load against a *full* re-index: the maintenance case.

    Rarer — a first index, or an index directory that lost its state file
    — and much heavier. Kept as its own scenario rather than folded into
    the one above because averaging the two would hide both.

    The state file is removed first: a warm pass takes 0.14 s, which no
    amount of load can overlap meaningfully.
    """
    server.force_full_reindex()
    return _while_indexing(server, workers, "search-during-index", "a full re-index")


def _while_indexing(server: Server, workers: int, scenario: str, what: str) -> Result:
    """Search until the indexing run finishes; report what that cost.

    Overlap by construction rather than by hope: the searchers run until
    the indexing thread sets `done`, so there is no quiet tail averaged
    into the percentiles.
    """
    done = threading.Event()
    indexing: list[Sample] = []

    def index() -> None:
        client = server.client()
        try:
            indexing.append(client.timed("POST", "/index"))
        finally:
            client.close()
            done.set()

    runner = threading.Thread(target=index)

    def work(index_of: int) -> list[Sample]:
        client = server.client()
        samples = []
        try:
            n = 0
            while not done.is_set():
                samples.append(
                    client.timed(
                        "GET",
                        "/search",
                        params={"q": QUERIES[(index_of + n) % len(QUERIES)], "k": 5},
                    )
                )
                n += 1
            return samples
        finally:
            client.close()

    runner.start()
    samples, seconds = fanned(work, workers)
    runner.join()
    missed = Sample(ms=0.0, status=0, error="the index run never reported")
    run = indexing[0] if indexing else missed
    return Result(
        scenario=scenario,
        samples=samples,
        seconds=seconds,
        note=f"{what} took {run.ms / 1000:.1f}s (status {run.status})",
    )


def scenario_index_contention(server: Server, workers: int) -> Result:
    """`WORKERS` simultaneous `POST /index`: one run, the rest refused.

    Against a *full* pass again, and that is what makes the assertion
    mean anything. Fired at a warm 0.14 s run, the callers would
    serialise on their own and several 200s would prove nothing about the
    lock.
    """
    server.force_full_reindex()

    def work(_: int) -> list[Sample]:
        client = server.client()
        try:
            return [client.timed("POST", "/index")]
        finally:
            client.close()

    samples, seconds = fanned(work, workers)
    accepted = sum(1 for sample in samples if sample.status == 200)
    refused = sum(1 for sample in samples if sample.status == 409)
    return Result(
        scenario="index-contention",
        samples=samples,
        seconds=seconds,
        note=f"{accepted} run, {refused} refused with 409",
    )


SCENARIOS: dict[str, Callable[[Server, int], Result]] = {
    "search": scenario_search,
    "search-during-sync": scenario_search_during_sync,
    "search-during-index": scenario_search_during_index,
    "index-contention": scenario_index_contention,
}


# --- the server under test -------------------------------------------------

TOKEN_ENV = "WSINDEX_LOAD_TOKEN"
"""Names the variable holding the token. The config names the variable;
the variable holds the secret. Never the other way round, and this file
is not an exception to that because a benchmark's config is still a
config somebody may copy."""


class Server:
    """A real `wsindex serve`, on a real workspace, for the length of a run."""

    def __init__(self, base: str, token: str, index_dir: Path, log: Path, corpus: Path) -> None:
        self.base = base
        self.token = token
        self.index_dir = index_dir
        self.log = log
        self.corpus = corpus
        self._commits = 0

    def client(self) -> Client:
        return Client(self.base, self.token)

    def commit_one_change(self) -> None:
        """Give the next sync one changed file, the way an ordinary day does.

        Undone by `undo_commits` when the run ends: the corpus is a
        cached clone shared with the next run of this harness, and a
        harness that leaves commits behind measures its own history.
        """
        touched = next(self.corpus.rglob("*.py"))
        touched.write_text(touched.read_text() + f"\n# load touch {self._commits}\n")
        self._git("add", str(touched.relative_to(self.corpus)))
        self._git(
            "-c", "user.email=l@e.invalid", "-c", "user.name=L", "commit", "-qm", "load touch"
        )
        self._commits += 1

    def undo_commits(self) -> None:
        """Put the corpus back exactly as it was found."""
        if self._commits:
            self._git("reset", "--hard", "-q", f"HEAD~{self._commits}")
            self._commits = 0

    def _git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.corpus, check=True, capture_output=True)

    def force_full_reindex(self) -> None:
        """Make the next `POST /index` a full pass rather than a warm one.

        By removing `state.json`, which is a state the product genuinely
        reaches — `FullPass.STATE_LOST` exists for it — and the only way
        to get a run long enough for a search to overlap.
        """
        (self.index_dir / "state.json").unlink(missing_ok=True)

    def metrics(self) -> str:
        """The server's own account of what just happened."""
        client = self.client()
        try:
            return client._http.get("/metrics").text
        finally:
            client.close()


@contextmanager
def serving(corpus: Path, port: int, *, rerank: bool = False) -> Iterator[Server]:
    """Start a server on a workspace of its own; stop it afterwards.

    Its own workspace, in a temporary directory, for the reason
    `bench_corpus` gives about clones: a harness that can disturb
    somebody's real index is a harness with a bug waiting.

    Args:
        corpus: Repository the server will index.
        port: Port to bind on loopback.
        rerank: Turn on `[rank] enabled`. Off by default because that is
            the product's default and the SLO was written for it.
    """
    token = "load-" + os.urandom(8).hex()
    with tempfile.TemporaryDirectory(prefix="wsindex-load-") as home:
        root = Path(home)
        config = write_config(root, corpus, rerank=rerank)
        log = root / "server.log"
        environ = {
            **os.environ,
            "WSINDEX_CONFIG": str(config),
            TOKEN_ENV: token,
        }
        with log.open("wb") as sink:
            # The console script beside this interpreter, not `-m
            # wsindex`: the package has no `__main__`, and the whole
            # point is to drive the command a person would type.
            command = [str(Path(sys.executable).parent / "wsindex")]
            process = subprocess.Popen(
                [*command, "serve", "--host", "127.0.0.1", "--port", str(port)],
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=environ,
            )
            server = Server(f"http://127.0.0.1:{port}", token, root / ".wsindex", log, corpus)
            try:
                await_healthy(server.base, process, log)
                yield server
            finally:
                # The corpus first: it outlives this directory, and a
                # crash here must not leave commits in a cached clone.
                server.undo_commits()
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:  # pragma: no cover - a wedged server
                    process.kill()


def write_config(root: Path, corpus: Path, *, rerank: bool = False) -> Path:
    """A workspace config pointing at the corpus, with a token and no ticker."""
    from wsindex.config import Config, Provider, Repository

    Config.reset()
    fast = os.environ.get("WSINDEX_LOAD_FAST") == "1"
    config = Config.default("load", provider=Provider.FAKE if fast else None)
    config.add_repo(Repository(id="corpus", path=str(corpus)))
    if rerank:
        # The second model, as a workspace turns it on. Nothing else
        # about the run changes: the same scenarios, the same budgets,
        # because a person waiting on an answer does not know which of
        # these two configurations they are talking to.
        config._data["rank"] = {"enabled": True}
    # No `interval`: a ticker would index in the middle of the idle
    # scenario and the "idle" number would be a different measurement
    # every run. Contention is a scenario here, not a background hum.
    config._data["server"] = {"token_env": TOKEN_ENV}
    path = root / "wsindex.toml"
    config.save(path)
    Config.reset()
    return path


def await_healthy(base: str, process: subprocess.Popen[bytes], log: Path) -> None:
    """Block until `/healthz` answers, or say why it never will.

    Raises:
        RuntimeError: The server exited, or took longer than
            `STARTUP_TIMEOUT_S`. Either way the server's own log is the
            message — a load harness that says "could not connect" and
            keeps the reason to itself wastes an afternoon.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the server exited with {process.returncode}:\n{tail(log)}")
        try:
            if httpx.get(f"{base}/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError(f"the server never became healthy:\n{tail(log)}")


def tail(log: Path, lines: int = 25) -> str:
    """The end of the server's log, for a failure message."""
    try:
        return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
    except OSError:  # pragma: no cover - the log may not exist yet
        return "(no server log)"


# --- the verdict -----------------------------------------------------------


@dataclass(frozen=True)
class Judgement:
    """One SLO rule, and whether this run kept it."""

    rule: str
    ok: bool
    detail: str


def judge(results: list[Result]) -> list[Judgement]:
    """Every rule against every scenario it applies to.

    The rules are read from the constants at the top of this file and
    nowhere else, so "what the SLO is" has one answer.
    """
    verdicts: list[Judgement] = []
    for result in results:
        if result.scenario.startswith("search"):
            p95, p99 = result.percentile(0.95), result.percentile(0.99)
            verdicts.append(
                Judgement(
                    rule=f"{result.scenario}: p95 < {P95_BUDGET_MS:.0f} ms",
                    ok=p95 < P95_BUDGET_MS,
                    detail=f"{p95:.1f} ms over {len(result.samples)} requests",
                )
            )
            verdicts.append(
                Judgement(
                    rule=f"{result.scenario}: p99 < {P99_BUDGET_MS:.0f} ms",
                    ok=p99 < P99_BUDGET_MS,
                    detail=f"{p99:.1f} ms",
                )
            )
        failures = result.failures
        verdicts.append(
            Judgement(
                rule=f"{result.scenario}: no failed requests",
                ok=len(failures) <= FAILURES_ALLOWED,
                detail="none"
                if not failures
                else f"{len(failures)}: {failures[0].error or failures[0].status}",
            )
        )
    contention = next((r for r in results if r.scenario == "index-contention"), None)
    if contention is not None:
        accepted = sum(1 for sample in contention.samples if sample.status == 200)
        refused = sum(1 for sample in contention.samples if sample.status == 409)
        verdicts.append(
            Judgement(
                rule="one-writer rule: exactly one concurrent run is accepted",
                ok=accepted == 1 and accepted + refused == len(contention.samples),
                detail=f"{accepted} accepted, {refused} refused, "
                f"{len(contention.samples) - accepted - refused} neither",
            )
        )
    return verdicts


def searches_counted(server: Server) -> float | None:
    """How many `/search` requests the server's own histogram has seen.

    None when the parser is not installed, which is a skipped check
    rather than a failed one.
    """
    try:
        from prometheus_client.parser import text_string_to_metric_families

        return sum(
            sample.value
            for family in text_string_to_metric_families(server.metrics())
            for sample in family.samples
            if sample.name == "wsindex_http_request_seconds_count"
            and sample.labels.get("route") == "/search"
        )
    except Exception:  # pragma: no cover - depends on the dev install
        return None


def cross_check(before: float | None, after: float | None, results: list[Result]) -> str:
    """Compare what the client timed against what the server counted.

    Not an SLO — a check that the metrics added in this same step are
    telling the truth. Two independent measurements of the same requests
    should agree on how many there were; if they do not, one of them is
    lying and it matters which.

    A *difference* between two readings, not one reading, and the first
    version of this got that wrong: it compared the server's lifetime
    total against the scenarios' samples and reported a disagreement of
    exactly one — the warm-up search, which the client makes and does not
    count. The metrics were right and the check was wrong, which is the
    more embarrassing of the two ways round.
    """
    if before is None or after is None:
        return "metrics cross-check skipped: no exposition parser installed"
    by_server = after - before
    by_client = sum(
        len(result.samples) for result in results if result.scenario.startswith("search")
    )
    agree = "agree" if by_server == by_client else "**disagree**"
    return (
        f"the server counted {by_server:.0f} searches, the client made {by_client} — they {agree}"
    )


def render(env: dict[str, str], results: list[Result], verdicts: list[Judgement], note: str) -> str:
    """The report, in the shape the plan's journal wants."""
    lines = [
        "# wsindex load",
        "",
        " · ".join(f"{key} `{value}`" for key, value in env.items()),
        "",
        f"SLO fixed before the run: p95 < {P95_BUDGET_MS:.0f} ms, "
        f"p99 < {P99_BUDGET_MS:.0f} ms, {FAILURES_ALLOWED} failed requests.",
        "",
        "| Scenario | Reqs | rps | p50 ms | p95 ms | p99 ms | max ms | hits | fail | statuses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        row = result.summary()
        statuses = " ".join(f"{code}x{n}" for code, n in row["statuses"].items())
        lines.append(
            f"| {row['scenario']} | {row['requests']} | {row['rps']} | {row['p50_ms']} | "
            f"{row['p95_ms']} | {row['p99_ms']} | {row['max_ms']} | {row['mean_hits']} | "
            f"{row['failures']} | {statuses} |"
        )
    notes = [result.note for result in results if result.note]
    if notes:
        lines += ["", *[f"- {note}" for note in notes]]
    lines += ["", "## Against the SLO", ""]
    for verdict in verdicts:
        lines.append(f"- {'PASS' if verdict.ok else 'FAIL'} — {verdict.rule}: {verdict.detail}")
    lines += ["", f"- {note}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Start a server, run every scenario against it, judge, report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=WORKERS, help="Concurrent clients")
    parser.add_argument("--port", type=int, default=8917, help="Port to serve on")
    parser.add_argument("--json", type=Path, help="Also write the measurements as JSON")
    parser.add_argument("--only", nargs="*", choices=sorted(SCENARIOS), help="Run a subset")
    parser.add_argument(
        "--rerank", action="store_true", help="Serve with [rank] enabled (a second model)"
    )
    parser.add_argument(
        "--anyway", action="store_true", help="Measure even though the machine is busy"
    )
    args = parser.parse_args()

    disturbed = busy()
    if disturbed is not None:
        print(f"warning: {disturbed}", file=sys.stderr)
        if not args.anyway:
            # Stricter than `bench.py`, which lets an exploratory run
            # through with a warning. This one always produces a verdict
            # — PASS or FAIL against a budget — and a verdict on a
            # machine somebody else is using is worse than no verdict:
            # a FAIL sends someone hunting a regression that is not
            # there, and a PASS is a promise made on borrowed evidence.
            print(
                "refusing to judge an SLO on a busy machine (--anyway overrides)", file=sys.stderr
            )
            return 2

    corpus = bench_corpus()
    results: list[Result] = []
    with serving(corpus, args.port, rerank=args.rerank) as server:
        print("  first index (cold) ...", file=sys.stderr, flush=True)
        client = server.client()
        try:
            first = client.timed("POST", "/index")
            if first.status != 200:
                raise RuntimeError(f"the first index failed: {first.status} {first.error or ''}")
            # A warm-up search before anything is measured, for the same
            # reason `bench.py` does one: the first search through a
            # fresh process pays for whatever the model still has to load,
            # and that cost belongs to `startup`, not to a percentile.
            client.timed("GET", "/search", params={"q": QUERIES[0], "k": 5})
        finally:
            client.close()
        # Read before the scenarios, not just after: the warm-up above is
        # a real request the server counts and the scenarios do not.
        before = searches_counted(server)
        for name in args.only or list(SCENARIOS):
            print(f"  {name} ...", file=sys.stderr, flush=True)
            results.append(SCENARIOS[name](server, args.workers))
        checked = cross_check(before, searches_counted(server), results)

    env = {
        **environment(),
        # `environment()` reads `bench.py`'s flag for this, and this file
        # has its own. Left alone, the report labels a fake-embedder run
        # `real`, which is worse than no label.
        "embedder": "fake" if os.environ.get("WSINDEX_LOAD_FAST") == "1" else "real",
        # In the header, not a footnote: two runs of this harness measure
        # different products, and a table that does not say which is a
        # table somebody will compare against the wrong one.
        "rank": "on" if args.rerank else "off",
        "workers": str(args.workers),
    }
    verdicts = judge(results)
    print(render(env, results, verdicts, checked))
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "environment": env,
                    "slo": {
                        "p95_ms": P95_BUDGET_MS,
                        "p99_ms": P99_BUDGET_MS,
                        "failures": FAILURES_ALLOWED,
                    },
                    "results": [result.summary() for result in results],
                    "verdicts": [asdict(verdict) for verdict in verdicts],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"saved to {args.json}", file=sys.stderr)
    return 0 if all(verdict.ok for verdict in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
