"""Performance benchmarks: the same corpus, the same numbers, every time.

The project has measured a lot — the plan is full of it — but always by
hand, one probe per question, thrown away afterwards. This is the same
discipline made repeatable: a fixed corpus, named scenarios, and a
baseline to compare against, so "did that change make things worse" is a
command rather than an afternoon.

**Every scenario runs in its own process.** Not tidiness: review 4
reported a 130x speed-up that turned out to be a model already loaded in
the same interpreter, and re-measuring in fresh processes gave 2.8x. A
benchmark that shares an interpreter between scenarios measures the
order they ran in. Each child also reports its own peak RSS, which is
the only way to get a per-scenario figure out of a high-water mark.

Usage:
    uv run poe bench                      # run and print a report
    uv run poe bench -- --save bench.json # keep it as a baseline
    uv run poe bench -- --baseline bench.json   # compare against one

Environment:
    WSINDEX_E2E_REPO   corpus repo (the same one acceptance grades on)
    WSINDEX_BENCH_DIR  clone cache dir; separate from acceptance's on
                       purpose — see `bench_corpus`
    WSINDEX_BENCH_FAST set to "1" for the fake embedder — for checking
                       the harness itself, never for a real number.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from acceptance import CRITERIA, REPO_URL

SEARCH_RUNS = 20
"""How many searches make one latency figure. Twenty of ten fixed
queries is 200 samples — enough for a p95 that does not swing on one
slow disk read, and still seconds rather than minutes."""

REPEATS = 3
"""How many processes each scenario is measured in; the report is their
**minimum**.

The minimum because a benchmark cannot make the machine quieter, and the
fastest run is the one least disturbed by whatever else the laptop was
doing.

A whole process per repeat, not a loop inside one — and that is the
second time this file learned it. Repeating in-process first, `startup`
fell from 3.36 s to 0.07 s, because the second repeat found the model
already warm. It was measuring the order of its own repeats, which is
precisely the mistake review 4 made and re-measured its way out of.
Every measurement here gets an interpreter of its own, uniformly, so no
scenario can be quietly wrong in that particular way."""

REGRESSION = 0.15
"""How much slower than the baseline counts as a regression.

Measured, and the measuring is the interesting part. The first version
of this file asserted 25% and was simply wrong: three back-to-back runs
of the whole harness differed by **127%** on `compact`, 52% on `startup`
and 47% on `search-rerank`. A threshold under that reports the machine,
not the code.

With `REPEATS` fresh processes per scenario and the minimum of them, two
full runs now differ by **at most 3%** on every scenario. 15% is five
times that floor: loose enough not to cry at the laptop, tight enough
that a real regression cannot hide under it.

`FLOOR` is the other half. `compact` takes 0.02 s, where even a 127%
swing is 25 milliseconds and nobody cares."""

FLOOR = 0.10
"""Seconds below which a change is not worth reporting whatever its
percentage. Without it the fastest scenarios cry loudest."""


@dataclass
class Measurement:
    """What one scenario cost.

    Attributes:
        scenario: Its name, stable across runs so a baseline can match.
        seconds: Wall clock for the work itself, model loading excluded
            where the scenario says so.
        peak_mb: Peak resident memory of the child that ran it.
        detail: Whatever else that scenario knows — chunk counts, index
            size, p50/p95. Free-form because the scenarios genuinely
            measure different things.
    """

    scenario: str
    seconds: float
    peak_mb: float
    detail: dict[str, Any] = field(default_factory=dict)


def peak_mb() -> float:
    """This process's high-water resident memory, in MB.

    `ru_maxrss` is bytes on macOS and kilobytes on Linux, which is the
    kind of difference that silently turns 400 MB into 400 GB in a
    report.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024


def tree_bytes(path: Path) -> int:
    """Everything under a directory, for the index-size figure."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# --- the scenarios, each run in a child of its own ------------------------


def build(corpus: Path, index_dir: Path) -> Any:
    """A pipeline over the fixed corpus, wired the way the CLI wires one."""
    from wsindex.config import Config, Repository
    from wsindex.embed import FakeEmbedder, SentenceTransformerEmbedder
    from wsindex.links import LinkStore
    from wsindex.pipeline import Pipeline
    from wsindex.store import LanceDBStore

    config = Config.default("bench")
    config._data["store"] = {"uri": str(index_dir / "db")}
    config.add_repo(Repository(id="corpus", path=str(corpus)))
    fast = os.environ.get("WSINDEX_BENCH_FAST") == "1"
    embedder = (
        FakeEmbedder(dim=384) if fast else SentenceTransformerEmbedder(config.model, dim=config.dim)
    )
    store = LanceDBStore(uri=str(index_dir / "db"), embedder=embedder)
    return Pipeline(store=store, state_dir=index_dir, config=config, links=LinkStore(index_dir))


def scenario_index_cold(corpus: Path, index_dir: Path) -> Measurement:
    """A first index of the whole corpus: the number people wait through."""
    pipeline = build(corpus, index_dir)
    started = time.monotonic()
    report = pipeline.index()
    elapsed = time.monotonic() - started
    return Measurement(
        scenario="index-cold",
        seconds=elapsed,
        peak_mb=peak_mb(),
        detail={
            "files": report.files,
            "chunks": report.chunks,
            "commits": report.commits,
            "index_mb": round(tree_bytes(index_dir) / 1024 / 1024, 2),
        },
    )


def scenario_index_warm(corpus: Path, index_dir: Path) -> Measurement:
    """A re-index with nothing changed — the incremental fast path."""
    build(corpus, index_dir).index()
    pipeline = build(corpus, index_dir)
    started = time.monotonic()
    report = pipeline.index()
    return Measurement(
        scenario="index-warm",
        seconds=time.monotonic() - started,
        peak_mb=peak_mb(),
        detail={"files": report.files, "written": report.written},
    )


def scenario_index_one_file(corpus: Path, index_dir: Path) -> Measurement:
    """One file changed and committed: what an ordinary edit costs."""
    build(corpus, index_dir).index()
    touched = next(corpus.rglob("*.py"))
    touched.write_text(touched.read_text() + "\n# bench touch\n")
    _git(corpus, "add", str(touched.relative_to(corpus)))
    _git(corpus, "-c", "user.email=b@e.invalid", "-c", "user.name=B", "commit", "-qm", "bench")
    try:
        pipeline = build(corpus, index_dir)
        started = time.monotonic()
        report = pipeline.index()
        elapsed = time.monotonic() - started
    finally:
        _git(corpus, "reset", "--hard", "-q", "HEAD~1")
    return Measurement(
        scenario="index-one-file",
        seconds=elapsed,
        peak_mb=peak_mb(),
        detail={"files": report.files, "written": report.written},
    )


def _git(corpus: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=corpus, check=True, capture_output=True)


def scenario_search(corpus: Path, index_dir: Path) -> Measurement:
    """Search latency, model already loaded — what the shell feels."""
    return _search(corpus, index_dir, rerank=False)


def scenario_search_rerank(corpus: Path, index_dir: Path) -> Measurement:
    """The same, with the cross-encoder on: what re-ranking costs."""
    return _search(corpus, index_dir, rerank=True)


def _search(corpus: Path, index_dir: Path, *, rerank: bool) -> Measurement:
    """Time the fixed acceptance queries, warm, and report p50/p95.

    The model load is deliberately outside the measurement: it is
    reported by `index-cold` and by `startup`, and folding it in here
    would drown the thing being measured. One warm-up query first, for
    the same reason.
    """
    build(corpus, index_dir).index()
    pipeline = build(corpus, index_dir)
    if rerank:
        from wsindex.rank.reranker import CrossEncoderReranker

        pipeline = type(pipeline)(
            store=pipeline.store,
            state_dir=index_dir,
            config=pipeline.config,
            links=pipeline.links,
            reranker=CrossEncoderReranker(model_name=pipeline.config.rank_model),
        )
    queries = [query for query, _ in CRITERIA]
    pipeline.search(queries[0], k=5)
    samples: list[float] = []
    started = time.monotonic()
    for _ in range(SEARCH_RUNS):
        for query in queries:
            at = time.perf_counter()
            pipeline.search(query, k=5)
            samples.append((time.perf_counter() - at) * 1000)
    return Measurement(
        scenario="search-rerank" if rerank else "search",
        seconds=time.monotonic() - started,
        peak_mb=peak_mb(),
        detail={
            "samples": len(samples),
            "p50_ms": round(statistics.median(samples), 2),
            "p95_ms": round(sorted(samples)[int(len(samples) * 0.95)], 2),
        },
    )


def scenario_startup(corpus: Path, index_dir: Path) -> Measurement:
    """Everything before the first answer: imports, model, store.

    The number a person actually experiences on `wsindex search`, and
    the reason `wsindex shell` exists.
    """
    started = time.monotonic()
    pipeline = build(corpus, index_dir)
    pipeline.store.embedder.embed(["warm the model"])
    return Measurement(
        scenario="startup",
        seconds=time.monotonic() - started,
        peak_mb=peak_mb(),
        detail={},
    )


def scenario_compact(corpus: Path, index_dir: Path) -> Measurement:
    """Reclaiming the disk an edited index still occupies."""
    build(corpus, index_dir).index()
    store = build(corpus, index_dir).store
    before = tree_bytes(index_dir)
    started = time.monotonic()
    report = store.compact()
    return Measurement(
        scenario="compact",
        seconds=time.monotonic() - started,
        peak_mb=peak_mb(),
        detail={
            "versions": f"{report.versions_before} -> {report.versions_after}",
            "mb": f"{before / 1024 / 1024:.2f} -> {tree_bytes(index_dir) / 1024 / 1024:.2f}",
        },
    )


SCENARIOS = {
    "startup": scenario_startup,
    "index-cold": scenario_index_cold,
    "index-warm": scenario_index_warm,
    "index-one-file": scenario_index_one_file,
    "search": scenario_search,
    "search-rerank": scenario_search_rerank,
    "compact": scenario_compact,
}
"""Every scenario, in the order a report reads best: what you wait for
before anything, then indexing from slowest path to fastest, then the
queries, then housekeeping."""


# --- running them, comparing them, reporting them -------------------------


def run_scenario(name: str, corpus: Path) -> Measurement:
    """Measure one scenario `REPEATS` times, in a fresh process each, take the best.

    Args:
        name: Scenario to run.
        corpus: The fixed corpus.

    Returns:
        The fastest of the repeats, carrying all of them in `detail`
        under `runs` — a reader who sees 8.1 and 12.4 side by side knows
        not to trust either very far.

    Raises:
        RuntimeError: A repeat failed; its stderr is the message.
    """
    samples = [_one(name, corpus) for _ in range(REPEATS)]
    best = min(samples, key=lambda m: m.seconds)
    best.detail["runs"] = [round(m.seconds, 3) for m in samples]
    return best


def _one(name: str, corpus: Path) -> Measurement:
    """One scenario, one interpreter."""
    finished = subprocess.run(
        [sys.executable, __file__, "--scenario", name, "--corpus", str(corpus)],
        capture_output=True,
        text=True,
    )
    if finished.returncode != 0:
        raise RuntimeError(f"scenario {name} failed:\n{finished.stderr[-2000:]}")
    return Measurement(**json.loads(finished.stdout.strip().splitlines()[-1]))


def environment() -> dict[str, str]:
    """What the numbers are numbers *of*. A benchmark without this is a rumour."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "commit": commit + ("-dirty" if dirty else ""),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "embedder": "fake" if os.environ.get("WSINDEX_BENCH_FAST") == "1" else "real",
    }


def compare(runs: list[Measurement], saved: dict[str, Any], env: dict[str, str]) -> list[str]:
    """Lines naming every scenario that moved past the noise floor.

    Both directions. An unexplained speed-up is as much a signal as a
    slowdown — review 4 chased one and found a measurement error.

    Args:
        runs: What this machine just measured.
        saved: A whole saved run, environment included.
        env: This run's environment, to compare against the saved one.

    Returns:
        One line per scenario that moved, headed by a warning when the
        two runs did not come from the same machine.
    """
    before_env = saved.get("environment", {})
    lines: list[str] = []
    for key in ("platform", "python", "embedder"):
        if before_env.get(key) not in (None, env[key]):
            # Numbers from another machine are not a baseline, they are
            # a different question. Said rather than silently compared.
            lines.append(
                f"- **the baseline is not from here**: {key} was "
                f"`{before_env[key]}`, now `{env[key]}` — read the rest as trivia"
            )
    was = {row["scenario"]: row["seconds"] for row in saved["runs"]}
    for run in runs:
        before = was.get(run.scenario)
        if not before:
            continue
        change = (run.seconds - before) / before
        if abs(change) < REGRESSION or abs(run.seconds - before) < FLOOR:
            continue
        word = "slower" if change > 0 else "faster"
        lines.append(
            f"- **{run.scenario}**: {before:.2f}s -> {run.seconds:.2f}s ({abs(change):.0%} {word})"
        )
    return lines


def render(env: dict[str, str], runs: list[Measurement], drift: list[str] | None) -> str:
    """The report, in the shape the plan's journal wants."""
    head = " · ".join(f"{key} `{value}`" for key, value in env.items())
    lines = [
        "# wsindex benchmarks",
        "",
        head,
        "",
        "| Scenario | Seconds | Peak MB | Detail |",
        "| --- | ---: | ---: | --- |",
    ]
    for run in runs:
        detail = ", ".join(f"{key} {value}" for key, value in run.detail.items())
        lines.append(f"| {run.scenario} | {run.seconds:.2f} | {run.peak_mb:.0f} | {detail} |")
    if drift:
        lines += ["", f"## Moved more than {REGRESSION:.0%} against the baseline", "", *drift]
    elif drift is not None:
        # `None` means no baseline was given, and saying nothing is
        # right; an empty list means one was and nothing moved, which is
        # the result worth printing.
        lines += ["", f"No scenario moved more than {REGRESSION:.0%} against the baseline."]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run every scenario, or one in a child; report; flag regressions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), help=argparse.SUPPRESS)
    parser.add_argument("--corpus", help=argparse.SUPPRESS)
    parser.add_argument("--baseline", type=Path, help="Compare against a saved run")
    parser.add_argument("--save", type=Path, help="Write the measurements as JSON")
    parser.add_argument("--only", nargs="*", choices=sorted(SCENARIOS), help="Run a subset")
    args = parser.parse_args()

    if args.scenario:
        # A child: one scenario, once, in an index directory of its own.
        # Repeats happen out there, one process each.
        with tempfile.TemporaryDirectory() as index_dir:
            measured = SCENARIOS[args.scenario](Path(args.corpus), Path(index_dir))
        print(json.dumps(asdict(measured)))
        return 0

    corpus = bench_corpus()
    wanted = args.only or list(SCENARIOS)
    runs = []
    for name in wanted:
        print(f"  {name} ...", file=sys.stderr, flush=True)
        runs.append(run_scenario(name, corpus))

    env = environment()
    drift = compare(runs, json.loads(args.baseline.read_text()), env) if args.baseline else None
    print(render(env, runs, drift))
    if args.save:
        args.save.write_text(
            json.dumps({"environment": env, "runs": [asdict(r) for r in runs]}, indent=2) + "\n"
        )
        print(f"saved to {args.save}", file=sys.stderr)
    return 1 if drift else 0


def bench_corpus() -> Path:
    """A clone of the acceptance corpus, with its history, of our own.

    Of our own, and the reason is independence rather than shape. Both
    harnesses now want the same thing — a corpus with history, since
    indexing commits is what the product does by default — so they could
    share one. They do not, because they did once: this file unshallowed
    the shared cache and acceptance silently went from 10/10 to 9/10.
    Nothing about that was visible from here. A harness that can change
    another harness's verdict is a harness with a bug, and the cure that
    outlives the specific mistake is a clone each.
    """
    home = Path(os.environ.get("WSINDEX_BENCH_DIR", Path.home() / ".cache" / "wsindex-bench"))
    corpus = home / REPO_URL.rstrip("/").rsplit("/", 1)[-1]
    if not corpus.exists():
        print(f"  cloning corpus with history into {corpus} (once) ...", file=sys.stderr)
        corpus.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO_URL, str(corpus)], check=True)
    return corpus


if __name__ == "__main__":
    raise SystemExit(main())
