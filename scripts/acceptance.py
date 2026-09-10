"""Automated MVP acceptance: index a real corpus, grade fixed criteria, report.

The criteria queries and their expected path fragments are fixed in code
BEFORE any run — the antidote to confirmation bias. The script indexes the
corpus on a local path and optionally against MinIO (the s3 storage
scenario, step 17d), grades every query by whether an expected fragment
surfaces in the top-k paths, cross-checks the runs, and writes a markdown
report to stdout and `acceptance_report.md`.

Usage:
    uv run python scripts/acceptance.py        # or: uv run poe acceptance

Environment:
    WSINDEX_E2E_REPO   corpus repo (default: tensorus/tensorus)
    WSINDEX_E2E_DIR    clone cache dir
    WSINDEX_ACCEPT_S3  set to "0" to skip the s3 (MinIO) run
    WSINDEX_S3_URI     s3 uri prefix (default: s3://wsindex/accept)

The s3 run assumes the compose MinIO (localhost:9000, bucket `wsindex`);
credentials default to minioadmin and can be overridden via AWS_* vars.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from wsindex.config import Config, Repository
from wsindex.embed import SentenceTransformerEmbedder
from wsindex.model import Hit
from wsindex.pipeline import IndexReport, Pipeline
from wsindex.rank.reranker import CrossEncoderReranker, Reranker
from wsindex.store import LanceDBStore, VectorStore

REPO_URL = os.environ.get("WSINDEX_E2E_REPO", "https://github.com/tensorus/tensorus")
K = 5

# Fixed acceptance criteria: (query, acceptable path fragments in top-K).
#
# Fixed before any run, and *kept* when a run fails them — which is the
# whole point and was tested on 2026-09-10. Giving the corpus its history
# (commits are indexed by default, so the shallow clone graded a product
# with a feature switched off) took this from 10/10 to 9/10. The failing
# criterion was not loosened to get the ten back.
#
# What that miss is, measured rather than guessed: `expose dataset
# operations over http` finds `tensorus/api.py` at rank 8 — inside the
# 20 candidates re-rank sees, so not a recall failure by the threshold
# the README sets. The cross-encoder scores every candidate for this
# query at 0.44 against 0.95+ for the other nine: it is saying the
# corpus has no good answer. And the chunk that *would* match is a bare
# comment header, `# --- Dataset Management Endpoints ---`, orphaned
# from the endpoints it labels. That is a chunking observation, kept
# here as one rather than acted on.
CRITERIA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("where are tensors stored on disk", ("storage",)),
    ("how is the api key validated", ("auth", "security")),
    ("parse a natural language query", ("nql",)),
    ("generate embeddings for text", ("embedding",)),
    ("compress tensors to save space", ("compression",)),
    ("build an index for faster lookups", ("index",)),
    ("expose dataset operations over http", ("api",)),
    # step 19v: identifier-flavoured queries — BM25 should own these.
    # Exact strings live in one place each: verified against the corpus
    # before pinning the expected fragments.
    ("list_to_tensor", ("api.py", "app.py")),
    ("tensorus-models>=0.0.3", ("pyproject",)),
    ("Scalar tensor data must be a single number", ("api.py", "app.py")),
)


@dataclass
class QueryResult:
    query: str
    expected: tuple[str, ...]
    hits: list[Hit]

    @property
    def hit_rank(self) -> int | None:
        for rank, hit in enumerate(self.hits, start=1):
            path = str(hit.metadata.get("path", ""))
            if any(fragment in path for fragment in self.expected):
                return rank
        return None

    @property
    def top(self) -> str:
        if not self.hits:
            return "no results"
        hit = self.hits[0]
        meta = hit.metadata
        span = f"{meta.get('start_line')}-{meta.get('end_line')}"
        return f"{meta.get('path')}:{span} ({hit.score:.3f})"


@dataclass
class BackendRun:
    name: str
    report: IndexReport
    index_seconds: float
    search_seconds: float
    results: list[QueryResult]

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.hit_rank is not None)


def ensure_corpus() -> Path:
    """The graded corpus, with its history.

    With, not `--depth 1`, and the change was earned. Indexing commit
    messages is the default (step 27), so a corpus holding one commit
    graded a configuration nobody runs: it was worth 10/10 while the
    realistic one is worth 9. The shallow clone was not measuring the
    product, it was measuring a product with the history switched off.

    An existing shallow clone is deepened in place rather than re-cloned:
    somebody's cache should not have to be deleted for this to take.
    """
    override = os.environ.get("WSINDEX_E2E_DIR")
    if override:
        corpus = Path(override).expanduser()
    else:
        name = REPO_URL.rstrip("/").rsplit("/", 1)[-1]
        corpus = Path.home() / ".cache" / "wsindex-e2e" / name
    if not corpus.exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(corpus)], check=True)
        return corpus
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=corpus,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if shallow == "true":
        subprocess.run(["git", "fetch", "--unshallow", "-q"], cwd=corpus, check=True)
    return corpus


def make_config(corpus: Path, repo_id: str) -> Config:
    # `Config.default` replaces the process-wide instance, which is exactly
    # what this script wants: it never reads a real workspace config.
    config = Config.default("acceptance")
    config.add_repo(Repository(id=repo_id, path=str(corpus)))
    return config


def run_backend(
    name: str, store: VectorStore, state_dir: Path, reranker: Reranker | None = None
) -> BackendRun:
    # Repos and metric come from the current Config, which make_config
    # installed as the process-wide instance.
    pipeline = Pipeline(store=store, state_dir=state_dir, reranker=reranker)
    started = time.perf_counter()
    report = pipeline.index()
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    results = [
        QueryResult(query, expected, pipeline.search(query, k=K)) for query, expected in CRITERIA
    ]
    search_seconds = time.perf_counter() - started
    return BackendRun(name, report, index_seconds, search_seconds, results)


@dataclass
class IncrementalRuns:
    """The step-22 measurements: what a re-index costs once nothing changed.

    Attributes:
        cold_seconds: The first, full index of the corpus — the baseline
            the two numbers below are worth comparing against.
        noop_seconds: A re-index with nothing changed at all.
        one_file_seconds: A re-index after exactly one committed edit.
        one_file_report: What that run actually did; `files` should be 1.
    """

    cold_seconds: float
    noop_seconds: float
    one_file_seconds: float
    one_file_report: IndexReport


def _git(corpus: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=corpus, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def measure_incremental(
    corpus: Path, store: VectorStore, state_dir: Path, cold: float
) -> IncrementalRuns:
    """Time a no-op re-index and a one-file re-index over the real corpus.

    The corpus is a shared clone cache, so the edit is made and then
    undone: the commit is created, measured, and the working copy is
    reset back to the sha it started at. A `finally` does the reset, so
    an exception mid-measurement cannot leave the developer's cache on a
    commit that only this script knows about.

    Args:
        corpus: The cloned corpus (a git repo, full history).
        store: The already-populated store to re-index into.
        state_dir: Where the incremental state was recorded.
        cold: Seconds the initial full index took, for the report.

    Returns:
        The three timings and the one-file report.
    """
    pipeline = Pipeline(store=store, state_dir=state_dir)

    started = time.perf_counter()
    pipeline.index()
    noop_seconds = time.perf_counter() - started

    original = _git(corpus, "rev-parse", "HEAD")
    target = next(corpus.rglob("*.py"))
    saved = target.read_text(encoding="utf-8", errors="replace")
    try:
        target.write_text(saved + "\n\n# acceptance: one-file incremental probe\n")
        _git(corpus, "add", "-A")
        # -c on the command, not global config: the acceptance run must
        # not need (or leave) a git identity on the machine.
        _git(
            corpus,
            "-c",
            "user.name=wsindex acceptance",
            "-c",
            "user.email=acceptance@wsindex.invalid",
            "commit",
            "-qm",
            "acceptance probe",
        )
        started = time.perf_counter()
        one_file_report = pipeline.index()
        one_file_seconds = time.perf_counter() - started
    finally:
        _git(corpus, "reset", "--hard", "-q", original)
    return IncrementalRuns(
        cold_seconds=cold,
        noop_seconds=noop_seconds,
        one_file_seconds=one_file_seconds,
        one_file_report=one_file_report,
    )


def render(
    corpus: Path,
    runs: list[BackendRun],
    skipped: list[str],
    incremental: IncrementalRuns | None = None,
) -> str:
    lines = [
        "# WSIndex MVP acceptance report",
        "",
        f"Corpus: `{REPO_URL}` (clone at `{corpus}`)",
        f"Criteria: {len(CRITERIA)} fixed queries, hit = expected fragment in top-{K} paths.",
        "",
    ]
    for run in runs:
        lines += [
            f"## Backend: {run.name}",
            "",
            f"Indexed {run.report.files} files, {run.report.chunks} chunks, "
            f"{run.report.written} written in {run.index_seconds:.1f}s; "
            f"{len(CRITERIA)} searches in {run.search_seconds:.1f}s.",
            "",
            "| Query | Expected | Hit rank | Top hit |",
            "| --- | --- | --- | --- |",
        ]
        for result in run.results:
            rank = str(result.hit_rank) if result.hit_rank is not None else "MISS"
            lines.append(
                f"| {result.query} | {'/'.join(result.expected)} | {rank} | {result.top} |"
            )
        lines += ["", f"**Verdict: {run.passed}/{len(run.results)} criteria passed.**", ""]
    for right_run in runs[1:]:
        lines += [
            f"## Cross-check: {runs[0].name} vs {right_run.name}",
            "",
            "| Query | Same top-1 | Score delta |",
            "| --- | --- | --- |",
        ]
        agreements = 0
        for left, right in zip(runs[0].results, right_run.results, strict=True):
            if not left.hits or not right.hits:
                lines.append(f"| {left.query} | n/a | n/a |")
                continue
            same = left.hits[0].metadata.get("path") == right.hits[0].metadata.get("path")
            delta = abs(left.hits[0].score - right.hits[0].score)
            agreements += same and delta < 1e-3
            lines.append(f"| {left.query} | {'yes' if same else 'NO'} | {delta:.4f} |")
        lines += ["", f"**Backends agree on {agreements}/{len(CRITERIA)} queries.**", ""]
    if incremental is not None:
        one = incremental.one_file_report
        lines += [
            "## Incremental re-index (step 22)",
            "",
            "| Scenario | Files read | Chunks written | Chunks deleted | Seconds |",
            "| --- | --- | --- | --- | --- |",
            f"| cold (first index) | - | - | - | {incremental.cold_seconds:.1f} |",
            f"| no changes | 0 | 0 | 0 | {incremental.noop_seconds:.2f} |",
            f"| one changed file | {one.files} | {one.written} | {one.deleted} "
            f"| {incremental.one_file_seconds:.2f} |",
            "",
            f"**Cold index {incremental.cold_seconds:.1f}s -> "
            f"no-op {incremental.noop_seconds:.2f}s "
            f"({incremental.cold_seconds / max(incremental.noop_seconds, 1e-9):.0f}x), "
            f"one changed file {incremental.one_file_seconds:.2f}s "
            f"({incremental.cold_seconds / max(incremental.one_file_seconds, 1e-9):.0f}x).**",
            "",
        ]
    for note in skipped:
        lines += [f"_{note}_", ""]
    return "\n".join(lines)


def run_s3(corpus: Path, embedder: SentenceTransformerEmbedder, state_dir: Path) -> BackendRun:
    """The step-17d storage scenario: the same store code over MinIO.

    Credentials default to the compose MinIO so `make acceptance` works
    out of the box; real S3 overrides them via the standard AWS_* vars.
    """
    os.environ.setdefault("AWS_ENDPOINT", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ROOT_USER", "minioadmin"))
    os.environ.setdefault(
        "AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
    )
    os.environ.setdefault("AWS_ALLOW_HTTP", "true")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    prefix = os.environ.get("WSINDEX_S3_URI", "s3://wsindex/accept")
    uri = f"{prefix}_{uuid.uuid4().hex[:8]}"
    store = LanceDBStore(uri=uri, embedder=embedder)
    try:
        # Reinstalls the process-wide Config: the pipeline reads its repo
        # list from `Config()` at call time, not from a constructor arg.
        make_config(corpus, "corpus")
        return run_backend("local-s3", store, state_dir / "s3")
    finally:
        # Leave the bucket clean: a re-run must not inherit our tables.
        # list_tables() returns a response object, not a list of names.
        for name in store.db.list_tables().tables:
            store.db.drop_table(name)


def main() -> None:
    corpus = ensure_corpus()
    runs: list[BackendRun] = []
    skipped: list[str] = []

    # Installs the acceptance config first, so the embedder picks its model
    # up from it — same source the two backends read everything else from.
    make_config(corpus, "corpus")
    embedder = SentenceTransformerEmbedder(Config().model)
    incremental: IncrementalRuns | None = None
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(corpus, "corpus")
        state_root = Path(tmp) / "state"
        store = LanceDBStore(uri=str(Path(tmp) / ".wsindex"), embedder=embedder)
        # Each backend gets its own state dir: a second backend starting
        # from the first one's recorded commit would index nothing and
        # measure nothing.
        local = run_backend("local", store, state_root / "local")
        runs.append(local)

        # Step 22: what a re-index costs once the corpus is already indexed.
        # Same store, same state the run above just recorded.
        if os.environ.get("WSINDEX_ACCEPT_INCREMENTAL", "1") != "0":
            incremental = measure_incremental(
                corpus, store, state_root / "local", local.index_seconds
            )
        else:
            skipped.append("incremental measurement skipped: WSINDEX_ACCEPT_INCREMENTAL=0")

        # Same corpus, same store — but with the cross-encoder reranker on top.
        # The cross-check delta shows how much re-rank moved ranks.
        if os.environ.get("WSINDEX_ACCEPT_RERANK", "1") != "0":
            reranker = CrossEncoderReranker(model_name=config.rank_model)
            runs.append(
                run_backend("local-reranked", store, state_root / "reranked", reranker=reranker)
            )
        else:
            skipped.append("rerank run skipped: WSINDEX_ACCEPT_RERANK=0")

    if os.environ.get("WSINDEX_ACCEPT_S3", "1") == "0":
        skipped.append("s3 run skipped: WSINDEX_ACCEPT_S3=0")
    else:
        try:
            runs.append(run_s3(corpus, embedder, state_root))
        except ValueError as exc:
            skipped.append(f"s3 run skipped: MinIO is not reachable ({exc})")

    text = render(corpus, runs, skipped, incremental)
    print(text)
    Path("acceptance_report.md").write_text(text, encoding="utf-8")
    print("written: acceptance_report.md")


if __name__ == "__main__":
    main()
