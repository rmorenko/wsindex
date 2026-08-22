"""Automated MVP acceptance: index a real corpus, grade fixed criteria, report.

The criteria queries and their expected path fragments are fixed in code
BEFORE any run — the antidote to confirmation bias. The script indexes the
corpus on a local path and optionally against MinIO (the s3 storage
scenario, step 17d), grades every query by whether an expected fragment
surfaces in the top-k paths, cross-checks the runs, and writes a markdown
report to stdout and `acceptance_report.md`.

Usage:
    uv run python scripts/acceptance.py        # or: make acceptance

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

from wsindex.config import Config
from wsindex.embed.embedder import SentenceTransformerEmbedder
from wsindex.model import Hit
from wsindex.pipeline import IndexReport, Pipeline
from wsindex.rank.reranker import CrossEncoderReranker, Reranker
from wsindex.store.base import VectorStore
from wsindex.store.lancedb import LanceDBStore

REPO_URL = os.environ.get("WSINDEX_E2E_REPO", "https://github.com/tensorus/tensorus")
K = 5

# Fixed acceptance criteria: (query, acceptable path fragments in top-K).
CRITERIA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("where are tensors stored on disk", ("storage",)),
    ("how is the api key validated", ("auth", "security")),
    ("parse a natural language query", ("nql",)),
    ("generate embeddings for text", ("embedding",)),
    ("compress tensors to save space", ("compression",)),
    ("build an index for faster lookups", ("index",)),
    ("expose dataset operations over http", ("api",)),
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
    override = os.environ.get("WSINDEX_E2E_DIR")
    if override:
        corpus = Path(override).expanduser()
    else:
        name = REPO_URL.rstrip("/").rsplit("/", 1)[-1]
        corpus = Path.home() / ".cache" / "wsindex-e2e" / name
    if not corpus.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(corpus)],
            check=True,
            capture_output=True,
        )
    return corpus


def make_config(corpus: Path, repo_id: str) -> Config:
    config = Config.default_config("acceptance")
    config.add_repo(repo_id, path=str(corpus))
    return config


def run_backend(
    name: str, store: VectorStore, config: Config, reranker: Reranker | None = None
) -> BackendRun:
    pipeline = Pipeline(config=config, store=store, reranker=reranker)
    started = time.perf_counter()
    report = pipeline.index()
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    results = [
        QueryResult(query, expected, pipeline.search(query, k=K)) for query, expected in CRITERIA
    ]
    search_seconds = time.perf_counter() - started
    return BackendRun(name, report, index_seconds, search_seconds, results)


def render(corpus: Path, runs: list[BackendRun], skipped: list[str]) -> str:
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
    for note in skipped:
        lines += [f"_{note}_", ""]
    return "\n".join(lines)


def run_s3(corpus: Path, embedder: SentenceTransformerEmbedder) -> BackendRun:
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
        return run_backend("local-s3", store, make_config(corpus, "corpus"))
    finally:
        # Leave the bucket clean: a re-run must not inherit our tables.
        # list_tables() returns a response object, not a list of names.
        for name in store.db.list_tables().tables:
            store.db.drop_table(name)


def main() -> None:
    corpus = ensure_corpus()
    runs: list[BackendRun] = []
    skipped: list[str] = []

    embedder = SentenceTransformerEmbedder(model_name=Config.default_config("x").model)
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(corpus, "corpus")
        store = LanceDBStore(uri=str(Path(tmp) / ".wsindex"), embedder=embedder)
        runs.append(run_backend("local", store, config))

        # Same corpus, same store — but with the cross-encoder reranker on top.
        # The cross-check delta shows how much re-rank moved ranks.
        if os.environ.get("WSINDEX_ACCEPT_RERANK", "1") != "0":
            reranker = CrossEncoderReranker(model_name=config.rank_model)
            runs.append(run_backend("local-reranked", store, config, reranker=reranker))
        else:
            skipped.append("rerank run skipped: WSINDEX_ACCEPT_RERANK=0")

    if os.environ.get("WSINDEX_ACCEPT_S3", "1") == "0":
        skipped.append("s3 run skipped: WSINDEX_ACCEPT_S3=0")
    else:
        try:
            runs.append(run_s3(corpus, embedder))
        except ValueError as exc:
            skipped.append(f"s3 run skipped: MinIO is not reachable ({exc})")

    text = render(corpus, runs, skipped)
    print(text)
    Path("acceptance_report.md").write_text(text, encoding="utf-8")
    print("written: acceptance_report.md")


if __name__ == "__main__":
    main()
