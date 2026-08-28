"""Automated MVP acceptance: index a real corpus, grade fixed criteria, report.

The criteria queries and their expected path fragments are fixed in code
BEFORE any run — the antidote to confirmation bias. The script indexes the
corpus with the local backend, optionally with the tensorus backend
(needs TENSORUS_API_KEY and a running server), grades every query by
whether an expected fragment surfaces in the top-k paths, cross-checks
the two backends, and writes a markdown report to stdout and
`acceptance_report.md`.

Usage:
    uv run python scripts/acceptance.py        # or: uv run poe acceptance

Environment:
    WSINDEX_E2E_REPO         corpus repo (default: tensorus/tensorus)
    WSINDEX_E2E_DIR          clone cache dir
    WSINDEX_ACCEPT_TENSORUS  set to "0" to skip the tensorus half
    TENSORUS_API_KEY         enables the tensorus half
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
from wsindex.store.base import VectorStore
from wsindex.store.local import LocalStore
from wsindex.store.tensorus import TensorusStore

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
    config.add_repo(repo_id, str(corpus))
    return config


def run_backend(name: str, store: VectorStore, config: Config) -> BackendRun:
    pipeline = Pipeline(config=config, store=store)
    started = time.perf_counter()
    report = pipeline.index()
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    results = [
        QueryResult(query, expected, pipeline.search(query, k=K)) for query, expected in CRITERIA
    ]
    search_seconds = time.perf_counter() - started
    return BackendRun(name, report, index_seconds, search_seconds, results)


def render(corpus: Path, runs: list[BackendRun], skipped: str | None) -> str:
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
    if len(runs) == 2:
        lines += [
            "## Backend cross-check",
            "",
            "| Query | Same top-1 | Score delta |",
            "| --- | --- | --- |",
        ]
        agreements = 0
        for left, right in zip(runs[0].results, runs[1].results, strict=True):
            if not left.hits or not right.hits:
                lines.append(f"| {left.query} | n/a | n/a |")
                continue
            same = left.hits[0].metadata.get("path") == right.hits[0].metadata.get("path")
            delta = abs(left.hits[0].score - right.hits[0].score)
            agreements += same and delta < 1e-3
            lines.append(f"| {left.query} | {'yes' if same else 'NO'} | {delta:.4f} |")
        lines += ["", f"**Backends agree on {agreements}/{len(CRITERIA)} queries.**", ""]
    if skipped:
        lines += [f"_Tensorus half skipped: {skipped}_", ""]
    return "\n".join(lines)


def main() -> None:
    corpus = ensure_corpus()
    runs: list[BackendRun] = []

    embedder = SentenceTransformerEmbedder(model_name=Config.default_config("x").model)
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(corpus, "corpus")
        store = LocalStore(root=Path(tmp) / ".wsindex", embedder=embedder)
        runs.append(run_backend("local", store, config))

    skipped: str | None = None
    api_key = os.environ.get("TENSORUS_API_KEY")
    if os.environ.get("WSINDEX_ACCEPT_TENSORUS", "1") == "0":
        skipped = "WSINDEX_ACCEPT_TENSORUS=0"
    elif not api_key:
        skipped = "TENSORUS_API_KEY is not set"
    else:
        dataset = f"accept_{uuid.uuid4().hex[:8]}"
        config = make_config(corpus, dataset)
        tensorus = TensorusStore(
            base_url=config.base_url, api_key=api_key, model_name=config.model, timeout=300.0
        )
        try:
            runs.append(run_backend("tensorus", tensorus, config))
        finally:
            tensorus.client.delete(f"/datasets/{dataset}")
            tensorus.close()

    text = render(corpus, runs, skipped)
    print(text)
    Path("acceptance_report.md").write_text(text, encoding="utf-8")
    print("written: acceptance_report.md")


if __name__ == "__main__":
    main()
