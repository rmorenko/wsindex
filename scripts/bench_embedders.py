"""Step 19d spike: compare embedders on the acceptance corpus.

Runs the same 10 acceptance criteria against several sentence-transformers
models to see whether swapping the embedder alone moves ranks — the
data-driven check for a default-model change (or a fast/quality flavor
split). No engine code is modified; each run gets its own tmp store.

Usage:
    uv run python scripts/bench_embedders.py

The matrix is defined below; skip a row by commenting it out. Models are
downloaded on first use (HF cache) and reused on subsequent runs.
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

# scripts/ is not a package; put its parent on the path so we can reuse
# the acceptance grader (CRITERIA + QueryResult) without duplicating it.
sys.path.insert(0, str(Path(__file__).parent))

from acceptance import (
    CRITERIA,
    K,
    QueryResult,
    ensure_corpus,
    make_config,
)

from wsindex.embed.embedder import Embedder
from wsindex.pipeline import Pipeline
from wsindex.store.lancedb import LanceDBStore

# (label, model_name, trust_remote_code).
# Ordered from cheapest to heaviest so partial runs still yield the
# baseline first.
MODELS: tuple[tuple[str, str, bool], ...] = (
    ("miniLM-L6 (baseline)", "sentence-transformers/all-MiniLM-L6-v2", False),
    ("mpnet-base", "sentence-transformers/all-mpnet-base-v2", False),
    ("jina-code", "jinaai/jina-embeddings-v2-base-code", True),
    ("bge-large", "BAAI/bge-large-en-v1.5", False),
)


class _AdHocEmbedder(Embedder):
    """Wraps a SentenceTransformer with per-model init knobs.

    Kept in the spike so the production adapter stays free of the
    trust_remote_code footgun — only jina-code needs it, and we don't
    want that switch in the main API surface.
    """

    def __init__(self, model_name: str, *, trust_remote_code: bool = False) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        dim: int | None = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"{model_name} does not report an embedding dimension")
        self._dim: int = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return cast("list[list[float]]", vectors.tolist())


@dataclass
class ModelRun:
    label: str
    model_name: str
    dim: int
    load_seconds: float
    index_seconds: float
    search_seconds: float
    results: list[QueryResult]
    failed: str | None = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.hit_rank is not None)

    @property
    def mean_rank(self) -> float | None:
        ranks = [r.hit_rank for r in self.results if r.hit_rank is not None]
        return sum(ranks) / len(ranks) if ranks else None


def run_model(label: str, model_name: str, trust_remote_code: bool, corpus: Path) -> ModelRun:
    print(f"\n=== {label} ({model_name}) ===")
    load_start = time.perf_counter()
    try:
        embedder = _AdHocEmbedder(model_name, trust_remote_code=trust_remote_code)
    except Exception as exc:
        print(f"  FAILED to load: {exc}")
        return ModelRun(label, model_name, 0, 0.0, 0.0, 0.0, [], failed=str(exc))
    load_seconds = time.perf_counter() - load_start
    print(f"  loaded (dim={embedder.dim}) in {load_seconds:.1f}s")

    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBStore(uri=str(Path(tmp) / ".wsindex"), embedder=embedder)
        # Installs the process-wide Config; the pipeline reads its repo
        # list from `Config()` at call time, not from a constructor arg.
        make_config(corpus, "corpus")
        pipeline = Pipeline(store=store)
        started = time.perf_counter()
        pipeline.index()
        index_seconds = time.perf_counter() - started
        print(f"  indexed in {index_seconds:.1f}s")
        started = time.perf_counter()
        results = [
            QueryResult(query, expected, pipeline.search(query, k=K))
            for query, expected in CRITERIA
        ]
        search_seconds = time.perf_counter() - started
        print(f"  searched in {search_seconds:.1f}s")

    return ModelRun(
        label, model_name, embedder.dim, load_seconds, index_seconds, search_seconds, results
    )


def render(runs: list[ModelRun]) -> str:
    lines = [
        "# Step 19d — embedder comparison (spike)",
        "",
        f"Corpus: 149 files, ~3.5k chunks. Criteria: {len(CRITERIA)} fixed queries, "
        f"hit = expected fragment in top-{K} paths.",
        "",
        "## Summary",
        "",
        "| Model | Dim | Load (s) | Index (s) | Search (s) | Passed | Mean rank |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        if run.failed:
            lines.append(f"| {run.label} | — | — | — | — | FAILED: {run.failed[:40]} | — |")
            continue
        mean = f"{run.mean_rank:.2f}" if run.mean_rank is not None else "—"
        lines.append(
            f"| {run.label} | {run.dim} | {run.load_seconds:.1f} | "
            f"{run.index_seconds:.1f} | {run.search_seconds:.2f} | "
            f"{run.passed}/{len(CRITERIA)} | {mean} |"
        )
    lines += ["", "## Ranks per query", ""]
    header = "| Query | " + " | ".join(r.label for r in runs if not r.failed) + " |"
    sep = "| --- |" + " --- |" * sum(1 for r in runs if not r.failed)
    lines += [header, sep]
    ok_runs = [r for r in runs if not r.failed]
    for i, (query, _) in enumerate(CRITERIA):
        cells = []
        for run in ok_runs:
            rank = run.results[i].hit_rank
            cells.append(str(rank) if rank is not None else "MISS")
        lines.append(f"| {query} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    corpus = ensure_corpus()
    runs: list[ModelRun] = []
    for label, model_name, trust_remote_code in MODELS:
        try:
            runs.append(run_model(label, model_name, trust_remote_code, corpus))
        except Exception as exc:
            print(f"  FAILED at run: {exc}")
            runs.append(ModelRun(label, model_name, 0, 0.0, 0.0, 0.0, [], failed=str(exc)))
    text = render(runs)
    print("\n\n" + text)
    Path("bench_embedders_report.md").write_text(text, encoding="utf-8")
    print("\nwritten: bench_embedders_report.md")


if __name__ == "__main__":
    main()
