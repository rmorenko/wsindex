"""Indexing pipeline: config repos -> walk -> chunk -> embed -> store.

The pipeline sees only the VectorStore and Embedder contracts; concrete
backends are constructed at the edge (CLI, plan step 11) and injected.
"""

from dataclasses import dataclass
from pathlib import Path

from wsindex.config import Config
from wsindex.embed.embedder import Embedder
from wsindex.ingest.chunker import chunk_file
from wsindex.ingest.walker import walk_repo
from wsindex.store.base import VectorStore


@dataclass(frozen=True)
class IndexReport:
    """Immutable summary of one index() run, totals across all repos.

    `written` below `chunks` means dedup skipped already-stored chunks.
    """

    files: int
    chunks: int
    written: int
    missing_repos: tuple[str, ...]


def index(config: Config, store: VectorStore, embedder: Embedder) -> IndexReport:
    """Index every repo from the config into its own dataset (dataset = repo id).

    Decisions fixed here: files are read with errors="replace" so a stray
    non-UTF-8 file cannot abort the run; a repo whose directory does not
    exist goes to `missing_repos` and is skipped (an existing repo with
    zero indexable files is NOT missing). Embedding is batched per file.
    """
    files = 0
    chunks_count = 0
    written = 0
    missing_repos: list[str] = []
    for repo in config.repos:
        if not Path(repo.path).is_dir():
            missing_repos.append(repo.id)
            continue
        store.create(dataset=repo.id, dim=embedder.dim, metric=config.metric)
        repo_files = walk_repo(root=Path(repo.path))
        for file in repo_files:
            files += 1
            chunks = chunk_file(
                text=file.abs_path.read_text(encoding="utf-8", errors="replace"),
                path=file.rel_path,
                repo=repo.id,
                lang=file.lang,
                kind=file.kind,
            )
            chunks_count += len(chunks)
            texts = [chunk.text for chunk in chunks]
            vectors = embedder.embed(texts)
            written += store.upsert(dataset=repo.id, chunks=chunks, vectors=vectors)
    return IndexReport(
        files=files, chunks=chunks_count, written=written, missing_repos=tuple(missing_repos)
    )
