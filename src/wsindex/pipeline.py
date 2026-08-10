"""Indexing and search pipeline: repos from the config in, Hits out.

The pipeline sees only the VectorStore and Embedder contracts; concrete
backends are constructed once at the edge (the CLI composition root) and
injected through the Pipeline constructor.
"""

from dataclasses import dataclass
from pathlib import Path

from wsindex.config import Config
from wsindex.embed.embedder import Embedder
from wsindex.ingest.chunker import chunk_file
from wsindex.ingest.walker import walk_repo
from wsindex.model import Hit
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


@dataclass(frozen=True)
class Pipeline:
    """The wired system: config + store + embedder, assembled once.

    Frozen on purpose: a Pipeline is a bundle of dependencies, not state —
    nothing may accumulate between calls. The composition root (the place
    that turns config values into concrete backends) lives with the CLI.
    """

    config: Config
    store: VectorStore
    embedder: Embedder

    def index(self) -> IndexReport:
        """Index every repo from the config into its own dataset (= repo id).

        Decisions fixed here: files are read with errors="replace" so a
        stray non-UTF-8 file cannot abort the run; a repo whose directory
        does not exist goes to `missing_repos` and is skipped (an existing
        repo with zero indexable files is NOT missing). Embedding is
        batched per file.
        """
        files = 0
        chunks_count = 0
        written = 0
        missing_repos: list[str] = []
        for repo in self.config.repos:
            if not Path(repo.path).is_dir():
                missing_repos.append(repo.id)
                continue
            self.store.create(dataset=repo.id, dim=self.embedder.dim, metric=self.config.metric)
            for file in walk_repo(root=Path(repo.path)):
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
                vectors = self.embedder.embed(texts)
                written += self.store.upsert(dataset=repo.id, chunks=chunks, vectors=vectors)
        return IndexReport(
            files=files, chunks=chunks_count, written=written, missing_repos=tuple(missing_repos)
        )

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Global top-k across all config repos, best score first.

        Merge policy lives here and only here: every dataset is asked for
        k hits (the global top may sit entirely in one repo, so asking for
        less is wrong), then one stable sort merges and cuts to k — on
        equal scores the config repo order wins, which keeps results
        deterministic. A repo that was never indexed (store raises
        ValueError) silently contributes zero hits: not yet indexed is a
        normal state, not an error.
        """
        vector = self.embedder.embed([query])[0]
        all_hits: list[Hit] = []
        for repo in self.config.repos:
            try:
                hits = self.store.search(repo.id, vector, k)
            except ValueError:
                continue
            all_hits.extend(hits)
        return sorted(all_hits, key=lambda h: h.score, reverse=True)[:k]
