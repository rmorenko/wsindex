"""Indexing and search pipeline: repos in, Hits out.

The pipeline sees only the VectorStore contract and works in plain text —
embedding is the store's private business. The concrete backend is built
at the edge (the CLI composition root) and injected through the
constructor; which repositories to index and with which metric comes from
`Config()`, read at call time. One process serves one workspace, so
threading those two values through the composition root only to hand them
back unchanged was ceremony.
"""

from dataclasses import dataclass
from pathlib import Path

from wsindex.config import Config
from wsindex.ingest import chunk_file, walk_repo
from wsindex.model import Hit
from wsindex.store import VectorStore


@dataclass(frozen=True, kw_only=True)
class IndexReport:
    """Immutable summary of one index() run, totals across all repos.

    Attributes:
        files: How many files were walked and chunked.
        chunks: How many chunks the files produced.
        written: How many chunks the store actually wrote; below `chunks`
            means dedup skipped already-stored ones.
        missing_repos: Ids of configured repos whose directory does not
            exist; they were skipped, not failed on.
    """

    files: int
    chunks: int
    written: int
    missing_repos: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class Pipeline:
    """The wired system: a store, plus whatever the current Config says.

    Frozen on purpose: a Pipeline is a bundle of dependencies, not state —
    nothing may accumulate between calls. The repo list and the metric are
    read from `Config()` inside the methods rather than captured at
    construction, so a config that changed (a repo added, say) is picked
    up by the next call instead of going stale in a field.

    Attributes:
        store: Any VectorStore backend; the pipeline never looks behind
            the contract.
    """

    store: VectorStore

    def index(self) -> IndexReport:
        """Index every repo into its own dataset (dataset name = repo id).

        Decisions fixed here: files are read with errors="replace" so a
        stray non-UTF-8 file cannot abort the run; a repo whose directory
        does not exist goes to `missing_repos` and is skipped (an existing
        repo with zero indexable files is NOT missing).

        Returns:
            Totals across all repos; see IndexReport field docs.
        """
        config = Config()
        files = 0
        chunks_count = 0
        written = 0
        missing_repos: list[str] = []
        for repo in config.repos:
            if not Path(repo.path).is_dir():
                missing_repos.append(repo.id)
                continue
            self.store.create_dataset(dataset_name=repo.id, metric=config.metric)
            root = Path(repo.path)
            for file in walk_repo(root=root):
                files += 1
                chunks = chunk_file(
                    text=(root / file.rel_path).read_text(encoding="utf-8", errors="replace"),
                    path=file.rel_path,
                    repo=repo.id,
                    lang=file.lang,
                    kind=file.kind,
                )
                chunks_count += len(chunks)
                written += self.store.add_chunks(dataset_name=repo.id, chunks=chunks)
        return IndexReport(
            files=files, chunks=chunks_count, written=written, missing_repos=tuple(missing_repos)
        )

    def search(self, query: str, *, k: int = 10) -> list[Hit]:
        """Global top-k across all repos, best score first.

        Merge policy lives here and only here: every dataset is asked for
        k hits (the global top may sit entirely in one repo, so asking for
        less is wrong), then one stable sort merges and cuts to k — on
        equal scores the given repo order wins, which keeps results
        deterministic. A repo that was never indexed (store raises
        ValueError) silently contributes zero hits: not yet indexed is a
        normal state, not an error.

        Args:
            query: Query text; embedding is the store's business.
            k: Maximum number of hits in the merged result.

        Returns:
            At most k hits across all repos, best score first.
        """
        all_hits: list[Hit] = []
        for repo in Config().repos:
            try:
                hits = self.store.search(dataset_name=repo.id, query=query, k=k)
            except ValueError:
                continue
            all_hits.extend(hits)
        return sorted(all_hits, key=lambda h: h.score, reverse=True)[:k]
