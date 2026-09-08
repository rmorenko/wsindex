"""Vector store contract: the interface every search backend implements.

The pipeline depends only on this ABC (ARCH §2, modifiability): LanceDBStore
(embedded, ADR-7) plugs in behind it, selected by `Config.backend` — the
enum keeps a single member so a future backend is a data change.
An ABC on purpose — the set of implementations is closed, and an
incomplete store must fail loudly at construction time.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from wsindex.model import Chunk, Hit, SearchFilter


class VectorStore(ABC):
    """A searchable store of chunks, grouped into datasets (one dataset = one repo).

    The contract is text in, hits out: who and where embeds is an
    implementation detail — LanceDBStore embeds with an injected Embedder.
    """

    @abstractmethod
    def create_dataset(self, dataset_name: str, *, metric: str) -> None:
        """Ensure the dataset exists; a no-op if it is already there.

        Idempotent on purpose: the pipeline calls it on every `index` run
        instead of tracking which datasets were created earlier.

        Args:
            dataset_name: Dataset to create; datasets are named after repo ids.
            metric: Similarity metric for the dataset's vector space.

        Raises:
            ValueError: The metric is unsupported, or the dataset already
                exists with incompatible parameters.
        """

    @abstractmethod
    def add_chunks(self, dataset_name: str, *, chunks: Sequence[Chunk]) -> int:
        """Embed and store chunks that are new to the dataset.

        The store owns embedding — callers never see vectors. Chunks whose
        deterministic id is already stored are skipped, so a return value
        below `len(chunks)` is how incremental indexing reports its savings.

        Args:
            dataset_name: Dataset to write into; must already exist
                (see `create_dataset`).
            chunks: Candidate chunks; already-stored ones are skipped by id.

        Returns:
            How many chunks were actually written.
        """

    @abstractmethod
    def search(
        self,
        dataset_name: str,
        *,
        query: str,
        k: int,
        filters: SearchFilter | None = None,
    ) -> list[Hit]:
        """Find the chunks of one dataset nearest to a text query.

        Single dataset on purpose: merging and re-ranking across datasets is
        pipeline policy (see Pipeline.search) and must live in exactly one
        place, not be reimplemented by every backend.

        Args:
            dataset_name: Dataset to search in.
            query: Query text; the store embeds it into the dataset's
                vector space itself.
            k: Maximum number of hits to return.
            filters: Structural filters (lang/kind/path/symbol) applied
                as a PREfilter — top-k is computed over the filtered
                subset, not slashed out of an unfiltered top-k.

        Returns:
            At most k hits, best score first; empty for an empty dataset.

        Raises:
            ValueError: The dataset was never indexed — a normal state,
                the pipeline skips such datasets silently.
        """

    @abstractmethod
    def chunk_ids(self, dataset_name: str, *, paths: Sequence[str] | None = None) -> set[str]:
        """Ids of the chunks currently stored for the given source paths.

        The read half of incremental indexing. `delete_chunks` can forget
        chunks by id, but the pipeline only learns *which* ids to forget
        by comparing what is stored against what re-chunking just
        produced — this is that comparison's left-hand side.

        `paths=None` means the whole dataset, which is what a full pass
        needs to reconcile: every stored id the current working tree no
        longer produces is stale, whatever file it came from.

        Args:
            dataset_name: Dataset to read from.
            paths: Restrict to chunks whose `path` is one of these
                (repo-relative, POSIX). None means every path. An empty
                sequence means no path, so the result is empty — asking
                about nothing is not the same as asking about everything.

        Returns:
            The stored chunk ids; empty when nothing matches.

        Raises:
            TypeError: `paths` is a bare string instead of a batch.
            ValueError: The dataset was never created.
        """

    @abstractmethod
    def delete_chunks(self, dataset_name: str, *, ids: Sequence[str]) -> int:
        """Delete chunks from a dataset.

        Deleting an id that is not in the dataset is a no-op; the method just returns 0.

        On-disk reclaim is NOT part of this contract — a backend may keep
        deleted rows physically until a separate housekeeping pass; the
        row is invisible to `search` immediately, but the storage footprint
        can drift up until that pass runs.

        Args:
            dataset_name: Dataset to delete from.
            ids: ids of chunks to delete.

        Returns:
            Number of deleted chunks.

        Raises:
            TypeError: when ids is bare string
            ValueError: when unknown dataset
        """
