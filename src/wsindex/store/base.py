"""Vector store contract: the interface every search backend implements.

The pipeline depends only on this ABC (ARCH §2, modifiability): LanceDBStore
(embedded, ADR-7) plugs in behind it, selected by `Config.backend` — the
enum keeps a single member so a future backend is a data change.
An ABC on purpose — the set of implementations is closed, and an
incomplete store must fail loudly at construction time.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from wsindex.model import Chunk, Hit, SearchFilter


@dataclass(frozen=True, kw_only=True)
class CompactReport:
    """What one housekeeping pass reclaimed.

    Sizes are optional because not every store can measure itself: a
    local directory can be walked, an `s3://` prefix would need a
    separate listing API. Reporting None is the honest answer — better
    than a zero that reads like "nothing was freed".

    Attributes:
        bytes_before: On-disk size before the pass, or None if the store
            cannot measure it.
        bytes_after: Same, after.
        versions_before: How many historical versions the store held.
        versions_after: How many it holds now.
    """

    bytes_before: int | None
    bytes_after: int | None
    versions_before: int
    versions_after: int

    @property
    def bytes_freed(self) -> int | None:
        """Space reclaimed, or None when the store could not measure."""
        if self.bytes_before is None or self.bytes_after is None:
            return None
        return self.bytes_before - self.bytes_after


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
        deleted rows physically until a separate housekeeping pass (see
        `compact`); the row is invisible to `search` immediately, but the
        storage footprint can drift up until that pass runs.

        Args:
            dataset_name: Dataset to delete from.
            ids: ids of chunks to delete.

        Returns:
            Number of deleted chunks.

        Raises:
            TypeError: when ids is bare string
            ValueError: when unknown dataset
        """

    @abstractmethod
    def chunk_text(self, dataset_name: str, *, ids: Sequence[str]) -> dict[str, str]:
        """The stored text of the given chunks, by id.

        The third read the contract needs, after `search` (find by
        meaning) and `chunk_ids` (what is stored for these paths): fetch
        exactly these, because something else already decided which. A
        blame edge names a commit's chunk id, and `wsindex why` has to
        turn that into the message a person reads.

        Args:
            dataset_name: Dataset to read from.
            ids: Chunk ids to fetch. Ids that are not stored are simply
                absent from the result — asking about a chunk that was
                re-indexed away is normal, not an error.

        Returns:
            Chunk id -> text, for the ids that were found.

        Raises:
            TypeError: `ids` is a bare string instead of a batch.
            ValueError: The dataset was never created.
        """

    @abstractmethod
    def refresh(self) -> None:
        """See what other processes have written since this store opened.

        A store may hold a snapshot: LanceDBStore does, and what that
        means was measured — a handle polled 40 times over 0.8 s while
        another process committed saw its opening version every time.
        Reads do not refresh themselves.

        Harmless in a CLI, where the process is younger than the
        question. A correctness bug in anything long-lived: after an
        outside `index`, a server would answer from the corpus as it was
        when the server started, and never fail while doing it.

        On the contract rather than in the server because staleness
        belongs to long-lived processes, not to HTTP — the MCP adapter
        and any library caller have the same problem (ADR-10).
        `Pipeline.search` calls this, so no caller has to remember.

        Cheap by design: 4 ms against a 111 ms search on a local path.
        A backend without snapshot semantics implements it as a no-op.
        """

    @abstractmethod
    def compact(self, *, older_than: timedelta = timedelta(0)) -> CompactReport:
        """Reclaim the disk that deleted and rewritten chunks still occupy.

        The other half of the promise `delete_chunks` makes above: a
        delete hides a row immediately but need not free its bytes, and
        now that incremental indexing deletes on every run, "need not"
        adds up. Store-wide rather than per dataset — reclaim is a
        property of the physical storage, and a backend is free to keep
        every dataset in one place (LanceDBStore does, ADR-7).

        Deliberately not called by `index`. Housekeeping is the user's
        decision because it is the one operation here that discards
        history: until it runs, a store keeps its old versions and can be
        rolled back; afterwards it cannot.

        Args:
            older_than: Keep versions younger than this. The default
                keeps none, which is what a user asking to reclaim space
                means. Raise it above zero when something else may be
                reading the same store — a search that started before
                this pass would be reading a version it removes. The
                current version is never touched.

        Returns:
            Sizes and version counts either side of the pass.
        """
