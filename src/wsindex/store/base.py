"""Vector store contract: the interface every search backend implements.

The pipeline depends only on this ABC (ARCH §2, modifiability): LocalStore
(numpy fallback) and TensorusStore (REST) plug in behind it, selected by
`Config.backend`. An ABC on purpose — the set of implementations is closed,
and an incomplete store must fail loudly at construction time.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from wsindex.model import Chunk, Hit


class VectorStore(ABC):
    """A store of chunk vectors, grouped into datasets (one dataset = one repo)."""

    @abstractmethod
    def create(self, dataset: str, *, dim: int, metric: str) -> None:
        """Ensure the dataset exists; a no-op if it is already there.

        Idempotent on purpose: the pipeline calls it on every `index` run
        instead of tracking which datasets were created earlier.
        """

    @abstractmethod
    def upsert(
        self, dataset: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        """Store chunks with their vectors; return how many were actually written.

        `vectors[i]` is the embedding of `chunks[i]`; implementations raise
        ValueError on a length mismatch. Chunks whose deterministic id is
        already stored are skipped, so a return value below `len(chunks)` is
        how incremental indexing reports its savings.
        """

    @abstractmethod
    def search(self, dataset: str, vector: Sequence[float], k: int) -> list[Hit]:
        """Return the k nearest chunks of one dataset, best score first.

        Single dataset on purpose: merging and re-ranking across datasets is
        pipeline policy (see Pipeline.search) and must live in exactly one
        place, not be reimplemented by every backend.
        """
