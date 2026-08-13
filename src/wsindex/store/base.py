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
    """A searchable store of chunks, grouped into datasets (one dataset = one repo).

    The contract is text in, hits out: who and where embeds is an
    implementation detail — LocalStore embeds with an injected Embedder,
    TensorusStore delegates to the server.
    """

    @abstractmethod
    def create(self, dataset: str, *, metric: str) -> None:
        """Ensure the dataset exists; a no-op if it is already there.

        Idempotent on purpose: the pipeline calls it on every `index` run
        instead of tracking which datasets were created earlier.
        """

    @abstractmethod
    def upsert(self, dataset: str, chunks: Sequence[Chunk]) -> int:
        """Embed and store chunks; return how many were actually written.

        Chunks whose deterministic id is already stored are skipped, so a
        return value below `len(chunks)` is how incremental indexing
        reports its savings.
        """

    @abstractmethod
    def search(self, dataset: str, query: str, k: int) -> list[Hit]:
        """Return the k nearest chunks of one dataset, best score first.

        Single dataset on purpose: merging and re-ranking across datasets is
        pipeline policy (see Pipeline.search) and must live in exactly one
        place, not be reimplemented by every backend.
        """
