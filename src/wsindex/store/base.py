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
    def search(self, dataset_name: str, *, query: str, k: int) -> list[Hit]:
        """Find the chunks of one dataset nearest to a text query.

        Single dataset on purpose: merging and re-ranking across datasets is
        pipeline policy (see Pipeline.search) and must live in exactly one
        place, not be reimplemented by every backend.

        Args:
            dataset_name: Dataset to search in.
            query: Query text; the store embeds it into the dataset's
                vector space itself.
            k: Maximum number of hits to return.

        Returns:
            At most k hits, best score first; empty for an empty dataset.

        Raises:
            ValueError: The dataset was never indexed — a normal state,
                the pipeline skips such datasets silently.
        """
