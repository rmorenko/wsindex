"""LanceDB-backed VectorStore: one workspace table with a `repo` column (ADR-7).

Layout inside the LanceDB database the `uri` points to (a local directory
or an s3:// prefix; S3 credentials come from the environment, never from
the config):

    data      - all chunks of the workspace; the service `dataset` column
                holds the dataset name (Chunk.repo stays untouched
                payload), metadata fields are real columns so filters
                can prefilter
    datasets  - the dataset registry (name + metric): LanceDB has no
                notion of "dataset", the contract's created/not-created
                distinction lives here

The `VectorStore` contract is unchanged: `dataset_name` maps onto the
`dataset` column, so the pipeline never learns there is only one table.
"""

from collections.abc import Sequence
from typing import Any, cast

import lancedb
import pyarrow as pa
from lancedb import DBConnection
from lancedb.query import LanceVectorQueryBuilder
from lancedb.table import Table

from wsindex.embed.embedder import Embedder
from wsindex.model import Chunk, Hit
from wsindex.store.base import VectorStore


class LanceDBStore(VectorStore):
    """Embedded LanceDB backend; disk layout in the module docstring."""

    def __init__(
        self,
        uri: str,
        *,
        embedder: Embedder,
    ) -> None:
        """Connect to the database and ensure both tables exist.

        Connecting is eager (LanceDB lists table manifests immediately),
        so a wrong uri or missing S3 environment fails here, not in the
        middle of an indexing run.

        Args:
            uri: Database location — a local path or `s3://bucket/prefix`;
                for s3 the endpoint and credentials come from standard
                `AWS_*` environment variables.
            embedder: Embeds chunk texts and queries; its `dim` is baked
                into the vector column, so switching the model requires
                re-indexing.
        """
        self.embedder = embedder
        self.schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), self.embedder.dim)),
                # Service column: dataset membership is the store's own
                # bookkeeping — Chunk.repo stays untouched payload.
                pa.field("dataset", pa.string()),
                pa.field("id", pa.string()),
                pa.field("repo", pa.string()),
                pa.field("path", pa.string()),
                pa.field("lang", pa.string()),
                pa.field("kind", pa.string()),
                pa.field("symbol", pa.string()),
                pa.field("node_type", pa.string()),
                pa.field("start_line", pa.int32()),
                pa.field("end_line", pa.int32()),
                pa.field("text", pa.string()),
            ]
        )
        self.db: DBConnection = lancedb.connect(uri)
        self.tbl: Table = self.db.create_table("data", schema=self.schema, exist_ok=True)
        datasets_schema = pa.schema(
            [
                pa.field("repo", pa.string()),
                pa.field("metric", pa.string()),
            ]
        )
        self.dataset_table: Table = self.db.create_table(
            "datasets", schema=datasets_schema, exist_ok=True
        )
        self._known_datasets: dict[str, dict[str, Any]] = {}

    def _get_datasets(self) -> dict[str, dict[str, Any]]:
        """Registry rows by dataset name; one scan per store instance.

        The cache is updated locally after writes — the store lives for
        a single CLI invocation, so it cannot grow stale.
        """
        if not self._known_datasets:
            for d in self.dataset_table.search().to_list():
                self._known_datasets[d["repo"]] = d
        return self._known_datasets

    def create_dataset(self, dataset_name: str, *, metric: str) -> None:
        """Register the dataset in the registry table; a no-op if known.

        LanceDB binds the metric at query time, not at table creation,
        so the metric guard is entirely ours.

        Args:
            dataset_name: Dataset to create; becomes a `repo` column value.
            metric: Similarity metric; this backend only supports "cosine".

        Raises:
            ValueError: The metric is not "cosine".
        """
        if metric != "cosine":
            raise ValueError("Metric must be 'cosine'")
        self._get_datasets()
        if self._known_datasets.get(dataset_name) is not None:
            return
        dataset = {"repo": dataset_name, "metric": metric}
        self.dataset_table.add([dataset])
        self._known_datasets[dataset_name] = dataset

    def add_chunks(self, dataset_name: str, *, chunks: Sequence[Chunk]) -> int:
        """Embed and store chunks that are new to the dataset.

        Dedup key is the deterministic chunk id (also within one batch),
        and dedup runs BEFORE embedding, so a re-index embeds nothing.
        All new chunks are written as one batch — one Lance commit.

        Args:
            dataset_name: Dataset to write into; must be registered
                (see `create_dataset`).
            chunks: Candidate chunks; already-stored ones are skipped by id.

        Returns:
            How many chunks were actually written.

        Raises:
            ValueError: The dataset was never created.
        """
        if self._get_datasets().get(dataset_name) is None:
            raise ValueError("Dataset is not present in the store")
        predicate = "dataset = '" + dataset_name.replace("'", "''") + "'"
        known = {r["id"] for r in self.tbl.search().where(predicate).select(["id"]).to_list()}
        new_chunks = []
        for chunk in chunks:
            if chunk.id in known:
                continue
            known.add(chunk.id)
            new_chunks.append(chunk)
        if not new_chunks:
            return 0
        vectors = self.embedder.embed([chunk.text for chunk in new_chunks])
        rows = []
        for chunk, vec in zip(new_chunks, vectors, strict=True):
            row = chunk.to_metadata()
            row["vector"] = vec
            row["dataset"] = dataset_name
            rows.append(row)
        self.tbl.add(rows)
        return len(rows)

    def search(self, dataset_name: str, *, query: str, k: int) -> list[Hit]:
        """Exact cosine top-k over one dataset via a prefiltered KNN.

        The `repo` predicate is applied BEFORE the vector search
        (prefilter), so the top-k is computed over the dataset subset —
        a post-filter would silently under-fill the result.

        Args:
            dataset_name: Dataset to search in.
            query: Query text; embedded locally with the injected embedder.
            k: Maximum number of hits to return.

        Returns:
            At most k hits, best score first; empty for an empty dataset.

        Raises:
            ValueError: The dataset was never created — a normal state,
                the pipeline skips such datasets silently.
        """
        if self._get_datasets().get(dataset_name) is None:
            raise ValueError("Dataset is not present in the store")
        predicate = "dataset = '" + dataset_name.replace("'", "''") + "'"
        vec = self.embedder.embed([query])[0]
        builder = cast("LanceVectorQueryBuilder", self.tbl.search(vec))
        rows = builder.where(predicate, prefilter=True).distance_type("cosine").limit(k).to_list()
        return [
            Hit(
                score=1 - row["_distance"],
                native_id=row["id"],
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in ("vector", "_distance", "dataset")
                },
            )
            for row in rows
        ]
