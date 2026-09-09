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
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import lancedb
import pyarrow as pa
from lancedb import DBConnection
from lancedb.query import LanceVectorQueryBuilder
from lancedb.table import Table

from wsindex.embed.embedder import Embedder
from wsindex.model import Chunk, Hit, SearchFilter
from wsindex.store.base import CompactReport, VectorStore


def _sql_quote(value: str) -> str:
    """Escape a value for embedding into a single-quoted SQL literal."""
    return value.replace("'", "''")


def _glob_to_like(glob: str) -> str:
    """Convert an fnmatch-style glob to a SQL LIKE pattern with `\\` escape.

    Both `*` and `**` map to `%` — DataFusion LIKE has no depth
    distinction, so `src/*.py` matches `src/a/b.py` too. Literal `%`,
    `_`, `\\` in the input are escaped, and the caller pairs the pattern
    with `ESCAPE '\\'` in the LIKE clause.
    """
    out: list[str] = []
    for ch in glob:
        if ch in ("%", "_", "\\"):
            out.append("\\" + ch)
        elif ch == "*":
            out.append("%")
        elif ch == "?":
            out.append("_")
        else:
            out.append(ch)
    return "".join(out)


def _filter_predicates(filters: SearchFilter) -> list[str]:
    """Turn SearchFilter fields into a list of AND-joinable SQL predicates.

    Empty tuples / None fields contribute no predicate. Lang and kind
    become IN clauses (OR within the field); path and symbol become
    LIKE clauses with `\\` as the escape character.
    """
    parts: list[str] = []
    if filters.lang:
        joined = ", ".join(f"'{_sql_quote(v)}'" for v in filters.lang)
        parts.append(f"lang IN ({joined})")
    if filters.kind:
        joined = ", ".join(f"'{_sql_quote(v.value)}'" for v in filters.kind)
        parts.append(f"kind IN ({joined})")
    if filters.path is not None:
        parts.append(f"path LIKE '{_sql_quote(_glob_to_like(filters.path))}' ESCAPE '\\'")
    if filters.symbol is not None:
        pattern = "%" + _sql_quote(_glob_to_like(filters.symbol)) + "%"
        parts.append(f"symbol LIKE '{pattern}' ESCAPE '\\'")
    return parts


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

        The cache is updated locally after writes. It used to say it
        could not grow stale because a store lived for one CLI
        invocation — true until Этап 11 gave the store a process that
        outlives the question, at which point a repo registered by
        somebody else stayed invisible here forever. `refresh` drops it.
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
        predicate = f"dataset = '{_sql_quote(dataset_name)}'"
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

    def search(
        self,
        dataset_name: str,
        *,
        query: str,
        k: int,
        filters: SearchFilter | None = None,
    ) -> list[Hit]:
        """Exact cosine top-k over one dataset via a prefiltered KNN.

        Both the `dataset` predicate and any user filters are applied
        BEFORE the vector search (prefilter=True) and joined with AND, so
        the top-k is computed over the fully filtered subset — a
        postfilter over an unfiltered top-k would silently under-fill.

        Args:
            dataset_name: Dataset to search in.
            query: Query text; embedded locally with the injected embedder.
            k: Maximum number of hits to return.
            filters: Structural filters composed into the same WHERE.

        Returns:
            At most k hits, best score first; empty for an empty dataset.

        Raises:
            ValueError: The dataset was never created — a normal state,
                the pipeline skips such datasets silently.
        """
        if self._get_datasets().get(dataset_name) is None:
            raise ValueError("Dataset is not present in the store")
        parts = [f"dataset = '{_sql_quote(dataset_name)}'"]
        if filters is not None and not filters.is_empty:
            parts.extend(_filter_predicates(filters))
        predicate = " AND ".join(parts)
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

    def chunk_ids(self, dataset_name: str, *, paths: Sequence[str] | None = None) -> set[str]:
        """Ids stored for the given paths, scoped to one dataset.

        `IN (...)` rather than OR-joined equalities for the path list:
        DataFusion turns an `InList` into a hash set at plan time, while
        a chain of ORs stays a BooleanOr tree evaluated per row (the
        step-20 probe measured 3.8x on 1000 terms).

        The dataset predicate is required for the same reason as in
        `delete_chunks`: one physical table holds every repo (ADR-7), and
        `chunk_id = sha256(text, path)` carries no repo, so two repos
        with the same file share an id.

        Args:
            dataset_name: Dataset to read from; must be registered.
            paths: Repo-relative POSIX paths, or None for the whole dataset.

        Returns:
            The stored chunk ids.

        Raises:
            TypeError: `paths` is a bare string instead of a batch.
            ValueError: The dataset was never created.
        """
        if isinstance(paths, str):
            raise TypeError("expected a batch of paths, got a single str")
        if self._get_datasets().get(dataset_name) is None:
            raise ValueError("Dataset is not present in the store")
        if paths is not None and not paths:
            # Asking about no paths is not asking about all of them; the
            # `None` default is the only way to say "everything".
            return set()
        predicate = f"dataset = '{_sql_quote(dataset_name)}'"
        if paths is not None:
            path_list = ", ".join(f"'{_sql_quote(p)}'" for p in paths)
            predicate += f" AND path IN ({path_list})"
        rows = self.tbl.search().where(predicate).select(["id"]).to_list()
        return {row["id"] for row in rows}

    def delete_chunks(self, dataset_name: str, *, ids: Sequence[str]) -> int:
        """Delete chunks by id, scoped to one dataset.

        The dataset predicate is required, not decorative: the workspace
        keeps every repo's chunks in one physical table (ADR-7), and a
        bare `id IN (...)` would silently wipe matching rows across every
        dataset. The risk is real because `Chunk.chunk_id = sha256(text, path)`
        does not include repo, so two repos with the same file share the
        same id. Both the dataset name and each id are `_sql_quote`-escaped
        before being embedded into the WHERE clause.

        Args:
            dataset_name: Dataset to delete from; must be registered
                (see `create_dataset`).
            ids: Chunk ids to remove; missing ids are silently skipped.

        Returns:
            How many rows the delete actually removed (may be below
            `len(ids)` when some were not there).

        Raises:
            TypeError: `ids` is a bare string instead of a batch.
            ValueError: The dataset was never created.
        """
        if isinstance(ids, str):
            raise TypeError("expected a batch of ids, got a single str")
        if self._get_datasets().get(dataset_name) is None:
            raise ValueError("Dataset is not present in the store")
        if not ids:
            return 0
        id_list = ", ".join(f"'{_sql_quote(x)}'" for x in ids)
        predicate = f"dataset = '{_sql_quote(dataset_name)}' AND id IN ({id_list})"
        result = self.tbl.delete(predicate)
        # LanceDB's DeleteResult carries num_deleted_rows at runtime (see
        # probes/step20/probe_delete.py), but the field is missing from
        # the stubs as of 0.21+; the cast + ignore is self-cleaning via
        # `warn_unused_ignores` when the stubs catch up.
        return cast("int", result.num_deleted_rows)  # type: ignore[attr-defined]

    def chunk_text(self, dataset_name: str, *, ids: Sequence[str]) -> dict[str, str]:
        """Text of the given chunks, scoped to one dataset.

        Same `IN (...)` and same dataset predicate as `chunk_ids`, for
        the same two reasons: DataFusion turns an `InList` into a hash
        set at plan time, and one physical table holds every repo, so a
        bare `id IN (...)` would read across datasets (ADR-7).

        Args:
            dataset_name: Dataset to read from; must be registered.
            ids: Chunk ids to fetch.

        Returns:
            Chunk id -> text, for the ids that were found.

        Raises:
            TypeError: `ids` is a bare string instead of a batch.
            ValueError: The dataset was never created.
        """
        if isinstance(ids, str):
            raise TypeError("expected a batch of ids, got a single str")
        if self._get_datasets().get(dataset_name) is None:
            raise ValueError("Dataset is not present in the store")
        if not ids:
            return {}
        id_list = ", ".join(f"'{_sql_quote(x)}'" for x in ids)
        predicate = f"dataset = '{_sql_quote(dataset_name)}' AND id IN ({id_list})"
        rows = self.tbl.search().where(predicate).select(["id", "text"]).to_list()
        return {row["id"]: row["text"] for row in rows}

    def _tables(self) -> tuple[Table, Table]:
        """Every physical table this store owns; both need housekeeping."""
        return (self.tbl, self.dataset_table)

    def _versions(self) -> int:
        """Total historical versions across the store's tables."""
        return sum(len(table.list_versions()) for table in self._tables())

    def _on_disk_bytes(self) -> int | None:
        """Bytes the database occupies, or None if that cannot be measured.

        Walking the directory is the only honest measure. The per-version
        `total_files_size` that `list_versions` reports cannot be summed:
        Lance is copy-on-write and versions share fragments, so the total
        would count the same file once per version that references it.

        Returns None for a remote uri (`s3://...`), where listing objects
        would need a separate storage API this store does not carry.
        """
        uri = self.db.uri
        if "://" in uri:
            return None
        root = Path(uri)
        if not root.is_dir():
            return None
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

    def refresh(self) -> None:
        """Move both handles to the newest committed version.

        `checkout_latest` on each table. Both, because the dataset
        registry is written by `create_dataset` in whichever process ran
        it — a server that refreshed only the data table would keep
        answering "no such dataset" for a repo somebody else registered.
        """
        for table in self._tables():
            # Untyped in lancedb's stubs, like most of its surface.
            table.checkout_latest()  # type: ignore[no-untyped-call]
        # And the registry cache above it: moving the table handle
        # forward means nothing while a dict remembers the old answer.
        self._known_datasets.clear()

    def compact(self, *, older_than: timedelta = timedelta(0)) -> CompactReport:
        """Merge small files and drop old versions, on every table.

        Order matters inside LanceDB's `optimize`: compaction writes a
        NEW, merged version and the versions it replaces stay on disk, so
        compacting without pruning makes the directory *grow*. Measured
        in probes/step22v: 20 append batches then 150 deletes left 273 KB,
        a bare `optimize()` took it to 317 KB, and only pruning brought it
        to 43 KB. Passing `cleanup_older_than` is therefore not a tuning
        knob here — it is the half that does the reclaiming.

        `delete_unverified` is left at its default. Files from a failed
        transaction are only removed once they are a week old, which is
        what keeps this safe to run while another process might be
        mid-write; overriding it can corrupt the dataset.

        Args:
            older_than: Keep versions younger than this; default keeps none.

        Returns:
            Sizes and version counts either side of the pass.
        """
        bytes_before = self._on_disk_bytes()
        versions_before = self._versions()
        for table in self._tables():
            table.optimize(cleanup_older_than=older_than)
        return CompactReport(
            bytes_before=bytes_before,
            bytes_after=self._on_disk_bytes(),
            versions_before=versions_before,
            versions_after=self._versions(),
        )
