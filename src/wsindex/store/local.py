"""File-backed VectorStore: numpy matrix + JSON metadata per dataset.

Layout under the store root, one directory per dataset:

    <root>/<dataset>/meta.json     - {"dim": ..., "metric": "cosine"}
    <root>/<dataset>/vectors.npy   - float32 matrix of shape (n, dim)
    <root>/<dataset>/chunks.json   - list of chunk metadata dicts

Invariant: row i of vectors.npy embeds element i of chunks.json — the row
index is the only link between a vector and its chunk.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wsindex.model import Chunk, Hit
from wsindex.store.base import VectorStore

META_JSON = "meta.json"

VECTORS_NPY = "vectors.npy"

CHUNKS_JSON = "chunks.json"


class LocalStore(VectorStore):
    """Offline brute-force cosine backend; disk layout in the module docstring."""

    root: Path

    def __init__(self, root: Path) -> None:
        self.root = root

    def _dataset_dir(self, dataset: str) -> Path:
        return self.root / dataset

    def create(self, dataset: str, *, dim: int, metric: str) -> None:
        """Ensure the dataset directory and its three files exist.

        Idempotent for identical (dim, metric); ValueError when the dataset
        exists with different parameters or the metric is not "cosine".
        """
        dataset_dir = self._dataset_dir(dataset)
        if metric != "cosine":
            raise ValueError("metric must be 'cosine'")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if (dataset_dir / META_JSON).exists():
            exists_meta = json.loads((dataset_dir / META_JSON).read_text())
            if exists_meta["dim"] == dim and exists_meta["metric"] == metric:
                return
            else:
                raise ValueError(
                    f"Dataset {dataset} already exists with "
                    f"dim {exists_meta['dim']} "
                    f"and metric {exists_meta['metric']}  "
                )
        meta = {
            "dim": dim,
            "metric": metric,
        }
        (dataset_dir / META_JSON).write_text(json.dumps(meta), encoding="utf-8")
        vectors = np.empty((0, dim), dtype=np.float32)
        np.save(dataset_dir / VECTORS_NPY, vectors)
        (dataset_dir / CHUNKS_JSON).write_text(json.dumps([]), encoding="utf-8")

    def upsert(
        self, dataset: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        """Append chunks that are not stored yet; return how many were written.

        Dedup key is the deterministic chunk id (also within one batch).
        ValueError on a chunks/vectors length mismatch; a dataset that was
        never created surfaces as FileNotFoundError.
        """
        if len(chunks) != len(vectors):
            raise ValueError("vectors shape mismatch")
        dataset_dir = self._dataset_dir(dataset)
        vector_path = dataset_dir / VECTORS_NPY
        m = np.load(vector_path)
        records = json.loads((dataset_dir / CHUNKS_JSON).read_text())
        new_records = []
        new_vectors = []
        known = {rec["id"] for rec in records}
        for chunk, vec in zip(chunks, vectors, strict=True):
            if chunk.id in known:
                continue
            known.add(chunk.id)
            new_records.append(chunk.to_metadata())
            new_vectors.append(vec)
        if len(new_vectors) == 0:
            return 0
        arr = np.vstack([m, np.asarray(new_vectors, dtype=np.float32)])
        records = records + new_records
        np.save(dataset_dir / VECTORS_NPY, np.array(arr, dtype=np.float32))
        (dataset_dir / CHUNKS_JSON).write_text(json.dumps(records), encoding="utf-8")
        return len(new_vectors)

    def search(self, dataset: str, vector: Sequence[float], k: int) -> list[Hit]:
        """Brute-force cosine top-k over one dataset, best score first.

        ValueError for an unknown dataset or a query of the wrong dim; an
        empty dataset yields [].
        """
        dataset_dir = self._dataset_dir(dataset)
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset} does not exist")
        exists_meta = json.loads((dataset_dir / META_JSON).read_text())
        vector_path = dataset_dir / VECTORS_NPY
        m = np.load(vector_path)
        if len(m) == 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        if q.shape[0] != exists_meta["dim"]:
            raise ValueError("vector shape mismatch")
        norms = np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        q_norm = max(float(np.linalg.norm(q)), 1e-12)
        scores = (m / norms) @ (q / q_norm)
        idx = np.argsort(scores)[::-1][:k]
        records = json.loads((dataset_dir / CHUNKS_JSON).read_text())
        return [
            Hit(score=float(scores[i]), metadata=records[i], native_id=records[i]["id"])
            for i in idx
        ]
