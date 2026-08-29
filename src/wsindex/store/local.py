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

from wsindex.embed import Embedder
from wsindex.model import Chunk, Hit
from wsindex.store.base import VectorStore

META_JSON = "meta.json"

VECTORS_NPY = "vectors.npy"

CHUNKS_JSON = "chunks.json"


class LocalStore(VectorStore):
    """Offline brute-force cosine backend; disk layout in the module docstring."""

    root: Path
    embedder: Embedder

    def __init__(self, root: Path, embedder: Embedder) -> None:
        """Wire the store to its disk location and embedding strategy.

        Args:
            root: Directory that holds one subdirectory per dataset;
                created lazily by `create_dataset`.
            embedder: Embeds chunk texts and queries; its `dim` defines
                the vector space of every dataset under this root.
        """
        self.root = root
        self.embedder = embedder

    def _dataset_dir(self, dataset: str) -> Path:
        return self.root / dataset

    def create_dataset(self, dataset_name: str, *, metric: str) -> None:
        """Ensure the dataset directory and its three files exist.

        Idempotent for the same embedder dim and metric.

        Args:
            dataset_name: Dataset to create; becomes a directory under `root`.
            metric: Similarity metric; this backend only supports "cosine".

        Raises:
            ValueError: The metric is not "cosine", or the dataset already
                exists with a different dim or metric.
        """
        dataset_dir = self._dataset_dir(dataset_name)
        if metric != "cosine":
            raise ValueError("metric must be 'cosine'")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if (dataset_dir / META_JSON).exists():
            exists_meta = json.loads((dataset_dir / META_JSON).read_text())
            if exists_meta["dim"] == self.embedder.dim and exists_meta["metric"] == metric:
                return
            else:
                raise ValueError(
                    f"Dataset {dataset_name} already exists with "
                    f"dim {exists_meta['dim']} "
                    f"and metric {exists_meta['metric']}  "
                )
        meta = {
            "dim": self.embedder.dim,
            "metric": metric,
        }
        (dataset_dir / META_JSON).write_text(json.dumps(meta), encoding="utf-8")
        vectors = np.empty((0, self.embedder.dim), dtype=np.float32)
        np.save(dataset_dir / VECTORS_NPY, vectors)
        (dataset_dir / CHUNKS_JSON).write_text(json.dumps([]), encoding="utf-8")

    def add_chunks(self, dataset_name: str, *, chunks: Sequence[Chunk]) -> int:
        """Embed and append chunks that are not stored yet.

        Dedup key is the deterministic chunk id (also within one batch),
        and dedup runs BEFORE embedding, so a re-index embeds nothing.

        Args:
            dataset_name: Dataset to write into.
            chunks: Candidate chunks; already-stored ones are skipped by id.

        Returns:
            How many chunks were actually written.

        Raises:
            FileNotFoundError: The dataset was never created.
        """
        dataset_dir = self._dataset_dir(dataset_name)
        vector_path = dataset_dir / VECTORS_NPY
        m = np.load(vector_path)
        records = json.loads((dataset_dir / CHUNKS_JSON).read_text())
        new_chunks = []
        known = {rec["id"] for rec in records}
        for chunk in chunks:
            if chunk.id in known:
                continue
            known.add(chunk.id)
            new_chunks.append(chunk)
        if not new_chunks:
            return 0
        vectors = self.embedder.embed([chunk.text for chunk in new_chunks])
        arr = np.vstack([m, np.asarray(vectors, dtype=np.float32)])
        records += [chunk.to_metadata() for chunk in new_chunks]
        np.save(dataset_dir / VECTORS_NPY, np.array(arr, dtype=np.float32))
        (dataset_dir / CHUNKS_JSON).write_text(json.dumps(records), encoding="utf-8")
        return len(vectors)

    def search(self, dataset_name: str, *, query: str, k: int) -> list[Hit]:
        """Brute-force cosine top-k over one dataset.

        Args:
            dataset_name: Dataset to search in.
            query: Query text; embedded locally with the injected embedder.
            k: Maximum number of hits to return.

        Returns:
            At most k hits, best score first; empty for an empty dataset.

        Raises:
            ValueError: The dataset does not exist, or the on-disk index was
                built with a different embedder dim (model changed without
                re-indexing) — failing loudly beats returning garbage scores.
        """
        dataset_dir = self._dataset_dir(dataset_name)
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} does not exist")
        exists_meta = json.loads((dataset_dir / META_JSON).read_text())
        vector_path = dataset_dir / VECTORS_NPY
        m = np.load(vector_path)
        if len(m) == 0:
            return []
        vector = self.embedder.embed([query])[0]
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
