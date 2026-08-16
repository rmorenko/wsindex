"""Tensorus-backed VectorStore: embedding and search happen server-side.

The store never sees vectors — chunks and queries travel as text and the
server embeds them with `model_name` (kept equal to the workspace model so
local and remote backends share one vector space). Accepted MVP debt:
add_chunks embeds one chunk per request (the embed endpoint takes a single
metadata object), known-id fetching is not paginated, and the server's
/index/build endpoint is broken upstream, so search runs brute-force.
"""

from collections.abc import Sequence
from typing import Any

import httpx

from wsindex.model import Chunk, Hit
from wsindex.store.base import VectorStore


class TensorusStore(VectorStore):
    """REST client for the Tensorus vector API (x-api-key auth)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        *,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model_name
        self.client = httpx.Client(
            base_url=base_url, headers={"x-api-key": api_key}, transport=transport, timeout=timeout
        )
        self._known: dict[str, set[str]] = {}

    def close(self) -> None:
        self.client.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Single place where network failures become human errors."""
        try:
            return self.client.request(method, url, **kwargs)
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"tensorus is unreachable at {self.client.base_url} — "
                "is it running? try `docker compose up -d`"
            ) from exc

    def create_dataset(self, dataset_name: str, *, metric: str) -> None:
        """Ensure the dataset exists; 409 means "already there" and is fine.

        `metric` is accepted for the contract but ignored: the server owns
        its similarity math.
        """
        response = self._request("POST", "/datasets/create", json={"name": dataset_name})
        if response.status_code not in (200, 201, 409):
            raise RuntimeError(f"dataset create failed: {response.status_code} {response.text}")

    def _known_ids(self, dataset: str) -> set[str]:
        """Ids of already-stored chunks; one GET per dataset per process.

        The cache is kept up to date locally after writes — the store lives
        for a single CLI invocation, so it cannot grow stale.
        """
        if dataset not in self._known:
            response = self._request(
                "GET", f"/datasets/{dataset}/records", params={"offset": 0, "limit": 100_000}
            )
            ids: set[str] = set()
            if response.is_success:
                for record in response.json().get("data", []):
                    chunk_id = record.get("metadata", {}).get("id")
                    if chunk_id is not None:
                        ids.add(chunk_id)
            self._known[dataset] = ids
        return self._known[dataset]

    def add_chunks(self, dataset_name: str, *, chunks: Sequence[Chunk]) -> int:
        """Embed server-side and store chunks that are not stored yet."""
        known = self._known_ids(dataset_name)
        written = 0
        for chunk in chunks:
            if chunk.id in known:
                continue
            response = self._request(
                "POST",
                "/api/v1/vector/embed",
                json={
                    "texts": [chunk.text],
                    "dataset_name": dataset_name,
                    "model_name": self.model_name,
                    "provider": "sentence-transformers",
                    "metadata": chunk.to_metadata(),
                },
            )
            if not response.is_success:
                raise RuntimeError(f"embed failed: {response.status_code} {response.text}")
            known.add(chunk.id)
            written += 1
        return written

    def search(self, dataset_name: str, *, query: str, k: int) -> list[Hit]:
        """Server-side semantic top-k; the server embeds the query itself."""
        response = self._request(
            "POST",
            "/api/v1/vector/search",
            json={"query": query, "dataset_name": dataset_name, "k": k},
        )
        if response.status_code == 404:
            # Contract: an unindexed dataset is a normal state (ValueError),
            # the pipeline skips it silently.
            raise ValueError(f"dataset {dataset_name!r} is not indexed yet")
        if not response.is_success:
            raise RuntimeError(f"search failed: {response.status_code} {response.text}")
        return [
            Hit(
                score=float(result["similarity_score"]),
                metadata=result.get("metadata", {}),
                native_id=result.get("record_id"),
            )
            for result in response.json()["results"]
        ]
