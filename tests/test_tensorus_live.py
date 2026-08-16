"""Integration test against a live tensorus server (docker compose up).

Excluded from the default run by the `live` marker; run explicitly with
`uv run pytest -m live`. Needs TENSORUS_API_KEY in the environment —
skips itself otherwise, so a missing key never fails a run.
"""

import os
import uuid

import pytest

from wsindex.model import Chunk, Kind
from wsindex.store.tensorus import TensorusStore

pytestmark = pytest.mark.live

BASE_URL = os.environ.get("TENSORUS_BASE_URL", "http://localhost:8000")


def make_chunk(text: str) -> Chunk:
    return Chunk(
        repo="live",
        path="probe.py",
        lang="python",
        kind=Kind.CODE,
        symbol=None,
        node_type=None,
        start_line=1,
        end_line=1,
        text=text,
    )


def test_create_add_search_round_trip() -> None:
    api_key = os.environ.get("TENSORUS_API_KEY")
    if not api_key:
        pytest.skip("TENSORUS_API_KEY is not set")
    dataset = f"wsindex_live_{uuid.uuid4().hex[:8]}"
    store = TensorusStore(
        base_url=BASE_URL,
        api_key=api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        timeout=120.0,  # the first embed call may download the model
    )
    try:
        store.create_dataset(dataset, metric="cosine")
        store.create_dataset(dataset, metric="cosine")  # idempotency against the real server
        chunks = [
            make_chunk("def cosine(a, b): return dot(a, b)"),
            make_chunk("walk the repository tree and skip binaries"),
        ]
        assert store.add_chunks(dataset, chunks=chunks) == 2
        assert store.add_chunks(dataset, chunks=chunks) == 0  # dedup round trip

        hits = store.search(dataset, query="where is cosine similarity computed", k=2)
        assert hits
        assert hits[0].metadata["text"] == "def cosine(a, b): return dot(a, b)"
        assert hits[0].score >= hits[-1].score
    finally:
        store.client.delete(f"/datasets/{dataset}")
        store.close()
