"""Tests for TensorusStore: request shapes, error translation, dedup cache.

All HTTP goes through httpx.MockTransport — a fake server in a function.
Canned responses copy the shapes probed on a live tensorus instance
(step 16, B1/B3.0 probes), so the mocks test the real API, not a guess.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from wsindex.model import Chunk, Kind
from wsindex.store.tensorus import TensorusStore

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_chunk(text: str, path: str = "a.py") -> Chunk:
    return Chunk(
        repo="r",
        path=path,
        lang="python",
        kind=Kind.CODE,
        symbol=None,
        node_type=None,
        start_line=1,
        end_line=1,
        text=text,
    )


def make_store(handler: Callable[[httpx.Request], httpx.Response]) -> TensorusStore:
    return TensorusStore(
        base_url="http://tensorus.test",
        api_key="k3y",
        model_name=MODEL,
        transport=httpx.MockTransport(handler),
    )


def test_api_key_header_travels_with_every_request() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["x-api-key"])
        if request.url.path == "/datasets/create":
            return httpx.Response(201, json={"success": True, "data": None})
        if request.url.path.endswith("/records"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"success": True, "record_ids": ["r1"]})

    store = make_store(handler)
    store.create_dataset("ds", metric="cosine")
    store.add_chunks("ds", chunks=[make_chunk("hello")])
    assert len(seen) == 3  # create + records + embed
    assert all(key == "k3y" for key in seen)


def test_create_swallows_409_for_idempotency() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "Dataset 'ds' already exists."})

    make_store(handler).create_dataset("ds", metric="cosine")  # must not raise


def test_create_other_errors_are_loud() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Internal server error"})

    with pytest.raises(RuntimeError, match="dataset create failed"):
        make_store(handler).create_dataset("ds", metric="cosine")


def test_add_chunks_skips_known_ids_and_reports_writes() -> None:
    a, b = make_chunk("alpha"), make_chunk("beta")
    records_calls = 0
    embedded: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal records_calls
        if request.url.path == "/datasets/ds/records":
            records_calls += 1
            # Shape from the live probe: our chunk id sits in metadata["id"].
            return httpx.Response(
                200, json={"data": [{"record_id": "r0", "metadata": {"id": a.id}}]}
            )
        if request.url.path == "/api/v1/vector/embed":
            embedded.append(json.loads(request.content))
            return httpx.Response(200, json={"success": True, "record_ids": ["r1"]})
        raise AssertionError(f"unexpected path: {request.url.path}")

    store = make_store(handler)
    assert store.add_chunks("ds", chunks=[a, b]) == 1  # a is already stored server-side
    assert store.add_chunks("ds", chunks=[a, b]) == 0  # b is now in the local cache
    assert records_calls == 1  # known ids fetched once per dataset, then cached
    assert [payload["texts"] for payload in embedded] == [["beta"]]


def test_add_chunks_sends_model_provider_and_per_chunk_metadata() -> None:
    chunk = make_chunk("hello")
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/records"):
            return httpx.Response(200, json={"data": []})
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "record_ids": ["r1"]})

    make_store(handler).add_chunks("ds", chunks=[chunk])
    (payload,) = payloads
    assert payload["texts"] == [chunk.text]
    assert payload["dataset_name"] == "ds"
    assert payload["model_name"] == MODEL
    assert payload["provider"] == "sentence-transformers"
    assert payload["metadata"] == chunk.to_metadata()


def test_add_chunks_embed_failure_is_loud() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/records"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(500, json={"detail": "Failed to embed text"})

    with pytest.raises(RuntimeError, match="embed failed"):
        make_store(handler).add_chunks("ds", chunks=[make_chunk("x")])


# Shape copied verbatim from the live B1 probe of /api/v1/vector/search.
SEARCH_RESPONSE = {
    "success": True,
    "query": "where is cosine similarity computed",
    "total_results": 2,
    "search_time_ms": 4384.5,
    "results": [
        {
            "record_id": "r1",
            "similarity_score": 0.553,
            "rank": 1,
            "source_text": "def cosine(a, b): return dot(a, b)",
            "metadata": {"id": "c1", "repo": "r", "path": "a.py", "text": "def cosine..."},
        },
        {
            "record_id": "r2",
            "similarity_score": -0.006,
            "rank": 2,
            "source_text": "unrelated",
            "metadata": {"id": "c2", "repo": "r", "path": "b.py", "text": "unrelated"},
        },
    ],
}


def test_search_normalizes_results_into_hits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/vector/search"
        assert json.loads(request.content) == {"query": "q", "dataset_name": "ds", "k": 2}
        return httpx.Response(200, json=SEARCH_RESPONSE)

    hits = make_store(handler).search("ds", query="q", k=2)
    assert [(hit.score, hit.native_id) for hit in hits] == [(0.553, "r1"), (-0.006, "r2")]
    assert hits[0].metadata["path"] == "a.py"  # chunk metadata survives the round trip


def test_search_missing_dataset_is_a_value_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "The requested resource was not found"})

    with pytest.raises(ValueError, match="not indexed"):
        make_store(handler).search("nope", query="q", k=3)


def test_search_server_errors_are_loud() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Internal server error"})

    with pytest.raises(RuntimeError, match="search failed"):
        make_store(handler).search("ds", query="q", k=3)


def test_unreachable_server_hints_at_docker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(RuntimeError, match="docker compose up"):
        make_store(handler).create_dataset("ds", metric="cosine")


def test_close_closes_the_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    store = make_store(handler)
    store.close()
    assert store.client.is_closed
