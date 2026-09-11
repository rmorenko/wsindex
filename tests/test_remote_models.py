"""The two halves that are allowed to leave the machine.

Offline, like every other test here: `httpx.post` is replaced, so what is
under test is the request built and the answer read, not a vendor. The
measurements that justify these existing at all are in the module
docstrings and in `docs/field-trial.md`; this file is about the mechanics
that could silently be wrong — a token in the wrong place, an order
quietly transposed, a retry that never stops.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from wsindex.embed.remote import RemoteEmbedder
from wsindex.rank.remote import RemoteReranker


class Recorder:
    """Stands in for `httpx.post`, remembering what it was asked."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> Any:
        self.calls.append({"url": url, "json": json, "headers": headers})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, int):
            asked = httpx.Request("POST", url)
            response = httpx.Response(reply, json={"detail": "no"}, request=asked)
            raise httpx.HTTPStatusError("boom", request=asked, response=response)
        return httpx.Response(200, json=reply, request=httpx.Request("POST", url))


def embedder(**over: Any) -> RemoteEmbedder:
    return RemoteEmbedder(
        model="voyage-code-4",
        url="https://example.invalid/v1/embeddings",
        token_env="TEST_EMBED_KEY",
        dim=3,
        **over,
    )


VECTORS = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}


def test_the_key_comes_from_the_environment_and_never_from_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The rule the whole project keeps: a config names a variable, never a
    # secret, because a token in a config file is a token in a git history.
    monkeypatch.setenv("TEST_EMBED_KEY", "secret-value")
    post = Recorder(VECTORS)
    monkeypatch.setattr(httpx, "post", post)

    embedder().embed(["def f(): pass"])

    assert post.calls[0]["headers"]["Authorization"] == "Bearer secret-value"


def test_a_missing_key_names_the_variable_it_wanted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Otherwise this surfaces as a 401 from a vendor, which reads as "the
    # service is broken" rather than "you did not export your own variable".
    monkeypatch.delenv("TEST_EMBED_KEY", raising=False)

    with pytest.raises(RuntimeError, match=r"\$TEST_EMBED_KEY is not set"):
        embedder().embed(["x"])


def test_a_question_and_a_passage_are_sent_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The asymmetry the contract grew for. Sent as the provider spells it,
    # and only when the provider has the field — OpenAI rejects it.
    monkeypatch.setenv("TEST_EMBED_KEY", "k")
    post = Recorder(VECTORS)
    monkeypatch.setattr(httpx, "post", post)
    subject = embedder(input_types=True, query_prefix="find code: ")

    subject.embed(["def f(): pass"])
    subject.embed_query("where do we retry")

    assert post.calls[0]["json"]["input_type"] == "document"
    assert post.calls[0]["json"]["input"] == ["def f(): pass"]
    assert post.calls[1]["json"]["input_type"] == "query"
    assert post.calls[1]["json"]["input"] == ["find code: where do we retry"]


def test_an_endpoint_without_input_types_is_not_sent_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_EMBED_KEY", "k")
    post = Recorder(VECTORS)
    monkeypatch.setattr(httpx, "post", post)

    embedder().embed(["x"])

    assert "input_type" not in post.calls[0]["json"]


def test_vectors_come_back_in_the_order_they_were_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shape permits any order, and a silent transposition here is a
    # ranking bug invisible from outside: every chunk gets its
    # neighbour's vector and the index is quietly wrong forever.
    monkeypatch.setenv("TEST_EMBED_KEY", "k")
    shuffled = {
        "data": [
            {"index": 2, "embedding": [3.0]},
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]
    }
    monkeypatch.setattr(httpx, "post", Recorder(shuffled))

    assert embedder().embed(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_a_batch_is_split_by_count_and_by_size(monkeypatch: pytest.MonkeyPatch) -> None:
    # Providers cap both, and hitting either one is a 4xx in the middle of
    # an index run rather than at the start of it.
    monkeypatch.setenv("TEST_EMBED_KEY", "k")
    post = Recorder(VECTORS)
    monkeypatch.setattr(httpx, "post", post)

    embedder().embed(["x"] * 300)

    assert len(post.calls) == 3
    assert [len(call["json"]["input"]) for call in post.calls] == [128, 128, 44]


def test_a_rate_limit_is_retried_and_a_bad_key_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 429 passes; 401 never will, and retrying it six times only makes the
    # failure slower to read.
    monkeypatch.setenv("TEST_EMBED_KEY", "k")
    monkeypatch.setattr("wsindex.embed.remote.httpx_sleep", lambda _: None)
    patient = Recorder(429, VECTORS)
    monkeypatch.setattr(httpx, "post", patient)
    assert embedder().embed(["x"]) == [[0.1, 0.2, 0.3]]
    assert len(patient.calls) == 2

    monkeypatch.setattr(httpx, "post", Recorder(401))
    with pytest.raises(RuntimeError, match="answered 401"):
        embedder().embed(["x"])


def test_the_reranker_puts_scores_back_in_the_callers_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This endpoint answers sorted by score. Reading it positionally hands
    # every candidate its neighbour's rank, which looks like a working
    # reranker that has learned nothing.
    monkeypatch.setenv("TEST_RANK_KEY", "k")
    monkeypatch.setattr(
        httpx,
        "post",
        Recorder(
            {
                "data": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.5},
                ]
            }
        ),
    )
    subject = RemoteReranker(
        model="rerank-2.5", url="https://example.invalid/v1/rerank", token_env="TEST_RANK_KEY"
    )

    assert subject.rank("q", ["a", "b", "c"]) == [0.1, 0.5, 0.9]


def test_the_reranker_asks_nothing_for_no_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    # A search that found nothing must not become a billed round trip.
    post = Recorder({"data": []})
    monkeypatch.setattr(httpx, "post", post)
    subject = RemoteReranker(
        model="rerank-2.5", url="https://example.invalid/v1/rerank", token_env="TEST_RANK_KEY"
    )

    assert subject.rank("q", []) == []
    assert post.calls == []
