"""An embedder that is not on this machine, and the reasons to want one.

**This breaks the promise every other part of this tool keeps**, and it is
off by default for that reason. Indexing through it sends every chunk of
every configured repository to somebody else's server. Nothing else here
does that, and if that matters to you, the answer is to not turn this on.

What it buys is measured rather than promised. On the sixty blind
questions of `poe relevance`, against the answers that are reachable at
all, swapping only the model — same chunks, same pipeline, same funnel:

| model | identifier | descriptive | cross-repo |
| --- | --- | --- | --- |
| all-MiniLM-L6-v2 (default, local) | 8 / 11 of 16 | 2 / 5 of 23 | 1 / 3 of 8 |
| voyage-code-4 | 13 / 16 | 9 / 17 | 3 / 7 |

Cells are top-three / top-ten. The gate this project set in advance —
descriptive hit@10 of 0.50 — is 0.22 with the default and 0.74 here.

**There is a middle option that keeps your code at home**, and it is
usually the better trade: leave the index local and turn on a remote
*reranker* instead (`wsindex.rank.remote`). Then only the query and a few
dozen candidate chunks travel, not the corpus, and it reaches the same
13 of 16 on identifier questions.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx

from wsindex.embed.embedder import Embedder

BATCH = 128
"""Texts per request. Providers cap the batch and the tokens in it; this
is the common ceiling, and `_batches` also watches the token estimate."""

TOKEN_BUDGET = 100_000
"""Estimated tokens per request, under the usual per-call limit."""

RETRIES = 6
"""Attempts before giving up on a request that failed for a reason that
might pass — rate limits and the 5xx family. A wrong key or a wrong model
fails on the first try, because retrying those is only slower."""


class RemoteEmbedder(Embedder):
    """Embeddings from an OpenAI-compatible HTTP endpoint.

    Attributes:
        model: What the provider calls the model.
        url: The endpoint. Voyage and OpenAI share this request shape;
            anything else that does is fine too.
        token_env: **The name of an environment variable**, never a
            token. A key in a config file is a key in somebody's git
            history — the same rule `token_env` keeps for connectors and
            `dsn_env` for the link store.
        dim: Vector width. Given rather than discovered because the store
            builds its column before the first call.
        query_prefix: Put in front of a question and never a passage.
        input_types: Whether the provider takes `input_type`
            (`document` / `query`), which is Voyage's way of spelling the
            same asymmetry. OpenAI does not, and sending it is an error
            there, so it is a switch rather than an assumption.
    """

    def __init__(
        self,
        *,
        model: str,
        url: str,
        token_env: str,
        dim: int,
        query_prefix: str = "",
        input_types: bool = False,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._url = url
        self._token_env = token_env
        self._dim = dim
        self._query_prefix = query_prefix
        self._input_types = input_types
        self._timeout = timeout

    @property
    def dim(self) -> int:
        return self._dim

    def _token(self) -> str:
        """The key, from the environment, or a named error.

        Named, because the alternative is a 401 from a vendor, which
        reads as "the service is broken" rather than "you did not export
        the variable your own config asked for".
        """
        value = os.environ.get(self._token_env)
        if not value:
            raise RuntimeError(
                f"${self._token_env} is not set, and `[embeddings] token_env` names it"
            )
        return value

    def _post(self, texts: Sequence[str], kind: str) -> list[list[float]]:
        payload: dict[str, object] = {"input": list(texts), "model": self._model}
        if self._input_types:
            payload["input_type"] = kind
        headers = {"Authorization": f"Bearer {self._token()}"}
        for attempt in range(RETRIES):
            try:
                reply = httpx.post(self._url, json=payload, headers=headers, timeout=self._timeout)
                reply.raise_for_status()
            except httpx.HTTPStatusError as exc:
                retriable = exc.response.status_code in (408, 429, 500, 502, 503, 504, 529)
                if not retriable or attempt == RETRIES - 1:
                    raise RuntimeError(
                        f"{self._model} at {self._url} answered "
                        f"{exc.response.status_code}: {exc.response.text[:200]}"
                    ) from exc
                time_to_wait = 2**attempt
                httpx_sleep(time_to_wait)
                continue
            rows = reply.json()["data"]
            # By index, not by arrival: the shape permits either order and
            # a silent transposition here would be a ranking bug nobody
            # could find from the outside.
            return [row["embedding"] for row in sorted(rows, key=lambda r: r["index"])]
        raise RuntimeError("unreachable")  # pragma: no cover - the loop returns or raises

    def _batches(self, texts: Sequence[str]) -> list[list[str]]:
        """Split by both limits a provider enforces: count and tokens."""
        out: list[list[str]] = []
        batch: list[str] = []
        budget = 0
        for text in texts:
            cost = len(text) // 3 + 1
            if batch and (len(batch) >= BATCH or budget + cost > TOKEN_BUDGET):
                out.append(batch)
                batch, budget = [], 0
            batch.append(text)
            budget += cost
        if batch:
            out.append(batch)
        return out

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            vectors.extend(self._post(batch, "document"))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """A question, encoded as the provider wants questions encoded."""
        return self._post([self._query_prefix + query], "query")[0]


def httpx_sleep(seconds: float) -> None:
    """Named so a test can hold the backoff still without holding `time`."""
    import time

    time.sleep(seconds)
