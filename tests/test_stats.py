"""The local search log, and the rule that it stays local.

The privacy test is the important one here. "Nothing leaves the machine"
is a claim, and a claim about behaviour is worth exactly as much as the
test under it — review 6 established the shape by measuring that a warm
search opens zero sockets, and this keeps that true now that searching
also writes a file.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from wsindex.config import Config, Repository
from wsindex.embed import FakeEmbedder
from wsindex.model import Kind, SourceFile
from wsindex.pipeline import Pipeline
from wsindex.stats import STATS_FILE, SearchLog
from wsindex.store import LanceDBStore


@pytest.fixture
def log(tmp_path: Path) -> SearchLog:
    return SearchLog(tmp_path / "idx")


def record(
    log: SearchLog, query: str, *, hits: int = 3, top: float | None = 0.5, ms: float = 8.0
) -> None:
    log.searched(query, k=10, repo=None, hits=hits, top_score=top, ms=ms)


# --- what it records ------------------------------------------------------


def test_an_empty_log_reports_nothing_rather_than_crashing(log: SearchLog) -> None:
    summary = log.summary()

    assert (summary.searches, summary.since, summary.empty) == (0, None, 0)
    assert (summary.p50_ms, summary.p95_ms) == (0.0, 0.0)
    assert summary.empty_rate == 0.0


def test_the_most_asked_queries_come_back_in_order(log: SearchLog) -> None:
    for _ in range(3):
        record(log, "how are chunks deduplicated")
    record(log, "where is the store")

    assert log.summary().common == [("how are chunks deduplicated", 3), ("where is the store", 1)]


def test_the_weakest_answers_lead(log: SearchLog) -> None:
    # The metric the plan asked for was zero-result rate, and it reads 0%
    # forever: semantic search answers *something*. What is actually
    # invisible from everywhere else is the query whose best hit was
    # poor, so that is what this reports — ranked, not thresholded.
    record(log, "how are chunks deduplicated", top=0.54)
    record(log, "how do i bake sourdough bread", top=0.19)
    record(log, "kubernetes ingress tls", top=0.23)

    assert [query for query, _ in log.summary().weakest] == [
        "how do i bake sourdough bread",
        "kubernetes ingress tls",
        "how are chunks deduplicated",
    ]


def test_latency_percentiles_are_nearest_rank(log: SearchLog) -> None:
    for ms in (10, 20, 30, 40, 100):
        record(log, "q", ms=ms)

    summary = log.summary()

    assert summary.p50_ms == 30.0
    assert summary.p95_ms == 100.0


def test_a_search_that_answered_nothing_is_counted(log: SearchLog) -> None:
    record(log, "answered", hits=3)
    record(log, "nothing at all", hits=0, top=None)

    summary = log.summary()

    assert (summary.empty, summary.searches) == (1, 2)
    assert summary.empty_rate == 0.5


def test_a_pick_is_the_only_signal_anyone_found_anything(log: SearchLog) -> None:
    record(log, "why is dedup before embedding")
    log.picked("why is dedup before embedding", rank=2, chunk_id="abc123")

    assert log.summary().picks == 1


def test_forgetting_says_how_much_it_forgot(log: SearchLog) -> None:
    for i in range(4):
        record(log, f"q{i}")
    log.picked("q0", rank=1, chunk_id=None)

    forgotten = log.forget()

    assert forgotten == 4
    assert log.summary().searches == 0
    assert log.summary().picks == 0


# --- what it must never do ------------------------------------------------


def test_recording_never_breaks_a_search(log: SearchLog) -> None:
    # It sits on the read path. A full disk or a locked database must
    # cost a missing row, not a failed search.
    log.close()

    record(log, "the database is closed under it")
    log.picked("same", rank=1, chunk_id=None)


def test_the_log_lives_beside_the_index_and_nowhere_else(tmp_path: Path) -> None:
    # Not in the link store: links may be a shared Postgres since
    # ADR-11, and one person's questions do not belong in a team's
    # database.
    with SearchLog(tmp_path / "idx") as opened:
        assert opened.path == tmp_path / "idx" / STATS_FILE
        assert opened.path.exists()


def test_searching_opens_no_sockets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The privacy rule, as a test rather than a promise.

    Strictly local is the whole basis for recording by default. If the
    search path — now including the write that records it — ever reaches
    the network, this fails before anybody notices it in a packet
    capture.

    What it does *not* prove, and should not be read as proving: this
    runs on the fake embedder, so it says nothing about the real model.
    That was measured separately (review 6: a warm search opens zero
    sockets) and made local-first in review 4. This guards the part that
    changed.
    """
    opened: list[Any] = []
    real = socket.socket.connect

    def watched(self: socket.socket, address: Any) -> Any:
        opened.append(address)
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", watched)

    config = Config.default("privacy")
    config._data["store"] = {"uri": str(tmp_path / "db")}
    config.add_repo(Repository(id="r", path=str(tmp_path)))
    store = LanceDBStore(uri=str(tmp_path / "db"), embedder=FakeEmbedder(dim=8))
    store.create_dataset("r", metric="cosine")
    src = SourceFile(repo="r", path="a.py", lang="python", kind=Kind.CODE)
    store.add_chunks("r", chunks=[src.chunk(text="def f(): pass", start_line=1, end_line=1)])

    with SearchLog(tmp_path / "idx") as recording:
        pipeline = Pipeline(store=store, state_dir=tmp_path, config=config, stats=recording)
        pipeline.search("anything at all", k=3)

        assert recording.summary().searches == 1

    assert opened == [], f"a search reached the network: {opened}"
