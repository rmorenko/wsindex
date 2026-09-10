"""What was asked, and whether it worked: a local record of searches.

The quality loop the plan asks for. Real questions from a real workspace
beat synthetic ones for deciding whether re-ranking earns its keep or
whether a hybrid index would — the acceptance criteria are ten queries
somebody made up, and this is however many the tool was actually asked.

**Strictly local, and that is a rule rather than a default.** Nothing
here leaves the machine, nothing is aggregated anywhere, and there is a
test asserting a search opens no sockets. The file lives beside the
index it describes.

Deliberately *not* in the link store. Links may live in a shared
Postgres since ADR-11, and a shared database is the wrong home for one
person's questions: this is a note about this machine, like `state.json`
and unlike links. SQLite, always, in the index directory.

Two things are recorded. A **search** — the text, how many hits came
back, the best score, how long it took — answers "what gets asked" and
"what comes back empty". A **pick** — the hit somebody opened out of the
shell — is the only signal here that anyone found what they wanted, and
it is the reason `wsindex shell` was worth building before this.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from wsindex.paths import make_index_dir

STATS_FILE = "stats.db"
"""Name of the database inside the index directory."""

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS searches (
        at        REAL    NOT NULL,
        query     TEXT    NOT NULL,
        k         INTEGER NOT NULL,
        repo      TEXT,
        hits      INTEGER NOT NULL,
        top_score REAL,
        ms        REAL    NOT NULL,
        reranked  INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS picks (
        at       REAL NOT NULL,
        query    TEXT NOT NULL,
        rank     INTEGER NOT NULL,
        chunk_id TEXT
    )
    """,
    # `query` for the "what gets asked" report, `at` for "lately". No
    # index on `hits` or `ms`: those are read by full aggregate anyway,
    # and an index costs every write to buy nothing.
    "CREATE INDEX IF NOT EXISTS searches_by_query ON searches (query)",
    "CREATE INDEX IF NOT EXISTS searches_recent ON searches (at)",
)


@dataclass(frozen=True, kw_only=True)
class Summary:
    """What the log knows, as `wsindex stats` prints it.

    Attributes:
        searches: How many were recorded.
        since: Timestamp of the oldest one, or None when there are none.
        empty: How many returned nothing at all. Rarer than the plan
            expected: semantic search answers *something* unless a
            filter excluded everything or the workspace is unindexed.
            "Zero-result rate" is a keyword-search metric and reads 0%
            here almost always, which is why `weakest` exists.
        p50_ms, p95_ms: Latency **as felt**, which is not the same as
            search latency and is much larger. Every `wsindex search` is
            a fresh process, so the first (and only) search in it pays
            the model load: 2.3 s against the 8 ms `poe bench` reports
            for the search itself. Both numbers are true and they answer
            different questions — this one is what a person waits, and
            it is the strongest argument for `wsindex shell`.
        common: The most-asked queries, with counts.
        weakest: Queries whose best hit scored lowest, worst first —
            the questions this corpus could not really answer. No
            threshold, deliberately: measured on the real model, an
            answered query tops out at 0.53-0.61 and one with no answer
            at 0.19-0.34, but that gap belongs to this model and this
            corpus. A ranking needs no constant and does not go stale.
        picks: How many times a hit was opened out of the shell.
    """

    searches: int
    since: float | None
    empty: int
    p50_ms: float
    p95_ms: float
    common: list[tuple[str, int]]
    weakest: list[tuple[str, float]]
    picks: int

    @property
    def empty_rate(self) -> float:
        """Share of searches that answered nothing; 0.0 when none were made."""
        return self.empty / self.searches if self.searches else 0.0


class SearchLog:
    """The local record. A context manager, like `LinkStore`."""

    def __init__(self, index_dir: Path) -> None:
        """Open (creating if needed) the log under `index_dir`.

        Args:
            index_dir: Where this machine keeps its index. The log goes
                beside it, in the directory `make_index_dir` keeps at
                0700 — this is nobody else's business.
        """
        make_index_dir(index_dir)
        self.path = index_dir / STATS_FILE
        # Same reasoning as `LinkStore`: the server answers searches in
        # worker threads while this was opened in the main one.
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        for statement in _SCHEMA:
            self._db.execute(statement)
        self._db.commit()

    def __enter__(self) -> SearchLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Let go of the connection."""
        self._db.close()

    def searched(
        self,
        query: str,
        *,
        k: int,
        repo: str | None,
        hits: int,
        top_score: float | None,
        ms: float,
        reranked: bool = False,
    ) -> None:
        """Record one search. Never raises.

        Never, and that is the whole contract: this sits on the read
        path, and a full disk or a locked database must cost the user a
        missing row, not a failed search. The tool's job is answering
        questions; keeping notes about them is a side effect and is
        allowed to fail like one.
        """
        try:
            self._db.execute(
                "INSERT INTO searches (at, query, k, repo, hits, top_score, ms, reranked) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), query, k, repo, hits, top_score, ms, int(reranked)),
            )
            self._db.commit()
        except sqlite3.Error:
            pass

    def picked(self, query: str, *, rank: int, chunk_id: str | None) -> None:
        """Record that a hit was opened. Never raises, for the same reason."""
        try:
            self._db.execute(
                "INSERT INTO picks (at, query, rank, chunk_id) VALUES (?, ?, ?, ?)",
                (time.time(), query, rank, chunk_id),
            )
            self._db.commit()
        except sqlite3.Error:
            pass

    def summary(self, *, top: int = 5) -> Summary:
        """Everything `wsindex stats` reports, in one pass per question.

        Args:
            top: How many of the most-asked queries to name.

        Returns:
            The summary; zeroed when nothing has been recorded.
        """
        row = self._db.execute("SELECT COUNT(*), MIN(at), SUM(hits = 0) FROM searches").fetchone()
        total = int(row[0])
        latencies = [
            float(value) for (value,) in self._db.execute("SELECT ms FROM searches ORDER BY ms")
        ]
        common = [
            (str(query), int(count))
            for query, count in self._db.execute(
                "SELECT query, COUNT(*) AS n FROM searches GROUP BY query "
                "ORDER BY n DESC, query LIMIT ?",
                (top,),
            )
        ]
        weakest = [
            (str(query), round(float(score), 3))
            for query, score in self._db.execute(
                "SELECT query, MAX(top_score) AS best FROM searches "
                "WHERE top_score IS NOT NULL GROUP BY query ORDER BY best LIMIT ?",
                (top,),
            )
        ]
        picks = int(self._db.execute("SELECT COUNT(*) FROM picks").fetchone()[0])
        return Summary(
            searches=total,
            since=float(row[1]) if row[1] is not None else None,
            empty=int(row[2] or 0),
            p50_ms=_percentile(latencies, 0.50),
            p95_ms=_percentile(latencies, 0.95),
            common=common,
            weakest=weakest,
            picks=picks,
        )

    def forget(self) -> int:
        """Delete everything recorded, and say how much that was.

        A log somebody cannot empty is a log they did not agree to.
        """
        before = self._db.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        self._db.execute("DELETE FROM searches")
        self._db.execute("DELETE FROM picks")
        self._db.commit()
        return int(before)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list; 0.0 when empty."""
    if not sorted_values:
        return 0.0
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return round(sorted_values[index], 2)


__all__ = ["STATS_FILE", "SearchLog", "Summary"]
