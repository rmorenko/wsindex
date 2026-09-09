"""Links between chunks, and the one rule that keeps them honest.

A link is an entity, not chunk metadata (ADR-9): the query inverts ("who
reads this key"), a link outlives the chunk it was found in, and the
relation is many-to-many with attributes of its own. Equality queries,
not similarity — so SQLite, not the vector store.

Both sides are links
--------------------
The obvious design stores only what code says: "this file references port
8080". Resolving that needs the other side — what the workspace's configs
publish — and *that is a cross-file question in an incremental indexer*.
Re-indexing one changed source file gives no access to config files
nobody touched.

So both sides are stored. Code emits `READS_KEY` ("I reference 8080"),
config emits `DECLARES` ("I publish 8000"), each attached to the chunk it
was found in. "Dangling" is then a query — a `READS_KEY` whose name no
`DECLARES` answers — computed from whatever is in the store right now.
Nothing has to be resolved at write time, which is what makes it survive
incremental indexing.

That also makes the drift detector work in the direction that matters
most. Delete the config that published a port and its `DECLARES` links
die with its chunks; the code that reads it becomes dangling on the next
query. The feature and the lifetime rule are the same mechanism.

Lifetimes
---------
Links are keyed by `chunk_id`, and a chunk id is `sha256(text, path)` —
it does not survive an edit. `Pipeline` deletes the chunks a changed file
no longer produces (step 22) and must delete their links in the same
breath. Otherwise two things rot: the store grows edges pointing at
nothing forever, and — worse — those orphans are indistinguishable from
real dangling links, so the drift report fills with noise from deleted
code and stops being worth reading.

`delete_by_source` exists for exactly that call, and `Pipeline` makes it
with the same set it hands `delete_chunks`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType

LINKS_FILE = "links.db"
"""Name of the database inside the index directory."""


class LinkKind(StrEnum):
    """What one link asserts.

    Only the two the spike measured at zero false alarms are here.
    ADR-9 lists more (CALLS, IMPORTS, READS_ENV, BLAMED_BY,
    EXPLAINED_IN); each arrives with the evidence that it pays.
    """

    READS_KEY = "reads_key"
    """Code names something a configuration is expected to declare."""

    DECLARES = "declares"
    """A configuration publishes a value code may name."""

    BLAMED_BY = "blamed_by"
    """A chunk's lines were last written by a commit. Unlike the pair
    above this one resolves at write time — `dst_chunk_id` names the
    commit's own chunk — because both ends are produced by the same
    indexing run and there is nothing to wait for."""


@dataclass(frozen=True, kw_only=True)
class Link:
    """One edge, anchored to the chunk it was found in.

    Attributes:
        src_chunk_id: Chunk the link was found in. Its lifetime governs
            the link's: when the chunk goes, so does this.
        kind: What the link asserts.
        name: The thing named — a port, a key. Both sides of a pair use
            the same spelling, which is what lets them meet.
        line: 1-based line within the file, for pointing a user at it.
        dst_chunk_id: The chunk this resolves to, when something already
            knows. Left None by the extractors: resolution is a query
            (see `dangling`), not a fact frozen at write time.
    """

    src_chunk_id: str
    kind: LinkKind
    name: str
    line: int
    dst_chunk_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class Drift:
    """A `READS_KEY` that no `DECLARES` answers — code and config apart.

    Attributes:
        name: What the code named and nothing declares.
        chunk_id: Chunk the reference sits in.
        line: Where in the file.
        repo: Repo the chunk belongs to.
        path: Repo-relative path of the file.
    """

    name: str
    chunk_id: str
    line: int
    repo: str
    path: str


class LinkStore:
    """SQLite-backed link storage, one file inside the index directory.

    A context manager: the connection is a real resource and the CLI runs
    one command per process, so the scope is the command.
    """

    def __init__(self, index_dir: Path) -> None:
        """Open (creating if needed) the link database under `index_dir`.

        Args:
            index_dir: Where the workspace keeps its index. Local by
                definition, even when the vectors live in S3 — the same
                reasoning that puts `state.json` there.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        self.path = index_dir / LINKS_FILE
        self._db = sqlite3.connect(self.path)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS links (
                src_chunk_id TEXT NOT NULL,
                kind         TEXT NOT NULL,
                name         TEXT NOT NULL,
                line         INTEGER NOT NULL,
                dst_chunk_id TEXT,
                repo         TEXT NOT NULL,
                path         TEXT NOT NULL,
                PRIMARY KEY (src_chunk_id, kind, name, line)
            );
            -- The two queries that exist: resolve a name, and forget a
            -- chunk. Both are equality lookups, which is the whole
            -- argument for keeping links out of the vector store.
            CREATE INDEX IF NOT EXISTS links_by_name ON links (kind, name);
            CREATE INDEX IF NOT EXISTS links_by_src ON links (src_chunk_id);
            """
        )
        self._db.commit()

    def __enter__(self) -> LinkStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the connection."""
        self._db.close()

    def add_links(self, links: Sequence[Link], *, repo: str, path: str) -> int:
        """Store links, ignoring ones already there.

        Idempotent by primary key, for the same reason `add_chunks` is:
        a re-index must not double what it finds.

        Args:
            links: Links to store.
            repo: Repo id of the file they came from.
            path: Repo-relative path of that file.

        Returns:
            How many rows were actually written.
        """
        if not links:
            return 0
        before = self._db.total_changes
        self._db.executemany(
            "INSERT OR IGNORE INTO links "
            "(src_chunk_id, kind, name, line, dst_chunk_id, repo, path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    link.src_chunk_id,
                    link.kind.value,
                    link.name,
                    link.line,
                    link.dst_chunk_id,
                    repo,
                    path,
                )
                for link in links
            ],
        )
        self._db.commit()
        return self._db.total_changes - before

    def delete_by_source(self, ids: Iterable[str]) -> int:
        """Forget every link found in the given chunks.

        The other half of `VectorStore.delete_chunks`. `Pipeline` calls
        both with the same set, because a link that outlives its chunk is
        not merely stale — it is indistinguishable from a real dangling
        link, and would quietly poison the drift report with references
        from code that no longer exists.

        Args:
            ids: Chunk ids that no longer exist.

        Returns:
            How many links were removed.
        """
        batch = [(chunk_id,) for chunk_id in ids]
        if not batch:
            return 0
        before = self._db.total_changes
        self._db.executemany("DELETE FROM links WHERE src_chunk_id = ?", batch)
        self._db.commit()
        return self._db.total_changes - before

    def count(self) -> int:
        """How many links are stored. Mostly for reports and tests."""
        row = self._db.execute("SELECT COUNT(*) FROM links").fetchone()
        return int(row[0])

    def dangling(self) -> list[Drift]:
        """Every `READS_KEY` that no `DECLARES` answers.

        The drift detector, and the reason both sides are stored: this is
        computed from what is in the store *now*, so it stays right under
        incremental indexing, where the two sides are almost never
        written in the same run.

        Returns:
            One entry per unanswered reference, ordered by file then line
            so a report reads top to bottom.
        """
        rows = self._db.execute(
            """
            SELECT r.name, r.src_chunk_id, r.line, r.repo, r.path
              FROM links AS r
             WHERE r.kind = ?
               AND NOT EXISTS (
                     SELECT 1 FROM links AS d
                      WHERE d.kind = ? AND d.name = r.name
                   )
             ORDER BY r.repo, r.path, r.line
            """,
            (LinkKind.READS_KEY.value, LinkKind.DECLARES.value),
        ).fetchall()
        return [
            Drift(name=name, chunk_id=chunk_id, line=line, repo=repo, path=path)
            for name, chunk_id, line, repo, path in rows
        ]
