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
no longer produces and must delete their links in the same
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

_COLUMNS = {
    "src_chunk_id": "TEXT NOT NULL",
    "kind": "TEXT NOT NULL",
    "name": "TEXT NOT NULL",
    "line": "INTEGER NOT NULL",
    "dst_chunk_id": "TEXT",
    "url": "TEXT",
    "repo": "TEXT NOT NULL",
    "path": "TEXT NOT NULL",
}
"""The table's columns, in order, so `_migrate` can tell what an older
database is missing. Kept beside the CREATE rather than parsed out of it:
two spellings of the same truth, but the alternative is parsing SQL."""


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

    REFERENCES = "references"
    """A commit message or document points at something outside the
    repository — a ticket, an issue, a merge request, a url."""

    BLAMED_BY = "blamed_by"
    """A chunk's lines were last written by a commit. Unlike the pair
    above this one resolves at write time — `dst_chunk_id` names the
    commit's own chunk — because both ends are produced by the same
    indexing run and there is nothing to wait for."""


KIND_LABELS: dict[LinkKind, str] = {
    LinkKind.READS_KEY: "read by",
    LinkKind.DECLARES: "declared by",
    LinkKind.REFERENCES: "mentioned in",
    LinkKind.BLAMED_BY: "wrote",
}
"""How each kind reads in a report, from the *named thing's* point of
view — so `BLAMED_BY` reads "wrote", not "written by": ask about a commit
and the answer is what that commit wrote. Here rather than in each
reader, so the CLI and an agent see one vocabulary.

Ordered as a report reads best, which `dict` preserves.
"""


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
            knows. Left None by the code-to-config extractors: resolution
            there is a query (see `dangling`), not a fact frozen at write
            time. `BLAMED_BY` does fill it, because both ends are built
            by the same run.
        url: Where a `REFERENCES` link points, outside the repository.
            Resolved at index time from the config's templates, so the
            link store reads on its own — at the cost of going stale if a
            template changes, which a full re-index fixes.
    """

    src_chunk_id: str
    kind: LinkKind
    name: str
    line: int
    dst_chunk_id: str | None = None
    url: str | None = None


@dataclass(frozen=True, kw_only=True)
class Edge:
    """A stored link, with the file it sits in — what a reader needs.

    `Link` is what an extractor produces: anchored to a chunk id, which
    is enough to store but not enough to show anyone. An `Edge` is the
    same thing read back with the repo and path joined on, so a command
    can point at a place.

    Attributes:
        kind: What the link asserts.
        name: The thing named — a port, a ticket, a commit sha.
        line: 1-based line in the file.
        chunk_id: Chunk the link sits in.
        dst_chunk_id: What it resolves to inside the index, if anything.
        url: Where it points outside the repository, if anywhere.
        repo: Repo the chunk belongs to.
        path: Repo-relative path of the file.
    """

    kind: LinkKind
    name: str
    line: int
    chunk_id: str
    dst_chunk_id: str | None
    url: str | None
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
        # `check_same_thread=False` because the server runs a sync
        # endpoint in a worker thread while this connection was opened in
        # the main one, and sqlite3 refuses that by default — measured as
        # a hard failure of `POST /index` before the flag went in. Safe
        # here on two counts: `sqlite3.threadsafety` is 3 (the library
        # serializes access itself), and every write goes through the
        # server's one-writer lock (ADR-10). The default exists to catch
        # accidental sharing; this sharing is the design.
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS links (
                src_chunk_id TEXT NOT NULL,
                kind         TEXT NOT NULL,
                name         TEXT NOT NULL,
                line         INTEGER NOT NULL,
                dst_chunk_id TEXT,
                url          TEXT,
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
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        """Add columns a database from an older wsindex is missing.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so a `links.db` written before a column was introduced
        keeps its old shape and every read fails with `no such column`.
        Found the hard way: `url` was added later and broke `refs` on
        any index built before it.

        Additive only, and that is enough by construction: a link is
        derived data, so a column that changes meaning is answered by
        re-indexing, not by rewriting rows. Old rows get NULL for the new
        column, which is the truth — they were written when there was
        nothing to put there.
        """
        present = {row[1] for row in self._db.execute("PRAGMA table_info(links)")}
        for column, ddl in _COLUMNS.items():
            if column not in present:
                self._db.execute(f"ALTER TABLE links ADD COLUMN {column} {ddl}")

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
            "(src_chunk_id, kind, name, line, dst_chunk_id, url, repo, path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    link.src_chunk_id,
                    link.kind.value,
                    link.name,
                    link.line,
                    link.dst_chunk_id,
                    link.url,
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

    def _rows(self, where: str, params: tuple[object, ...]) -> list[Edge]:
        """Read edges matching a WHERE clause, ordered for reading."""
        rows = self._db.execute(
            "SELECT kind, name, line, src_chunk_id, dst_chunk_id, url, repo, path "
            f"FROM links WHERE {where} ORDER BY repo, path, line",
            params,
        ).fetchall()
        return [
            Edge(
                kind=LinkKind(kind),
                name=name,
                line=line,
                chunk_id=src,
                dst_chunk_id=dst,
                url=url,
                repo=repo,
                path=path,
            )
            for kind, name, line, src, dst, url, repo, path in rows
        ]

    def by_name(self, name: str) -> list[Edge]:
        """Every link that names this thing — the inverted index.

        The query the whole store exists to answer: given a port, a
        ticket or a sha, who mentions it. Equality on an indexed column,
        which is why links live in SQLite and not beside the vectors.

        Args:
            name: Exactly as it was recorded — `8080`, `PROJ-412`,
                `3964bb7`.

        Returns:
            Every edge with that name, ordered by file then line.
        """
        return self._rows("name = ?", (name,))

    def out_of(self, chunk_ids: Sequence[str], *, kind: LinkKind | None = None) -> list[Edge]:
        """Links found inside the given chunks.

        The other direction from `by_name`: not "who names this" but
        "what does this point at". `why` walks it to get from a
        definition to the commits that wrote it.

        Args:
            chunk_ids: Chunks to read links out of.
            kind: Restrict to one kind, or None for all.

        Returns:
            The edges, ordered by file then line.
        """
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        where = f"src_chunk_id IN ({placeholders})"
        params: tuple[object, ...] = tuple(chunk_ids)
        if kind is not None:
            where += " AND kind = ?"
            params += (kind.value,)
        return self._rows(where, params)

    def dangling(self) -> list[Edge]:
        """Every `READS_KEY` that no `DECLARES` answers.

        The drift detector, and the reason both sides are stored: this is
        computed from what is in the store *now*, so it stays right under
        incremental indexing, where the two sides are almost never
        written in the same run.

        Returns:
            One entry per unanswered reference, ordered by file then line
            so a report reads top to bottom.
        """
        return self._rows(
            "kind = ? AND NOT EXISTS ("
            "  SELECT 1 FROM links AS d WHERE d.kind = ? AND d.name = links.name)",
            (LinkKind.READS_KEY.value, LinkKind.DECLARES.value),
        )
