"""Links as entities: a small SQLite store beside the vectors.

A link is a fact about a name — that code reads a port, that a config
declares one, that a commit wrote a chunk's lines. It is an entity rather
than chunk metadata because the query inverts ("who reads X"), because it
outlives the chunk it was found in, and because the relation carries
attributes of its own. See ADR-9 for which edge kinds earned their place.

SQLite rather than the vector store: this is a join, and a vector store
that answered joins would be a database with a worse query language.
It lives in the index directory — local to this machine, like
`state.json`, even when the vectors sit in S3.

Links have the lifetime of their source chunk. A link that outlives its
chunk is indistinguishable from a real dangling one, so the drift report
would fill with references from code that no longer exists.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any

from wsindex.paths import make_index_dir

LINKS_FILE = "links.db"
"""Name of the database inside the index directory."""

_COLUMNS = {
    "src_chunk_id": "TEXT NOT NULL",
    "kind": "TEXT NOT NULL",
    "name": "TEXT NOT NULL",
    "line": "INTEGER NOT NULL",
    "dst_chunk_id": "TEXT",
    "url": "TEXT",
    "via": "TEXT",
    "norm": "TEXT",
    "repo": "TEXT NOT NULL",
    "path": "TEXT NOT NULL",
}
"""The table's columns, in order, so `_migrate` can tell what an older
database is missing. Kept beside the CREATE rather than parsed out of it:
two spellings of the same truth, but the alternative is parsing SQL."""


_TABLE = """
    CREATE TABLE IF NOT EXISTS links (
        src_chunk_id TEXT NOT NULL,
        kind         TEXT NOT NULL,
        name         TEXT NOT NULL,
        line         INTEGER NOT NULL,
        dst_chunk_id TEXT,
        url          TEXT,
        via          TEXT,
        norm         TEXT,
        repo         TEXT NOT NULL,
        path         TEXT NOT NULL,
        PRIMARY KEY (src_chunk_id, kind, name, line)
    )
    """
"""The table. Separate from its indexes because a migration runs between
them: an index over a column `_migrate` is about to add fails on any
database written before that column, with `no such column`. Found by the
test that exists for exactly this, when `norm` arrived."""


_PRUNED_TABLE = """
    CREATE TABLE IF NOT EXISTS pruned (
        norm TEXT PRIMARY KEY
    )
    """
"""Names whose mentions `prune_unjoinable` removed.

The one way pruning turns into a wrong answer rather than a smaller
file: a repository joins the workspace later, defines a name whose
mentions were already dropped, and the files holding those mentions have
not changed — so nothing re-extracts them and `refs` reports a
definition with no uses. Silence is the worst failure mode here, because
it reads as a fact about the code.

Names only, and normalised, so this is small against what it guards:
measured, it is 2% of what pruning saves."""


_INDEXES = (
    # Three queries, three indexes. All equality lookups, which is the
    # whole argument for keeping links in SQL rather than beside the
    # vectors: `dangling` is an anti-join, which a vector store's filter
    # language cannot express at all, and links are deleted per changed
    # file on every run — measured at 13x against the columnar store.
    #
    # `links_by_name` covers (kind, name) and serves `dangling`, which
    # constrains both. It does NOT serve `by_name`, which constrains only
    # `name`: a composite index is sorted by its leading column, so a
    # query that leaves that column free has nothing to descend. `refs` —
    # the query this table exists to answer — was therefore reading every
    # row, which EXPLAIN QUERY PLAN says plainly (SCAN, not SEARCH) and
    # which costs 20 ms against 0.1 at a million links.
    "CREATE INDEX IF NOT EXISTS links_by_name ON links (kind, name)",
    "CREATE INDEX IF NOT EXISTS links_by_bare_name ON links (name)",
    "CREATE INDEX IF NOT EXISTS links_by_src ON links (src_chunk_id)",
    # `by_name` asks for both spellings at once, so both columns have to
    # be searchable or the query degrades to a scan on half of itself.
    "CREATE INDEX IF NOT EXISTS links_by_norm ON links (norm)",
)
"""Every index over the table, built after the migration.

Separate statements rather than one script: `executescript` is SQLite's
and takes no parameters, and everything here is plain SQL both backends
accept."""


_SEPARATORS = re.compile(r"[_.\-]")
"""What separates the words of a name when a name has words. Which of
them a project uses is a house style, and the two ends of the same
setting rarely share one."""


def normalised(name: str) -> str:
    """One spelling for every way the same name gets written.

    A setting is `max_retries` in the yaml and `MaxRetries` in the code
    that reads it, and those are different strings, so a store keyed by
    name joins neither to the other. This is the key they meet under.

    It is the one place `refs` can beat `grep`, which is why it is worth
    a column: `rg -w max_retries` misses `MaxRetries` and `rg -i` misses
    it too, because they differ by more than case.

    The cost, stated plainly: `Listen` and `listen` also collapse, and in
    Go those are a different symbol — exported and not. For "where is
    this named" that is noise a reader can see through, which is why
    `by_name` reports the exact spelling first and labels the rest.

    Args:
        name: A name as some file spells it.

    Returns:
        Separators removed, lowercased.
    """
    return _SEPARATORS.sub("", name).lower()


class LinkKind(StrEnum):
    """What one link asserts.

    Six, and they are not equally useful — measured on the pinned corpus
    rather than assumed. In a 569-file infrastructure workspace:
    `BLAMED_BY` 15 086 edges, `REFERENCES` 1 384, `READS_KEY` 170 and
    **`DECLARES` 9**. Since `refs` answers only where a name has both a
    `READS_KEY` and a `DECLARES`, that ceiling is nine — and across five
    workspaces exactly three names of 6 896 had both. `why`, which walks
    `BLAMED_BY`, answered 40 of the 40 commonest symbols.

    The two halves had opposite verdicts, and the reason was where each
    came from: blame is given by git, while the config pair is found by
    four regular expressions over chunk *text*. `links_for` never looked
    at `chunk.symbol` — the name the syntax tree already extracted and
    the store already held — so a vocabulary of nine was the ceiling of
    what four regexes happened to catch, not of what was known.

    `DEFINES`/`MENTIONS` is that omission repaired, and it is a repair
    rather than the `CALLS` edge ADR-9 defers: a mention is not a
    resolved call, and this claims only what it can see. Measured on
    caddyserver, 9 741 chunks: 810 names defined in one file and named in
    another, against **three** for the config pair across five
    workspaces.
    """

    READS_KEY = "reads_key"
    """Code names something a configuration is expected to declare."""

    DECLARES = "declares"
    """A configuration publishes a value code may name."""

    DEFINES = "defines"
    """A chunk is the definition of a symbol — the name the syntax tree
    gave it. One per chunk that has one, so this side is nearly free."""

    MENTIONS = "mentions"
    """A chunk names a symbol it does not define. Unresolved on purpose:
    the extractor sees one file and cannot know whose definition this is,
    so both ends are stored as names and meet in `by_name`, exactly as
    the config pair does. Deliberately *not* a call graph — a name in a
    comment counts, which is a feature for search and would be a lie in
    a call graph."""

    REFERENCES = "references"
    """A commit message or document points at something outside the
    repository — a ticket, an issue, a merge request, a url."""

    BLAMED_BY = "blamed_by"
    """A chunk's lines were last written by a commit. Unlike the pair
    above this one resolves at write time — `dst_chunk_id` names the
    commit's own chunk — because both ends are produced by the same
    indexing run and there is nothing to wait for."""


class Occurrence(StrEnum):
    """How a name occurs where it was found — an attribute of `MENTIONS`.

    This exists because of what it replaced. A resolved call graph was
    the obvious next edge, and measuring it said no: of the mentions in
    caddyserver that name something the workspace defines, 51% are calls
    and 42% are other real code — method receivers, struct literals,
    type parameters, field accesses. Eighteen sampled at random were
    legitimate references, none of them junk. A call graph would discard
    that 42% to remove the 6% that is comment, import and string.

    So the occurrence is *recorded* rather than filtered on. A reader
    asking "where is this used" wants the type references; a reader
    asking "who calls this" wants them ranked below the calls. One
    column serves both, and — unlike resolution — it is decidable from
    the line the name sits on, which is what keeps links extractable one
    file at a time.

    Ordered by how much a reader usually wants it, which is the order
    `refs` reports in.
    """

    CALL = "call"
    """The name is applied to arguments: `ServeHTTP(w, r)`."""

    CODE = "code"
    """A real reference that is not a call — a type, a field, a value.
    Deliberately not called `type`: `p.ClientAuthentication` is a field
    and `&FileWriter{}` is a literal, and this does not parse enough to
    tell them apart. What it does claim is that this is code."""

    IMPORT = "import"
    """The line brings the name into scope rather than using it."""

    STRING = "string"
    """Inside a string literal. Often a real reference — a module path,
    a template name — which is why it is kept and labelled rather than
    dropped."""

    COMMENT = "comment"
    """Prose about the name. Last because it is the least likely answer
    to "where is this used", and kept because it is sometimes the best
    answer to "what is this"."""


_ANCHORS = (LinkKind.DEFINES.value, LinkKind.DECLARES.value)
"""The two kinds a `MENTIONS` can join to: code defines a name, a config
declares one. Both, because a mention of `TrustedProxies` meets a json
key `trusted_proxies` and that is the join `grep` cannot make."""


OCCURRENCE_ORDER: dict[Occurrence, int] = {
    occurrence: order for order, occurrence in enumerate(Occurrence)
}
"""How much a reader usually wants each occurrence, from the order they
are declared in. Used twice and therefore defined once: the extractor
picks which occurrence of a name in a chunk to keep, and `refs` sorts
what it reports. Two spellings of one order would drift."""


KIND_LABELS: dict[LinkKind, str] = {
    LinkKind.READS_KEY: "read by",
    LinkKind.DECLARES: "declared by",
    LinkKind.DEFINES: "defined in",
    LinkKind.MENTIONS: "named in",
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
        via: How the name occurs here, for `MENTIONS`. None for every
            other kind, which have nothing to say about it.
    """

    src_chunk_id: str
    kind: LinkKind
    name: str
    line: int
    dst_chunk_id: str | None = None
    url: str | None = None
    via: Occurrence | None = None


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
        via: How the name occurs here, for `MENTIONS`; None otherwise.
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
    via: Occurrence | None = None


@dataclass(frozen=True, kw_only=True)
class _Dialect:
    """The four places SQLite and Postgres disagree, and no others.

    Two full implementations was the obvious shape and the wrong one:
    ADR-2 paired Tensorus with LocalStore and spent the rest of its life
    keeping them at semantic parity, because their semantics really did
    differ (HNSW against brute force). SQLite and Postgres are both SQL
    with the same semantics — including the anti-join `dangling` needs —
    so the queries are written once and only these four things vary.
    Parity is then not a discipline anybody has to keep; there is only
    one set of queries to be right.

    Attributes:
        placeholder: `?` for sqlite3, `%s` for psycopg. Every query in
            this module is written with `?` and translated once, in
            `_sql`, so the source reads in one dialect.
        insert_prefix, insert_suffix: How each spells "skip a row that is
            already there" — a prefix for SQLite, a suffix for Postgres.
        columns_query: How to ask which columns a table has, for the
            additive migration.
    """

    placeholder: str
    insert_prefix: str
    insert_suffix: str
    columns_query: str


SQLITE = _Dialect(
    placeholder="?",
    insert_prefix="INSERT OR IGNORE INTO",
    insert_suffix="",
    columns_query="SELECT name FROM pragma_table_info('links')",
)
"""The default, and the only one that needs no service."""

POSTGRES = _Dialect(
    placeholder="%s",
    insert_prefix="INSERT INTO",
    insert_suffix=" ON CONFLICT DO NOTHING",
    columns_query=("SELECT column_name FROM information_schema.columns WHERE table_name = 'links'"),
)
"""For a workspace whose index is shared. Links are workspace data —
every field of one is derived from content, so two machines indexing the
same commit produce the same links — which is why they can be shared at
all, and why leaving them on one machine while the vectors live in S3
was an asymmetry rather than a design."""


class LinkStore:
    """SQLite-backed link storage, one file inside the index directory.

    A context manager: the connection is a real resource and the CLI runs
    one command per process, so the scope is the command.
    """

    def __init__(self, index_dir: Path) -> None:
        """Open (creating if needed) the SQLite database under `index_dir`.

        The default, and the only backend that needs no service running.
        For a shared index see `postgres`.

        Args:
            index_dir: Where this machine keeps its index.
        """
        make_index_dir(index_dir)
        self.path: Path | None = index_dir / LINKS_FILE
        self.dialect = SQLITE
        # `check_same_thread=False` because the server runs a sync
        # endpoint in a worker thread while this connection was opened in
        # the main one, and sqlite3 refuses that by default — measured as
        # a hard failure of `POST /index` before the flag went in. Safe
        # here on two counts: `sqlite3.threadsafety` is 3 (the library
        # serializes access itself), and every write goes through the
        # server's one-writer lock (ADR-10).
        self._db: Any = sqlite3.connect(self.path, check_same_thread=False)
        # WAL: a reader no longer blocks the writer, and a crash mid-write
        # leaves the database usable. Postgres is journalled already.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._create()

    @classmethod
    def postgres(cls, dsn: str) -> LinkStore:
        """Open the same store against a Postgres database.

        For a workspace whose index is shared — an `s3://` store makes
        the vectors common to a team, and links are workspace data by
        the same argument: every field of one is derived from content,
        so two machines indexing the same commit produce identical
        links. Leaving them on one machine while the vectors were shared
        was an asymmetry, not a design.

        One database per shared index, exactly as there is one
        `[store] uri` per shared index: the table is keyed by repo id
        and nothing else, so two workspaces pointing at one DSN merge
        their links the way two workspaces pointing at one store uri
        merge their datasets. The DSN is the boundary. (Found by
        pointing a probe at the test database and watching `refs` answer
        with rows the tests had left there.)

        Args:
            dsn: Connection string. It carries a password, so a config
                names the *variable* that holds it and never the value —
                the same rule `token_env` keeps.

        Returns:
            A store backed by Postgres, with the same queries.

        Raises:
            RuntimeError: psycopg is not installed.
        """
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "postgres links need the `postgres` extra — `uv sync --extra postgres`"
            ) from exc
        store = cls.__new__(cls)
        store.path = None
        store.dialect = POSTGRES
        store._db = psycopg.connect(dsn, autocommit=False)
        store._create()
        return store

    def _sql(self, query: str) -> str:
        """One query, in this backend's spelling.

        Every statement in this module is written with `?`; this is the
        single place that knows psycopg wants `%s`. Writing each query
        twice is how two backends drift.
        """
        return query if self.dialect.placeholder == "?" else query.replace("?", "%s")

    def _create(self) -> None:
        """The table, then the migration, then the indexes.

        That order is load-bearing and was learned by breaking it: an
        index over a column the migration is about to add does not exist
        yet on an older database, and `CREATE INDEX` says `no such
        column` rather than skipping.
        """
        self._db.execute(_TABLE)
        self._db.execute(_PRUNED_TABLE)
        self._db.commit()
        self._migrate()
        for statement in _INDEXES:
            self._db.execute(statement)
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
        present = {row[0] for row in self._db.execute(self.dialect.columns_query)}
        for column, ddl in _COLUMNS.items():
            if column not in present:
                # NOT NULL cannot be added to a populated table without a
                # default in either dialect, and every column this could
                # add arrived optional. See the docstring.
                self._db.execute(f"ALTER TABLE links ADD COLUMN {column} {ddl.split(' NOT')[0]}")
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
        # Through a cursor, and reading `rowcount` off it. Two portability
        # facts, both learned the hard way against a real Postgres:
        # `executemany` lives on the cursor in psycopg (sqlite3 also has
        # it on the connection, which is what hid this), and
        # `total_changes` is sqlite3's own attribute while `rowcount` is
        # DB-API and both report it.
        cursor = self._db.cursor()
        cursor.executemany(
            self._sql(
                f"{self.dialect.insert_prefix} links "
                "(src_chunk_id, kind, name, line, dst_chunk_id, url, via, norm, repo, path) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?){self.dialect.insert_suffix}"
            ),
            [
                (
                    link.src_chunk_id,
                    link.kind.value,
                    link.name,
                    link.line,
                    link.dst_chunk_id,
                    link.url,
                    link.via.value if link.via else None,
                    # Derived here rather than taken from the caller, so
                    # no extractor can spell it differently and quietly
                    # put a name where nothing will find it.
                    normalised(link.name),
                    repo,
                    path,
                )
                for link in links
            ],
        )
        self._db.commit()
        return max(int(cursor.rowcount), 0)

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
        cursor = self._db.cursor()
        cursor.executemany(self._sql("DELETE FROM links WHERE src_chunk_id = ?"), batch)
        self._db.commit()
        return max(int(cursor.rowcount), 0)

    def count(self) -> int:
        """How many links are stored. Mostly for reports and tests."""
        row = self._db.execute("SELECT COUNT(*) FROM links").fetchone()
        return int(row[0])

    def _rows(self, where: str, params: tuple[object, ...], *, order: str = "") -> list[Edge]:
        """Read edges matching a WHERE clause, ordered for reading."""
        rows = self._db.execute(
            self._sql(
                "SELECT kind, name, line, src_chunk_id, dst_chunk_id, url, via, repo, path "
                f"FROM links WHERE {where} ORDER BY {order}repo, path, line"
            ),
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
                # A database written before this column has NULL here,
                # which is the truth about what it recorded.
                via=Occurrence(via) if via else None,
                repo=repo,
                path=path,
            )
            for kind, name, line, src, dst, url, via, repo, path in rows
        ]

    def by_name(self, name: str) -> list[Edge]:
        """Every link that names this thing — the inverted index.

        The query the whole store exists to answer: given a port, a
        ticket, a sha or a setting, who mentions it. Equality on an
        indexed column, which is why links live in SQLite and not beside
        the vectors.

        Spelling is not required to match. A setting is `max_retries` in
        the config and `MaxRetries` in the code that reads it, and asking
        a person to guess which half of their own system spells it which
        way is asking them to already know the answer. Both are found
        (see `normalised`), and the spelling that was asked for is
        reported first, so a variant is visible as a variant rather than
        arriving disguised as an exact hit.

        The `OR name = ?` is for databases written before `norm` existed:
        their rows have NULL there and would otherwise become
        unfindable, which is a silent wrong answer rather than an
        obvious failure.

        Args:
            name: As some file spells it — `8080`, `PROJ-412`,
                `3964bb7`, `max_retries`.

        Returns:
            Every edge naming it, exact spellings first, then by file
            and line.
        """
        return self._rows(
            "(norm = ? OR name = ?)",
            (normalised(name), name, name),
            order="CASE WHEN name = ? THEN 0 ELSE 1 END, ",
        )

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

    def prune_unjoinable(self) -> int:
        """Drop `MENTIONS` of names nothing here defines or declares.

        Both kinds, and the second was learned by deleting it: a first
        version asked only about `DEFINES`, and `refs trusted_proxies`
        went from four uses to none. A config key is anchored by
        `DECLARES`, so a mention of `TrustedProxies` joins perfectly well
        with a json file — and that join is the one thing `refs` does
        that `grep` cannot, which pruning had quietly destroyed.

        Two thirds of them, measured: 18 871 of caddyserver's 28 545
        mentions name something no `DEFINES` answers — `WriteHeader`,
        `ParseInt`, the standard library and the vendored world. They
        can never be half of a join.

        Only safe to call when the whole workspace has just been
        indexed, which is why `Pipeline.index` calls it and nothing else
        does. Mid-run the `DEFINES` table is incomplete and this would
        delete mentions whose definition had not been reached yet.

        Rows written before `norm` existed are left alone. Their `norm`
        is NULL, every comparison against it is false, and this would
        read that as "nothing defines it" and empty the table.

        Returns:
            How many links were removed.
        """
        unjoinable = (
            "kind = ? AND norm IS NOT NULL AND NOT EXISTS ("
            "  SELECT 1 FROM links AS d WHERE d.kind IN (?, ?) AND d.norm = links.norm)"
        )
        kinds = (LinkKind.MENTIONS.value, *_ANCHORS)
        cursor = self._db.cursor()
        # A debt this run settled: the name has a definition now, so a
        # complete pass has just re-extracted whatever mentions it, and
        # the warning would be about a loss that no longer exists. Kept
        # first so a name can be forgiven and re-recorded in one call.
        cursor.execute(
            self._sql(
                "DELETE FROM pruned WHERE EXISTS ("
                "  SELECT 1 FROM links AS d WHERE d.kind IN (?, ?) AND d.norm = pruned.norm)"
            ),
            _ANCHORS,
        )
        # Remembered before they are deleted, and only then deleted, so a
        # crash between the two leaves a name recorded that was never
        # dropped. That way round the failure is a spurious warning; the
        # other way round it is a wrong answer nobody is told about.
        cursor.execute(
            self._sql(
                f"{self.dialect.insert_prefix} pruned (norm) "
                f"SELECT DISTINCT norm FROM links WHERE {unjoinable}"
                f"{self.dialect.insert_suffix}"
            ),
            kinds,
        )
        cursor.execute(self._sql(f"DELETE FROM links WHERE {unjoinable}"), kinds)
        self._db.commit()
        dropped = max(int(cursor.rowcount), 0)
        if dropped:
            self._reclaim()
        return dropped

    def _reclaim(self) -> None:
        """Give the freed pages back to the filesystem.

        Without this the delete frees nothing a user can see: SQLite
        keeps emptied pages in the file for reuse, and the first
        measured run came out *larger* after removing 17 550 rows,
        because the `pruned` table grew and nothing shrank. Since the
        whole point of the delete was size, not reclaiming it would have
        shipped the cost with none of the benefit.

        SQLite only. Postgres has autovacuum, and `VACUUM` there cannot
        run inside the transaction this connection holds open.
        """
        if self.dialect is not SQLITE:
            return
        # VACUUM rewrites the database and refuses to run in one, so the
        # open transaction has to be closed first — `commit` above did
        # that, and isolation_level=None is not set, so be explicit.
        self._db.commit()
        self._db.execute("VACUUM")

    def rescued(self, names: Iterable[str]) -> set[str]:
        """Of these normalised names, which had mentions pruned away.

        Asked after a run that added definitions. A name here means the
        store now holds a definition whose uses were deleted as
        unjoinable, and only a full re-index will put them back.

        Args:
            names: Normalised names defined during this run.

        Returns:
            The subset that pruning had already given up on.
        """
        wanted = list(dict.fromkeys(names))
        if not wanted:
            return set()
        # Chunked because SQLite caps a statement at 999 parameters by
        # default and a run can define thousands of names.
        found: set[str] = set()
        for start in range(0, len(wanted), 500):
            batch = wanted[start : start + 500]
            placeholders = ", ".join("?" for _ in batch)
            found |= {
                row[0]
                for row in self._db.execute(
                    self._sql(f"SELECT norm FROM pruned WHERE norm IN ({placeholders})"),
                    tuple(batch),
                )
            }
        return found

    def anchor_names(self) -> set[str]:
        """Every normalised name this store defines or declares.

        The names a `MENTIONS` can join to, which is what makes one worth
        keeping. `Pipeline.index` compares this across a run to notice
        when such a name arrives that would have rescued mentions an
        earlier prune removed — the one way pruning turns into a wrong
        answer rather than a smaller file.
        """
        return {
            row[0]
            for row in self._db.execute(
                self._sql("SELECT DISTINCT norm FROM links WHERE kind IN (?, ?)"),
                _ANCHORS,
            )
            if row[0]
        }

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
