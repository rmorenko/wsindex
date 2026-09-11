"""What one indexing run is made of: its tallies, its rules, its report.

Split out of `wsindex.pipeline` when that module reached four roles and
992 lines — the threshold its own docstring had named. What lives here is
everything about a *run* that is not the engine itself: the running
sums, the decision about why a repo had to be read whole, the shapes a
finished index answers questions in, and the context manager that makes
a failure say which file it was reading.

Nothing here knows about the store or the config beyond `Repository`, so
it can be read on its own — which is the point of the split.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath

from wsindex.config import Repository
from wsindex.ingest import PARSE_ERROR, IndexState, Skip, WalkedFile, examine
from wsindex.model import Chunk, Hit, Kind

log = logging.getLogger("wsindex.pipeline")
"""Shares the pipeline's logger: to a reader of the log these are one
subsystem, and two names for it would only be a puzzle."""


_WRITE_BATCH = 2000
"""How many chunks accumulate before one write to the store.

Every write is a round trip for deduplication and a version in the store,
so writing per file made both proportional to the file count. Two
thousand chunks is a few megabytes in flight and turns a 3458-chunk repo
into two writes instead of 149."""


UNCLAIMED_SHARE = 0.1
"""Share of a repo's examined paths that may go unclaimed in silence.

Not every skipped suffix is news: a `LICENSE`, a `.png`, a `.lock` are
left out by design and saying so every run would be noise. What is news
is a *language* nobody claims, and the field trial of 2026-09-11 showed
what that looks like from the outside — `wsindex index` printed
`files: 30` on a workspace holding 417 source files, and `status` showed
three healthy repositories. 249 `.ex` and 132 `.exs` had been skipped
whole, and nothing said so.

A tenth, because the same trial put a gap there with room on both sides:
the fourteen workspaces that indexed properly left out a few per cent,
and the three that were broken left out two thirds, 85% and 93%. Any
line between 0.06 and 0.6 would separate them; a tenth is the round one,
far from both edges."""


@dataclass(frozen=True, kw_only=True)
class _Totals:
    """One repo's contribution to an IndexReport; summed by `index`."""

    files: int
    chunks: int
    written: int
    deleted: int
    commits: int
    unreadable: tuple[str, ...] = ()
    unparsed: tuple[str, ...] = ()
    unclaimed: tuple[tuple[str, int], ...] = ()


class FullPass(StrEnum):
    """Why a repo was read whole instead of by delta.

    The note this feeds used to list three possible causes and let the
    reader guess, which is fine until the real cause is a fourth one —
    a state file that could not be read named none of them. Each member
    is the sentence itself: there is no second place where these are
    turned into words, so they cannot drift out of step.
    """

    NEVER_INDEXED = "a first index"
    STATE_LOST = "the index state file could not be read, so the last run is unknown"
    MARKUP_CHANGED = "its ignore/formats markup changed"
    DIRTY_TREE = "uncommitted work, which a commit-to-commit diff cannot see; commit or stash it"
    HISTORY_MOVED = "the commit it was last indexed at is gone (a rebase, a gc, a re-clone)"


@dataclass(frozen=True, kw_only=True)
class _History:
    """What indexing one repo's commit messages produced.

    Attributes:
        written: Message chunks the store actually wrote.
        ids: Their chunk ids, which count as fresh (see `_index_commits`).
        by_sha: Commit sha -> its chunk id, which is what a blame edge
            needs to point at the message that explains a line.
    """

    written: int
    ids: set[str]
    by_sha: dict[str, str]


@dataclass(frozen=True, kw_only=True)
class _Written:
    """Running totals over the files of one repo."""

    files: int = 0
    chunks: int = 0
    written: int = 0
    ids: frozenset[str] = frozenset()
    unparsed: tuple[str, ...] = ()

    def read(self, *, chunks: int, ids: set[str], path: str | None = None) -> _Written:
        """This plus one more file, read and chunked but not yet written.

        `path` is given only when that file's grammar reported errors, so
        the count of them costs one boolean per file rather than a second
        parse.
        """
        return replace(
            self,
            files=self.files + 1,
            chunks=self.chunks + chunks,
            ids=self.ids | ids,
            unparsed=(*self.unparsed, path) if path is not None else self.unparsed,
        )

    def wrote(self, written: int) -> _Written:
        """This plus one batch that reached the store."""
        return replace(self, written=self.written + written)


@dataclass(frozen=True, kw_only=True)
class _Selection:
    """What one repo's changed paths turned into.

    Attributes:
        indexable: The files worth reading.
        forget: Paths whose stored chunks must go — deleted ones, plus
            paths that changed into something unindexable.
        unreadable: Paths git tracks that could not be opened. Neither
            indexable nor forgettable: they are a problem to report, not
            a decision to act on, and folding them into either list is
            how they went unmentioned for as long as they did.
        unclaimed: Suffixes no language claims, and how many files carry
            each. The same kind of fact as `unreadable` and reported for
            the same reason — see `UNCLAIMED_SHARE`.
    """

    indexable: list[WalkedFile]
    forget: list[str]
    unreadable: list[str]
    unclaimed: Counter[str]


def _selection(
    repo: Repository, *, root: Path, changed: tuple[str, ...], deleted: tuple[str, ...]
) -> _Selection:
    """Split the changed paths into what to read, what to forget, what broke.

    A path that changed into something unindexable — renamed to a `.png`,
    grown past the size limit, turned binary — is a deletion as far as the
    store is concerned, which is why it joins the second list rather than
    being skipped.

    Args:
        repo: The repo, for its own `ignore` and `formats` markup.
        root: Repository root.
        changed: Paths git reports as added or modified.
        deleted: Paths git reports as gone.

    Returns:
        The three lists; see `_Selection`.
    """
    indexable: list[WalkedFile] = []
    forget: list[str] = list(deleted)
    unreadable: list[str] = []
    unclaimed: Counter[str] = Counter()
    for rel_path in changed:
        walked = examine(root, rel_path, ignore=repo.ignore, formats=repo.formats)
        if isinstance(walked, WalkedFile):
            indexable.append(walked)
            continue
        if walked is Skip.UNREADABLE:
            unreadable.append(rel_path)
        if walked is Skip.UNKNOWN_SUFFIX:
            unclaimed[PurePosixPath(rel_path).suffix or "(no suffix)"] += 1
        # Unreadable paths are forgotten too: whatever the store still
        # holds for one is from a version nobody can confirm any more.
        forget.append(rel_path)
    return _Selection(
        indexable=indexable, forget=forget, unreadable=unreadable, unclaimed=unclaimed
    )


@dataclass(frozen=True, kw_only=True)
class Reference:
    """Something outside the repository that a commit message named."""

    name: str
    url: str | None


@dataclass(frozen=True, kw_only=True)
class Authorship:
    """One commit that wrote part of a definition.

    Attributes:
        commit: Short sha, as a blame edge records it.
        message: The commit's text, or None when this run's index no
            longer holds it — ordinary, not an error: a blame edge names
            a commit whether or not its message was indexed.
        references: Tickets and urls that message pointed at. The CLI
            showed these and the MCP tool did not, which is the second
            way the two copies of this had drifted.
    """

    commit: str
    message: str | None
    references: tuple[Reference, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Definition:
    """A definition, and the commits that wrote it.

    What `wsindex why` and the MCP tool of the same name both answer. It
    is a type rather than a shape each of them assembles, because they
    used to assemble it separately and had already begun to differ.
    """

    hit: Hit
    commits: tuple[Authorship, ...]


@dataclass(frozen=True, kw_only=True)
class FileReport:
    """What the index knows about one file — `wsindex explain` in data.

    Attributes:
        repo: Repo id the file belongs to.
        rel_path: Its path inside that repo.
        skipped: The rule that left it out, or None when it is indexed.
        lang: Detected language; None when skipped before detection.
        kind: Its category; None for the same reason.
        chunks: How many chunks it produces.
        symbols: How many of those carry a symbol — the difference
            between a file read as definitions and one read as text.
        parsed_cleanly: False when the grammar reported errors, so the
            parts it could not read became text windows.
    """

    repo: str
    rel_path: str
    skipped: Skip | None
    lang: str | None = None
    kind: Kind | None = None
    chunks: int = 0
    symbols: int = 0
    parsed_cleanly: bool = True


@dataclass(frozen=True, kw_only=True)
class IndexReport:
    """Immutable summary of one index() run, totals across all repos.

    Attributes:
        files: How many files were read and chunked. In an incremental
            run this counts the changed files only, so it is a measure
            of work done, not of corpus size.
        chunks: How many chunks those files produced.
        written: How many chunks the store actually wrote; below `chunks`
            means dedup skipped already-stored ones.
        deleted: How many stored chunks were removed as stale — chunks
            of deleted files, and chunks a changed file no longer
            produces.
        commits: How many commit messages were newly indexed. Counted
            apart from `chunks` on purpose: folding them in would make
            `files: 2  chunks: 4` fail to add up for the reader, since
            two of those chunks came from no file at all.
        missing_repos: Ids of configured repos whose directory does not
            exist; they were skipped, not failed on.
        full_repos: Repos that could not go incremental this run, each
            with the reason it could not. A pair rather than a bare id
            because "why did this take nine seconds" is the question the
            field exists to answer.
        unreadable: Paths git tracks that could not be opened, as
            `repo/path`. Not a policy skip: these were meant to be
            indexed and are not, and a run that only printed `files: 1`
            when there were two said something untrue.
        unparsed: Paths whose grammar reported errors, as `repo/path`.
            They are in the index, as text windows rather than
            definitions — worse to search and, until now, impossible to
            notice.
        unclaimed: Suffixes no language claims, commonest first, with
            how many files carry each. Always populated; ask
            `mostly_unclaimed` before saying anything about it.
        seconds: How long the run took. The server already recorded this
            per run; a person at a terminal deserves the same, and it is
            the only number that makes two runs comparable.
    """

    files: int
    chunks: int
    written: int
    deleted: int
    commits: int
    missing_repos: tuple[str, ...]
    full_repos: tuple[tuple[str, FullPass], ...]
    unreadable: tuple[str, ...] = ()
    unparsed: tuple[str, ...] = ()
    unclaimed: tuple[tuple[str, int], ...] = ()
    seconds: float = 0.0

    @property
    def unclaimed_files(self) -> int:
        """How many files were left out for want of a language."""
        return sum(count for _, count in self.unclaimed)

    @property
    def candidates(self) -> int:
        """Files the walker reached that a language could have claimed.

        Read plus left out, and nothing else. Counting everything the
        walker looked at would fold in paths rejected for reasons that
        have nothing to do with languages — inside a hidden directory,
        excluded by `ignore` — and a repository with a large `.wsindex`
        beside it would dilute its own share into silence.
        """
        return self.files + self.unclaimed_files

    @property
    def mostly_unclaimed(self) -> bool:
        """Whether so much went unclaimed that the run must say so.

        The judgement lives here rather than in the CLI because it is a
        fact about the run, and the server and MCP have the same reason
        to want it (ADR-10). See `UNCLAIMED_SHARE`.
        """
        return bool(self.candidates) and self.unclaimed_files > self.candidates * UNCLAIMED_SHARE


def _unparsed(chunks: list[Chunk]) -> bool:
    """True when the grammar could not read the file these came from."""
    return any(chunk.node_type == PARSE_ERROR for chunk in chunks)


@contextmanager
def _blaming(repo_id: str, rel_path: str) -> Iterator[None]:
    """Make sure a failure in here says which file it was reading.

    An index run touches a hundred-odd files, and a bug in chunking one
    of them used to surface as `error: the chunker fell over` — measured,
    with no way to tell which of five files in a toy repo, let alone
    which of 122. The loop knows the path at exactly that moment and was
    dropping it.

    Always a RuntimeError, never `type(exc)(...)`: not every exception
    can be rebuilt from one string — `UnicodeDecodeError` takes five
    arguments — and a TypeError raised from inside an `except` would bury
    the failure it was meant to describe. The original type is named in
    the message and chained underneath, so nothing is lost and
    `WSINDEX_DEBUG` still shows every frame.

    Args:
        repo_id: The repo being indexed.
        rel_path: The file being read, repo-relative.

    Yields:
        Nothing; this is here for the `except`.
    """
    try:
        yield
    except Exception as exc:
        raise RuntimeError(f"{repo_id}/{rel_path}: {type(exc).__name__}: {exc}") from exc


@dataclass
class _Tally:
    """Running sums over the repos of one `index` call.

    Mutable, unlike everything else in this module, and deliberately: it
    is a loop counter with nine fields. Nine locals threaded through the
    loop is what this replaced, and the loop is easier to read than the
    bookkeeping was.
    """

    files: int = 0
    chunks: int = 0
    written: int = 0
    deleted: int = 0
    commits: int = 0
    missing: list[str] = field(default_factory=list)
    full: list[tuple[str, FullPass]] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    unclaimed: Counter[str] = field(default_factory=Counter)

    def add(self, repo_id: str, totals: _Totals) -> None:
        """Fold in what one repo produced, qualifying its paths by repo."""
        self.files += totals.files
        self.chunks += totals.chunks
        self.written += totals.written
        self.deleted += totals.deleted
        self.commits += totals.commits
        self.unclaimed.update(dict(totals.unclaimed))
        self.unreadable.extend(f"{repo_id}/{path}" for path in totals.unreadable)
        self.unparsed.extend(f"{repo_id}/{path}" for path in totals.unparsed)

    def report(self, *, seconds: float) -> IndexReport:
        """Freeze the run into the immutable summary callers get."""
        return IndexReport(
            files=self.files,
            chunks=self.chunks,
            written=self.written,
            deleted=self.deleted,
            commits=self.commits,
            missing_repos=tuple(self.missing),
            full_repos=tuple(self.full),
            unreadable=tuple(self.unreadable),
            unparsed=tuple(self.unparsed),
            unclaimed=tuple(self.unclaimed.most_common()),
            seconds=seconds,
        )


def _why_full(repo: Repository, *, state: IndexState, dirty: bool) -> FullPass:
    """Which of the five reasons made this repo a full pass.

    Ordered by which one the reader can still do something about, which
    is not the same as which came first. A dirty tree wins even when the
    repo has also never been indexed: a dirty tree *is* why nothing was
    recorded, and it will cost a full pass on every run until somebody
    commits, while "a first index" is true once and then never again.
    Found by watching a live server call a workspace it had indexed all
    week "a first index" — correct, and useless.

    Then the lost state file, because it looks exactly like a first index
    and is not; then the markup, which the user changed on purpose; and
    last the one nobody chose.

    Args:
        repo: The repo just planned.
        state: The state as it was loaded, before this run wrote to it.
        dirty: Whether its working tree has uncommitted work.

    Returns:
        The reason, as the sentence the note will print.
    """
    if dirty:
        return FullPass.DIRTY_TREE
    if state.commits.get(repo.id) is None:
        return FullPass.STATE_LOST if state.lost else FullPass.NEVER_INDEXED
    if state.markup.get(repo.id) != repo.markup_key:
        return FullPass.MARKUP_CHANGED
    # A `since` was known, the tree is clean, the markup is the same, and
    # the diff still came back full: git no longer resolves that commit.
    return FullPass.HISTORY_MOVED
