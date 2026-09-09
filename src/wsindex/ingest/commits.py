"""Commit messages as corpus, blame as edges.

A repository's reasoning is not in its code. "Why is dedup before
embedding" is answered in a commit message and nowhere else, so the
history is indexed alongside the files.

Two halves. Reading the log is cheap — milliseconds for a whole history
— and each message becomes one chunk under a synthetic path
(`commits/2026-09-09-abc1234`), because a commit has no file. Blame is
the expensive half at ~28 ms per file, so it is paid per *indexed* file,
which the incremental pass already keeps down to what changed.

An untracked file has no history and simply gets no edges; git says so
with an error, and that is a normal answer here rather than a failure.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from wsindex.ingest.git_state import GitCommandError, decode_path, run_git
from wsindex.links import Link, LinkKind
from wsindex.model import Chunk, Kind

MAX_COMMITS = 1000
"""How far back a full pass reaches. History is unbounded; the questions
people ask of it are not. Incremental runs are bounded by the diff
instead and never hit this."""

COMMIT_LANG = "git-commit"
"""`lang` recorded on a commit chunk, so `--lang git-commit` narrows to
them and `--lang python` never returns one."""

_RECORD = "\x1e"
_FIELD = "\x1f"
_LOG_FORMAT = f"%H{_FIELD}%aI{_FIELD}%B{_RECORD}"

_BLAME_LINE = re.compile(r"^([0-9a-f]{40}) \d+ (\d+)")
"""A porcelain group header: `<sha> <orig-line> <final-line> [<count>]`.
The final line number is the only other field this needs."""


@dataclass(frozen=True, kw_only=True)
class Commit:
    """One commit, as much of it as is worth embedding.

    Attributes:
        sha: Full 40-character sha; `short` is what a person reads.
        date: Author date, ISO 8601. Part of the chunk's synthetic path
            so that history sorts and reads correctly in search output.
        message: Subject and body, verbatim. This is the text that gets
            embedded, and the reason the whole exercise pays.
    """

    sha: str
    date: str
    message: str

    @property
    def short(self) -> str:
        """The abbreviation everything user-facing uses."""
        return self.sha[:7]

    @property
    def path(self) -> str:
        """Synthetic path a commit chunk is filed under.

        `commits/2026-09-09-e29017f`. A commit has no path on disk, and
        every chunk needs one: it is half of `chunk_id`, and it is what
        search prints. Encoding the date here keeps that output readable
        and history sorted, at no schema cost — the alternative was a new
        column on every chunk in the store for a field only commits use.
        """
        return f"commits/{self.date[:10]}-{self.short}"


def read_commits(root: Path, *, since: str | None, limit: int = MAX_COMMITS) -> list[Commit]:
    """The commits worth indexing this run, newest first.

    Args:
        root: Repository root.
        since: Last indexed commit, or None for a full pass. Given one,
            only what the repo gained after it is read.
        limit: Cap for a full pass.

    Returns:
        The commits, newest first. Empty when nothing is new.

    Raises:
        GitCommandError: git failed.
    """
    # Records and fields are separated by control characters, not
    # newlines: a commit message contains newlines by definition, and
    # any line-oriented split would tear bodies apart.
    args = ["log", f"--format={_LOG_FORMAT}"]
    if since is None:
        args.append(f"--max-count={limit}")
    else:
        args.append(f"{since}..HEAD")
    raw = decode_path(run_git(root, *args))
    commits: list[Commit] = []
    for record in raw.split(_RECORD):
        if not record.strip():
            continue
        sha, date, message = record.lstrip("\n").split(_FIELD, 2)
        commits.append(Commit(sha=sha, date=date, message=message.strip()))
    return commits


def commit_chunks(commits: list[Commit], *, repo: str) -> list[Chunk]:
    """One chunk per commit, ready for the store.

    Args:
        commits: What `read_commits` returned.
        repo: Repo id; commit chunks live in the same dataset as its code
            so one search covers both.

    Returns:
        The chunks, in the order given.
    """
    return [
        Chunk(
            repo=repo,
            path=commit.path,
            lang=COMMIT_LANG,
            kind=Kind.COMMIT,
            # The sha as symbol, so `--symbol e29017f` finds a commit and
            # `wsindex why` has something to join blame links on.
            symbol=commit.short,
            node_type="commit",
            start_line=1,
            end_line=max(len(commit.message.splitlines()), 1),
            text=commit.message,
        )
        for commit in commits
        if commit.message
    ]


def _blame(root: Path, rel_path: str) -> dict[int, str]:
    """Line number -> the sha that last wrote it, for one file.

    One `git blame` per file rather than one per chunk: the porcelain
    output already covers the whole file, and a chunk-sized `-L` range
    would pay the process cost once per chunk instead of once per file.

    An empty result is a normal answer, not a failure. A full pass
    indexes untracked files too (`ls-files --others` lists them), and git
    cannot blame a file that is in no
    commit: `fatal: no such path ... in HEAD`. A file with no history has
    no blame edges, which is exactly right — and letting that kill the
    run would mean one new file breaks indexing for the whole workspace.
    """
    try:
        raw = decode_path(run_git(root, "blame", "--porcelain", "--", rel_path))
    except GitCommandError:
        return {}
    by_line: dict[int, str] = {}
    for line in raw.splitlines():
        header = _BLAME_LINE.match(line)
        if header is not None:
            by_line[int(header.group(2))] = header.group(1)
    return by_line


BLAME_WORKERS = 8
"""How many `git blame` processes run at once.

Blame is per file by nature, so a run forks once per indexed file, and
waiting for them one at a time was 91% of an indexing run. The work
happens in another process and `subprocess` releases the GIL while it
does, so threads spend that wait in parallel. Measured over 122 files:
one worker 2.20 s, eight 0.94 s, sixteen 1.16 s — past the machine's
cores the forks only compete with each other."""


def blame_map(root: Path, rel_paths: Sequence[str]) -> dict[str, dict[int, str]]:
    """Blame every file at once: path -> {line -> commit sha}.

    Args:
        root: Repository root.
        rel_paths: Files to blame, repo-relative.

    Returns:
        One entry per path; a file git cannot blame — an untracked one —
        maps to an empty dict, which is a normal answer.
    """
    if not rel_paths:
        return {}
    with ThreadPoolExecutor(max_workers=BLAME_WORKERS) as pool:
        blamed = pool.map(lambda rel_path: _blame(root, rel_path), rel_paths)
        return dict(zip(rel_paths, blamed, strict=True))


def blame_links(
    *, chunks: list[Chunk], known: dict[str, str], by_line: dict[int, str]
) -> list[Link]:
    """`BLAMED_BY` edges from a file's chunks to the commits that wrote them.

    Resolved at write time, unlike the code-to-config pair: both ends
    exist by the time this runs, so there is nothing to defer. A chunk
    spanning several commits gets one edge per distinct commit, which is
    what makes "who last touched this, and why" answerable at all.

    Args:
        chunks: That file's chunks, with their line ranges.
        known: Full sha -> commit chunk id, for the commits this run
            indexed. A commit outside that set yields an edge with no
            destination rather than none at all: knowing *which* commit
            still answers "when did this change", and the message can be
            fetched later.
        by_line: What `blame_map` found for this file. Passed in rather
            than fetched here, so the forks can all be in flight at once.

    Returns:
        One link per (chunk, commit) pair.
    """
    links: list[Link] = []
    for chunk in chunks:
        seen: set[str] = set()
        for line in range(chunk.start_line, chunk.end_line + 1):
            sha = by_line.get(line)
            if sha is None or sha in seen:
                continue
            seen.add(sha)
            links.append(
                Link(
                    src_chunk_id=chunk.id,
                    kind=LinkKind.BLAMED_BY,
                    name=sha[:7],
                    line=line,
                    dst_chunk_id=known.get(sha),
                )
            )
    return links
