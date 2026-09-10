"""Commit messages as corpus, blame as edges.

A repository's reasoning is not in its code. "Why is dedup before
embedding" is answered in a commit message and nowhere else, so the
history is indexed alongside the files.

Two halves. Reading the log is cheap — milliseconds for a whole history
— and each message becomes one chunk under a synthetic path
(`commits/2026-09-09-abc1234`), because a commit has no file. Blame is
the expensive half: one `git blame` per file, paid per *indexed* file,
which the incremental pass already keeps down to what changed. See
`BLAME_WORKERS` for why they run in a pool, `HELPER_FROM` for why a batch
of any size runs in a small child process, and `wsindex.ingest.blame` for
what that measurably buys.

An untracked file has no history and simply gets no edges; git says so
with an error, and that is a normal answer here rather than a failure.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from wsindex.ingest import blame as blaming
from wsindex.ingest.git_state import GIT_TIMEOUT, GitCommandError, decode_path, run_git
from wsindex.links import Link, LinkKind
from wsindex.model import Chunk, Kind

log = logging.getLogger(__name__)

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


def _blame(root: Path, rel_path: str) -> bytes | None:
    """`git blame --porcelain` for one file, through the safe invoker.

    None when git refused the file, and that is a normal answer rather
    than a failure. A full pass indexes untracked files too (`ls-files
    --others` lists them), and git cannot blame a file that is in no
    commit: `fatal: no such path ... in HEAD`. A file with no history has
    no blame edges, which is exactly right — and letting that kill the
    run would mean one new file breaks indexing for the whole workspace.
    """
    try:
        return run_git(root, "blame", "--porcelain", "--", rel_path)
    except GitCommandError:
        return None


BLAME_WORKERS = 8
"""How many `git blame` processes run at once.

Blame is per file by nature, so a run forks once per indexed file, and
waiting for them one at a time was 91% of an indexing run. The work
happens in another process and `subprocess` releases the GIL while it
does, so threads spend that wait in parallel. Measured over 122 files:
one worker 2.20 s, eight 0.94 s, sixteen 1.16 s — past the machine's
cores the forks only compete with each other."""

HELPER_FROM = 4
"""How many files it takes before the batch is worth a child process.

Derived rather than chosen. Starting the helper costs 21 ms (measured,
and the reason it imports nothing from this package). A spawn from a
process holding the model costs ~9.7 ms and gets no parallelism from
threads; from the small child the same spawns do parallelise across
`BLAME_WORKERS`. So the child wins once

    N * 9.7  >  21 + N * 9.7 / 8

which is N > 2.5. Four, for the margin — and below it the in-process
path is the cheaper one, not merely the older one."""


def blame_map(root: Path, rel_paths: Sequence[str]) -> dict[str, dict[int, str]]:
    """Blame every file at once: path -> {line -> commit sha}.

    A batch of any size runs in a child process (see `wsindex.ingest.blame`
    for what that is worth and what it is not known to be worth); a small
    one runs here, because the child would cost more than the forks it
    saves. Both paths are the same function over the same parser, so the
    two cannot answer differently.

    Args:
        root: Repository root.
        rel_paths: Files to blame, repo-relative.

    Returns:
        One entry per path; a file git cannot blame — an untracked one —
        maps to an empty dict, which is a normal answer.

    Raises:
        RuntimeError: One file could not be blamed, named. A failure in a
            worker thread does reach the caller — `map` re-raises when
            the results are walked — but it arrives with a traceback
            through `concurrent.futures` and no idea which of a hundred
            files caused it. The name is added here, where it is known.
    """
    if not rel_paths:
        return {}
    if len(rel_paths) >= HELPER_FROM:
        blamed = _in_child(root, rel_paths)
        if blamed is not None:
            return blamed
    return blaming.blame_files(root, rel_paths, workers=BLAME_WORKERS, blame=_blame)


def _in_child(root: Path, rel_paths: Sequence[str]) -> dict[str, dict[int, str]] | None:
    """The same batch, spawned from a small process; None if that failed.

    None rather than an exception, always: this is an optimisation, and
    an optimisation that can stop an index run is a liability. Whatever
    went wrong — no interpreter to hand, a frozen build with no source
    file, git missing, a wedged child — the caller does the work here
    instead and any real error arrives from `run_git` with its own type
    and message.
    """
    helper = getattr(blaming, "__file__", None)
    if not helper or not sys.executable:  # pragma: no cover - frozen or embedded builds
        return None
    request = json.dumps(
        {
            "root": str(root),
            "paths": list(rel_paths),
            "workers": BLAME_WORKERS,
            "timeout": GIT_TIMEOUT,
        }
    )
    try:
        finished = subprocess.run(
            [sys.executable, helper],
            input=request.encode("ascii"),
            capture_output=True,
            check=True,
            # The read-only hint is set here and inherited by every git
            # the child starts, so the child makes no policy of its own.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            # Per file, times the batch: each git inside has GIT_TIMEOUT
            # of its own, and this only stops a child that stopped
            # answering altogether.
            timeout=GIT_TIMEOUT * len(rel_paths),
        )
        blamed = json.loads(finished.stdout)
        return {path: {int(n): sha for n, sha in lines.items()} for path, lines in blamed.items()}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        log.debug("blame helper unavailable, blaming in-process: %s", exc)
        return None


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
