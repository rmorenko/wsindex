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

MAX_COMMITS = 10_000
"""How far back a full pass reaches, unless `[index] max_commits` says
otherwise. Incremental runs are bounded by the diff and never hit this.

History is unbounded and the questions people ask of it are not, so a cap
has to exist — a repository the size of the Linux kernel would otherwise
add over a million chunks. But **a thousand was measured and found to be
an order of magnitude too tight.**

The measurement, across six real repositories: on five of them the cap
never fires, because they have fewer than a thousand commits. On the
sixth, openemr, it cut 10 351 commits down to the newest 1 000 — leaving
history visible back to 2022 out of a project that starts in 2005.
**Seventeen and a half years and 90% of the commits, invisible**, for a
saving of 23 MB and 15 seconds on an index that already takes 130
seconds and 181 MB.

That trade is the wrong way round for a feature whose whole premise is
that a repository's reasoning lives in its commit messages. The price of
history is small and linear — about 1.4 s and 2.5 MB per thousand
commits — so ten thousand bounds the worst case at roughly 14 s and
23 MB, which is what openemr's *entire* history costs.

How much of the index this governs varies more than any other constant
here, which is why it is the one that became configurable. Measured share
of chunks that are commit messages: 0% for a repository with no history,
1.5% for openemr, 6.1% for this project, 20.6% for the acceptance corpus,
31.3% for another. For that last one the cap decides a third of
everything searchable."""

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


def read_commits(root: Path, *, since: str | None, limit: int | None = None) -> list[Commit]:
    """The commits worth indexing this run, newest first.

    Args:
        root: Repository root.
        since: Last indexed commit, or None for a full pass. Given one,
            only what the repo gained after it is read.
        limit: Cap for a full pass; None means `MAX_COMMITS`.

            **Not `limit: int = MAX_COMMITS`.** Python binds a keyword
            default when the function is defined, so the module attribute
            would be read exactly once ever and `[index] max_commits`
            could never reach it. Found by a probe that set the constant,
            re-indexed three times and got three identical answers.

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
        args.append(f"--max-count={MAX_COMMITS if limit is None else limit}")
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


BLAME_WORKERS = 16
"""How many `git blame` processes run at once.

Blame is per file by nature, so a run starts one process per indexed
file, and waiting for them one at a time was 91% of an indexing run.

**Re-measured after the pass moved into a child (ADR-12), and the old
number did not survive it.** Eight came from a sweep taken when the forks
happened in the engine process, where they cost 9.7 ms each and got no
parallelism: one worker 2.20 s, eight 0.94 s, sixteen 1.16 s — sixteen
was *worse*, so eight looked like the top of a curve. In the child the
forks are cheap and the curve keeps going: over the same corpus, a full
pass takes 4.31 s at one worker, 1.21 s at eight, 1.02 s at sixteen and
0.93 s at thirty-two. A constant justified by a measurement that no
longer applies is a constant nobody has checked.

**And on the case that dominates it barely matters, which is the more
useful half of the answer.** That sweep was a full re-index: the store
already held the chunks, so nothing was embedded and the parent sat idle
while blame ran. A *cold* index — the number anyone actually waits
through — has the parent embedding thousands of chunks, and there eight,
sixteen and thirty-two come out identical: 4.62 s, 4.56 s and 4.59 s,
against 4.76 s at four, which is a spread of 1.3%. So this is not a tuning knob worth anybody's
attention; it is a constant that had to stop citing a measurement that no
longer held.

Sixteen rather than thirty-two, because the sweep was repeated under
different CPU limits and the ends are where the danger is. Blaming 178
files in a container: at two CPUs the best is eight (0.32 s) and sixteen
costs 0.42; at four and at eight CPUs the best is sixteen (0.22 s).
Sixty-four is catastrophic everywhere — 2.44 s at two CPUs, seven times
the optimum. Between eight and thirty-two everything sits within about
30% of best on every machine tried, so this picks the middle of a wide
flat band rather than the peak on one laptop.

**Deliberately not derived from `os.cpu_count()`**, which is the obvious
idea and is measurably wrong where it would matter most: inside a
container limited to two CPUs it still reports fourteen. An
auto-tuned value would be confidently wrong on exactly the CI runners and
Docker hosts that need it, and it would make two benchmark runs
incomparable for a reason nobody could see."""

HELPER_FROM = 4
"""How many files it takes before the batch is worth a child process.

Derived rather than chosen. Starting the helper costs 21 ms (measured,
and the reason it imports nothing from this package). A spawn from a
process holding the model costs ~9.7 ms and gets no parallelism from
threads; from the small child the same spawns do parallelise across
`BLAME_WORKERS`. So the child wins once

    N * 9.7  >  21 + N * 9.7 / BLAME_WORKERS

which is N > 2.3 at sixteen workers (it was 2.5 at eight — the threshold
barely moves, because the child's start dominates it). Four, for the
margin, and below it the in-process path is the cheaper one rather than
merely the older one."""


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
