"""Keeping working copies up to date: clone, fetch, fast-forward.

Separate from `git_state` on purpose. That module *observes* — it asks
git what changed and never writes. This one *changes the working copy*,
which is a different kind of risk.

The policy on local work is the whole design, and it is deliberately
timid: **never touch anything the user has not pushed.** Uncommitted
changes, or local commits the remote does not have, both mean sync
declines and says so. The only write it performs is a fast-forward, which
destroys nothing, and a clone into a directory that did not exist.

That timidity costs nothing downstream: `index` handles a dirty tree with
a full pass, so declining degrades speed, never correctness.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from wsindex.ingest.git_state import (
    GitCommandError,
    NotAGitRepositoryError,
    decode_path,
    ensure_repo_root,
    has_uncommitted_changes,
    run_git,
)


class SyncOutcome(StrEnum):
    """What `sync_repo` did, or why it declined.

    A StrEnum so the value is its own human-readable label: the CLI
    prints these verbatim, and a message that lives next to the decision
    cannot drift from it.
    """

    CLONED = "cloned"
    UPDATED = "updated"
    UP_TO_DATE = "up to date"
    DIRTY = "skipped: uncommitted changes"
    DIVERGED = "skipped: local commits the remote does not have"
    NO_UPSTREAM = "skipped: branch tracks no remote branch"

    @property
    def is_skip(self) -> bool:
        """True when nothing was touched and the user should know why."""
        return self in (SyncOutcome.DIRTY, SyncOutcome.DIVERGED, SyncOutcome.NO_UPSTREAM)


def _clone(remote: str, into: Path) -> None:
    """Clone `remote` into `into`, which must not exist yet.

    Full clone, not `--depth 1`: incremental indexing diffs from the
    commit it last indexed, and a shallow history stops resolving that
    commit as soon as the shallow boundary moves past it. `diff_since`
    would degrade to a full listing every time — correct, but it would
    quietly undo incremental indexing.

    Args:
        remote: Clone url, straight from the config.
        into: Target directory; its parent is created if missing.
    """
    into.parent.mkdir(parents=True, exist_ok=True)
    # `--` before the url: a remote that begins with a dash is a filename
    # or a typo, never an option for git to interpret. The parent
    # directory is the cwd because `into` does not exist yet.
    run_git(into.parent, "clone", "--", remote, str(into))


def _upstream(root: Path) -> str | None:
    """The commit the current branch's upstream points at, or None.

    None means the branch tracks nothing — a detached HEAD, or a branch
    created locally and never pushed. Either way there is nothing to
    fast-forward *to*, and guessing a branch would be worse than saying so.
    """
    try:
        return decode_path(run_git(root, "rev-parse", "--verify", "--quiet", "@{upstream}")).strip()
    except GitCommandError:
        return None


def _is_ancestor(root: Path, older: str, newer: str) -> bool:
    """True when `older` is reachable from `newer` — i.e. ff is possible."""
    try:
        run_git(root, "merge-base", "--is-ancestor", older, newer)
    except GitCommandError:
        # Exit 1 means "not an ancestor"; there is no other expected
        # failure for two commits that both resolved a moment ago.
        return False
    return True


def sync_repo(root: Path, *, remote: str) -> SyncOutcome:
    """Bring one working copy in line with its remote, or decline to.

    Clones when `root` does not exist yet. Otherwise fetches and
    fast-forwards, unless doing so would touch work the remote does not
    have — see the module docstring for why that is a hard rule.

    The dirty check happens before the fetch. Fetching is harmless in
    itself, but a sync that cannot merge has nothing to do with the
    objects it would download, and the network round trip is the
    expensive part.

    Args:
        root: Where the working copy lives, per the config.
        remote: Clone url for the repo.

    Returns:
        What happened, or why nothing did.

    Raises:
        NotAGitRepositoryError: `root` exists but is not a repository
            root — the same rule `index` enforces, checked here first so
            sync fails on it rather than clobbering the directory.
        GitCommandError: git could not run, or clone/fetch failed.
    """
    if not root.exists():
        _clone(remote, root)
        return SyncOutcome.CLONED
    ensure_repo_root(root)
    if has_uncommitted_changes(root):
        return SyncOutcome.DIRTY
    run_git(root, "fetch", "--quiet")
    upstream = _upstream(root)
    if upstream is None:
        return SyncOutcome.NO_UPSTREAM
    head = decode_path(run_git(root, "rev-parse", "HEAD")).strip()
    if head == upstream:
        return SyncOutcome.UP_TO_DATE
    if not _is_ancestor(root, head, upstream):
        # The remote does not contain our HEAD: either local commits, or
        # a force-push upstream. Merging or resetting could destroy work,
        # so this is the user's call, not ours.
        return SyncOutcome.DIVERGED
    run_git(root, "merge", "--ff-only", "--quiet", upstream)
    return SyncOutcome.UPDATED


__all__ = ["NotAGitRepositoryError", "SyncOutcome", "sync_repo"]
