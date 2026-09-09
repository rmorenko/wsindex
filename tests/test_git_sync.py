"""Sync tests: real clones of a real local remote, no mocking.

The remote is a bare repository in tmp_path and the "origin" side is a
working copy that pushes into it. That is enough to exercise every branch
— clone, fast-forward, divergence — without a network, and it keeps the
tests honest about what git actually does with an upstream.

`sync_repo` is the only code in the project that writes to a working
copy, so the cases that matter most here are the ones where it must
refuse to.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from wsindex.ingest.git_state import GitCommandError, NotAGitRepositoryError
from wsindex.ingest.git_sync import SyncOutcome, sync_repo

GitRunner = Callable[..., str]


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> GitRunner:
    """Run git with a hermetic identity and no user config in the way."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")

    def run(root: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
        return completed.stdout.strip()

    return run


@pytest.fixture
def remote(tmp_path: Path, git: GitRunner) -> Path:
    """A bare repository with one commit, standing in for a real origin."""
    bare = tmp_path / "origin.git"
    bare.mkdir()
    git(bare, "init", "-q", "--bare", "--initial-branch=main")

    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "--initial-branch=main")
    (seed / "a.py").write_text("print('one')\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "first")
    git(seed, "remote", "add", "origin", str(bare))
    git(seed, "push", "-q", "-u", "origin", "main")
    return bare


def push_new_commit(remote: Path, tmp_path: Path, git: GitRunner, text: str) -> str:
    """Advance the remote by one commit; returns its sha."""
    work = tmp_path / f"pusher-{text}"
    git(tmp_path, "clone", "-q", str(remote), str(work))
    (work / "b.py").write_text(f"print('{text}')\n")
    git(work, "add", "-A")
    git(work, "commit", "-qm", text)
    git(work, "push", "-q", "origin", "main")
    return git(work, "rev-parse", "HEAD")


# --- cloning -------------------------------------------------------------


def test_missing_working_copy_is_cloned(tmp_path: Path, remote: Path, git: GitRunner) -> None:
    target = tmp_path / "checkout"
    assert sync_repo(target, remote=str(remote)) == SyncOutcome.CLONED
    assert (target / "a.py").read_text() == "print('one')\n"


def test_clone_creates_missing_parent_directories(
    tmp_path: Path, remote: Path, git: GitRunner
) -> None:
    target = tmp_path / "deep" / "nested" / "checkout"
    assert sync_repo(target, remote=str(remote)) == SyncOutcome.CLONED
    assert (target / "a.py").is_file()


def test_clone_of_a_bad_remote_raises(tmp_path: Path, git: GitRunner) -> None:
    with pytest.raises(GitCommandError):
        sync_repo(tmp_path / "checkout", remote=str(tmp_path / "no-such-remote"))


def test_clone_keeps_history_for_incremental_indexing(
    tmp_path: Path, remote: Path, git: GitRunner
) -> None:
    # Not a shallow clone: `diff_since` needs to resolve the commit the
    # last index run recorded, and a shallow boundary would eventually
    # swallow it, silently turning every run into a full pass.
    push_new_commit(remote, tmp_path, git, "second")
    target = tmp_path / "checkout"
    sync_repo(target, remote=str(remote))
    assert not (target / ".git" / "shallow").exists()
    assert len(git(target, "log", "--oneline").splitlines()) == 2


# --- fast-forward --------------------------------------------------------


def test_up_to_date_working_copy_reports_so(tmp_path: Path, remote: Path, git: GitRunner) -> None:
    target = tmp_path / "checkout"
    sync_repo(target, remote=str(remote))
    assert sync_repo(target, remote=str(remote)) == SyncOutcome.UP_TO_DATE


def test_new_upstream_commit_fast_forwards(tmp_path: Path, remote: Path, git: GitRunner) -> None:
    target = tmp_path / "checkout"
    sync_repo(target, remote=str(remote))
    pushed = push_new_commit(remote, tmp_path, git, "second")

    assert sync_repo(target, remote=str(remote)) == SyncOutcome.UPDATED
    assert git(target, "rev-parse", "HEAD") == pushed
    assert (target / "b.py").is_file()


# --- the refusals, which are the point -----------------------------------


def test_uncommitted_changes_are_left_alone(tmp_path: Path, remote: Path, git: GitRunner) -> None:
    target = tmp_path / "checkout"
    sync_repo(target, remote=str(remote))
    push_new_commit(remote, tmp_path, git, "second")
    (target / "a.py").write_text("print('my own work')\n")

    assert sync_repo(target, remote=str(remote)) == SyncOutcome.DIRTY
    # Untouched: the edit survives and HEAD did not move.
    assert (target / "a.py").read_text() == "print('my own work')\n"
    assert not (target / "b.py").exists()


def test_untracked_file_also_blocks_the_sync(tmp_path: Path, remote: Path, git: GitRunner) -> None:
    target = tmp_path / "checkout"
    sync_repo(target, remote=str(remote))
    (target / "scratch.py").write_text("print('scratch')\n")
    assert sync_repo(target, remote=str(remote)) == SyncOutcome.DIRTY


def test_local_commits_are_not_discarded(tmp_path: Path, remote: Path, git: GitRunner) -> None:
    # A fast-forward is impossible, and anything else would throw away a
    # commit the remote has never seen. Declining is the whole policy.
    target = tmp_path / "checkout"
    sync_repo(target, remote=str(remote))
    (target / "mine.py").write_text("print('mine')\n")
    git(target, "add", "-A")
    git(target, "commit", "-qm", "local work")
    mine = git(target, "rev-parse", "HEAD")
    push_new_commit(remote, tmp_path, git, "second")

    assert sync_repo(target, remote=str(remote)) == SyncOutcome.DIVERGED
    assert git(target, "rev-parse", "HEAD") == mine
    assert (target / "mine.py").is_file()


def test_branch_without_an_upstream_is_skipped(
    tmp_path: Path, remote: Path, git: GitRunner
) -> None:
    target = tmp_path / "checkout"
    sync_repo(target, remote=str(remote))
    git(target, "checkout", "-q", "-b", "local-only")
    assert sync_repo(target, remote=str(remote)) == SyncOutcome.NO_UPSTREAM


def test_detached_head_is_skipped(tmp_path: Path, remote: Path, git: GitRunner) -> None:
    target = tmp_path / "checkout"
    sync_repo(target, remote=str(remote))
    git(target, "checkout", "-q", "--detach", "HEAD")
    assert sync_repo(target, remote=str(remote)) == SyncOutcome.NO_UPSTREAM


def test_existing_non_git_directory_is_rejected(
    tmp_path: Path, remote: Path, git: GitRunner
) -> None:
    # The directory is there but is not a checkout. Cloning over it would
    # be destructive, so this is the same config error `index` reports.
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "important.txt").write_text("do not clobber me\n")

    with pytest.raises(NotAGitRepositoryError):
        sync_repo(plain, remote=str(remote))
    assert (plain / "important.txt").read_text() == "do not clobber me\n"


# --- outcome labels ------------------------------------------------------


def test_skip_outcomes_are_flagged_as_skips() -> None:
    assert SyncOutcome.DIRTY.is_skip
    assert SyncOutcome.DIVERGED.is_skip
    assert SyncOutcome.NO_UPSTREAM.is_skip
    assert not SyncOutcome.CLONED.is_skip
    assert not SyncOutcome.UPDATED.is_skip
    assert not SyncOutcome.UP_TO_DATE.is_skip
