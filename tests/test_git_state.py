"""Git-layer tests: real repositories on disk, no mocking of subprocess.

Mocking git here would test our idea of git's output rather than git's
output — and the whole point of the module is that the `-z` record layout
and the rename statuses are easy to get wrong. So every test builds an
actual repository in tmp_path and drives real commands.

The `git` fixture pins identity and disables signing/hooks/templates, so
the suite is hermetic: it cannot pick up the developer's `user.email`,
their `commit.gpgsign`, or an `init.templateDir` full of hooks.
"""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from wsindex.ingest.git_state import (
    STATE_FILE,
    STATE_VERSION,
    GitCommandError,
    GitUnavailableError,
    IndexState,
    NotAGitRepositoryError,
    _parse_name_status,
    diff_since,
    ensure_repo_root,
    has_uncommitted_changes,
    head_commit,
)

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
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    return run


@pytest.fixture
def repo(tmp_path: Path, git: GitRunner) -> Path:
    """A repository with one commit containing `a.py` and `README.md`."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "--initial-branch=main")
    (root / "a.py").write_text("print('a')\n")
    (root / "README.md").write_text("# Title\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "first")
    return root


def commit_all(git: GitRunner, root: Path, message: str) -> str:
    """Stage everything and commit; returns the new HEAD sha."""
    git(root, "add", "-A")
    git(root, "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD")


# --- guards: git-only, and the root specifically -------------------------


def test_plain_directory_is_rejected(tmp_path: Path, git: GitRunner) -> None:
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    with pytest.raises(NotAGitRepositoryError, match="not a git repository"):
        ensure_repo_root(plain)


def test_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(NotAGitRepositoryError, match="not a directory"):
        ensure_repo_root(tmp_path / "nowhere")


def test_subdirectory_of_a_repo_is_rejected(repo: Path, git: GitRunner) -> None:
    # The trap this guard exists for: `git diff` inside src/ reports paths
    # relative to the repo root, so they would match no chunk we stored.
    sub = repo / "src"
    sub.mkdir()
    with pytest.raises(NotAGitRepositoryError, match="is not its root"):
        ensure_repo_root(sub)


def test_repo_root_is_accepted(repo: Path, git: GitRunner) -> None:
    assert ensure_repo_root(repo) == repo.resolve()


def test_head_commit_is_the_full_sha(repo: Path, git: GitRunner) -> None:
    head = head_commit(repo)
    assert head == git(repo, "rev-parse", "HEAD")
    assert len(head) == 40


# --- diff: first run, no-op run, and every status code -------------------


def test_first_run_lists_every_project_file(repo: Path, git: GitRunner) -> None:
    diff = diff_since(repo, since=None)
    assert sorted(diff.changed) == ["README.md", "a.py"]
    assert diff.deleted == ()
    assert diff.head == git(repo, "rev-parse", "HEAD")


def test_untracked_file_is_in_the_first_listing(repo: Path, git: GitRunner) -> None:
    # A module written but not yet committed is the single most likely
    # thing a developer wants indexed, so the full listing includes it.
    (repo / "fresh.py").write_text("print('fresh')\n")
    assert "fresh.py" in diff_since(repo, since=None).changed


def test_gitignored_file_is_not_in_the_first_listing(repo: Path, git: GitRunner) -> None:
    # The other half of --others --exclude-standard: build output and
    # local scratch files stay out, which a filesystem walk could not do.
    (repo / ".gitignore").write_text("secret.py\n")
    (repo / "secret.py").write_text("print('secret')\n")
    changed = diff_since(repo, since=None).changed
    assert "secret.py" not in changed
    assert ".gitignore" in changed


def test_same_commit_yields_an_empty_diff(repo: Path, git: GitRunner) -> None:
    head = git(repo, "rev-parse", "HEAD")
    diff = diff_since(repo, since=head)
    assert diff.changed == ()
    assert diff.deleted == ()
    assert diff.head == head


def test_added_and_modified_are_changed(repo: Path, git: GitRunner) -> None:
    before = git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("print('a2')\n")
    (repo / "b.py").write_text("print('b')\n")
    commit_all(git, repo, "second")

    diff = diff_since(repo, since=before)
    assert sorted(diff.changed) == ["a.py", "b.py"]
    assert diff.deleted == ()


def test_deleted_file_is_reported_as_deleted(repo: Path, git: GitRunner) -> None:
    before = git(repo, "rev-parse", "HEAD")
    (repo / "a.py").unlink()
    commit_all(git, repo, "drop a")

    diff = diff_since(repo, since=before)
    assert diff.deleted == ("a.py",)
    assert diff.changed == ()


def test_rename_deletes_the_old_path_and_changes_the_new(repo: Path, git: GitRunner) -> None:
    # The record that desynchronizes a naive parser: `R100` carries TWO
    # paths, and both sides matter to us — old chunks must go, new ones
    # must be written.
    before = git(repo, "rev-parse", "HEAD")
    git(repo, "mv", "a.py", "renamed.py")
    commit_all(git, repo, "rename")

    diff = diff_since(repo, since=before)
    assert diff.deleted == ("a.py",)
    assert diff.changed == ("renamed.py",)


def test_rename_followed_by_another_file_stays_in_sync(repo: Path, git: GitRunner) -> None:
    # Regression for the off-by-one a rename causes: if the parser
    # consumed one path for R100, the NEXT status would be read from a
    # path field and the whole tail would be garbage.
    before = git(repo, "rev-parse", "HEAD")
    git(repo, "mv", "a.py", "renamed.py")
    (repo / "zzz.py").write_text("print('z')\n")
    commit_all(git, repo, "rename plus add")

    diff = diff_since(repo, since=before)
    assert diff.deleted == ("a.py",)
    assert sorted(diff.changed) == ["renamed.py", "zzz.py"]


def test_path_with_spaces_and_quotes_survives(repo: Path, git: GitRunner) -> None:
    # Without -z git would hand back a quoted, escaped name and every
    # path here would need unquoting.
    before = git(repo, "rev-parse", "HEAD")
    weird = 'weird "name" with spaces.py'
    (repo / weird).write_text("print('w')\n")
    commit_all(git, repo, "weird name")

    assert diff_since(repo, since=before).changed == (weird,)


def test_path_with_a_newline_survives(repo: Path, git: GitRunner) -> None:
    # The case a line-oriented parser cannot get right at all.
    before = git(repo, "rev-parse", "HEAD")
    weird = "two\nlines.py"
    (repo / weird).write_text("print('n')\n")
    commit_all(git, repo, "newline name")

    assert diff_since(repo, since=before).changed == (weird,)


def test_unknown_since_degrades_to_a_full_listing(repo: Path, git: GitRunner) -> None:
    # History rewritten, repo re-cloned, commit gc'd: the state is a
    # cache, so the answer is "index everything", not an exception.
    absent = "0" * 40
    diff = diff_since(repo, since=absent)
    assert sorted(diff.changed) == ["README.md", "a.py"]
    assert diff.deleted == ()


def test_since_pointing_at_a_tree_degrades_too(repo: Path, git: GitRunner) -> None:
    # A sha that exists but is not a commit — what a hand-edited state
    # file could hold. `^{commit}` peeling is what catches it.
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    assert sorted(diff_since(repo, since=tree).changed) == ["README.md", "a.py"]


def test_diff_on_a_plain_directory_is_rejected(tmp_path: Path, git: GitRunner) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(NotAGitRepositoryError):
        diff_since(plain, since=None)


# --- when git itself will not run ----------------------------------------
#
# The only tests here that patch subprocess: the subject is our mapping of
# an environment failure onto a typed error, and neither a missing git nor
# a hung one can be produced honestly inside a test run.


def test_missing_git_binary_becomes_a_typed_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr("wsindex.ingest.git_state.subprocess.run", explode)
    with pytest.raises(GitUnavailableError, match="not installed"):
        head_commit(repo)


def test_git_timeout_becomes_a_typed_error(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=1.0)

    monkeypatch.setattr("wsindex.ingest.git_state.subprocess.run", hang)
    with pytest.raises(GitUnavailableError, match="timed out"):
        head_commit(repo)


# --- the -z record parser, driven directly -------------------------------


def test_parser_handles_a_record_without_its_path() -> None:
    with pytest.raises(GitCommandError, match="truncated"):
        _parse_name_status(b"M\x00")


def test_parser_rejects_an_unmerged_status() -> None:
    # U cannot occur between two commits; guessing would corrupt the index.
    with pytest.raises(GitCommandError, match="unexpected diff status"):
        _parse_name_status(b"U\x00conflict.py\x00")


def test_parser_treats_a_copy_as_a_new_file_only() -> None:
    # C75 carries two paths like a rename, but the source still exists —
    # only the destination is new.
    changed, deleted = _parse_name_status(b"C75\x00src.py\x00copy.py\x00")
    assert changed == ["copy.py"]
    assert deleted == []


def test_parser_handles_a_type_change() -> None:
    changed, deleted = _parse_name_status(b"T\x00link.py\x00")
    assert changed == ["link.py"]
    assert deleted == []


def test_parser_accepts_empty_output() -> None:
    assert _parse_name_status(b"") == ([], [])


# --- uncommitted changes -------------------------------------------------


def test_clean_tree_has_no_uncommitted_changes(repo: Path, git: GitRunner) -> None:
    assert not has_uncommitted_changes(repo)


def test_modified_file_counts_as_uncommitted(repo: Path, git: GitRunner) -> None:
    (repo / "a.py").write_text("print('edited')\n")
    assert has_uncommitted_changes(repo)


def test_untracked_file_counts_as_uncommitted(repo: Path, git: GitRunner) -> None:
    # The reason this helper reaches for `status --porcelain` instead of
    # a plumbing diff: an untracked file is invisible to diff-index, but
    # walk_repo would index it.
    (repo / "new.py").write_text("print('new')\n")
    assert has_uncommitted_changes(repo)


def test_staged_file_counts_as_uncommitted(repo: Path, git: GitRunner) -> None:
    (repo / "staged.py").write_text("print('s')\n")
    git(repo, "add", "staged.py")
    assert has_uncommitted_changes(repo)


# --- IndexState: a cache, so nothing about it is fatal -------------------


def test_state_roundtrip(tmp_path: Path) -> None:
    state = IndexState(commits={}).with_commit("repo1", "abc123")
    path = state.save(tmp_path)
    assert IndexState.load(tmp_path).commits == {"repo1": "abc123"}
    assert path == tmp_path / STATE_FILE


def test_save_creates_the_index_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "index"
    IndexState(commits={"r": "sha"}).save(target)
    assert IndexState.load(target).commits == {"r": "sha"}


def test_with_commit_does_not_mutate_the_original() -> None:
    first = IndexState(commits={"repo1": "one"})
    second = first.with_commit("repo2", "two")
    assert first.commits == {"repo1": "one"}
    assert second.commits == {"repo1": "one", "repo2": "two"}


def test_with_commit_overwrites_an_existing_repo() -> None:
    state = IndexState(commits={"repo1": "old"}).with_commit("repo1", "new")
    assert state.commits == {"repo1": "new"}


def test_missing_state_file_is_an_empty_state(tmp_path: Path) -> None:
    assert IndexState.load(tmp_path).commits == {}


def test_corrupt_state_file_is_an_empty_state(tmp_path: Path) -> None:
    # A crash mid-write leaves truncated JSON; that must cost a full
    # reindex, not a dead workspace.
    (tmp_path / STATE_FILE).write_text('{"version": 1, "commi')
    assert IndexState.load(tmp_path).commits == {}


def test_future_version_is_an_empty_state(tmp_path: Path) -> None:
    payload = {"version": STATE_VERSION + 1, "commits": {"repo1": "abc"}}
    (tmp_path / STATE_FILE).write_text(json.dumps(payload))
    assert IndexState.load(tmp_path).commits == {}


def test_state_file_holding_a_list_is_an_empty_state(tmp_path: Path) -> None:
    (tmp_path / STATE_FILE).write_text("[1, 2, 3]")
    assert IndexState.load(tmp_path).commits == {}


def test_state_with_a_non_mapping_commits_key_is_an_empty_state(tmp_path: Path) -> None:
    # Right version, wrong shape underneath — the version check alone
    # would wave this through and `.items()` would blow up at load time.
    payload = {"version": STATE_VERSION, "commits": "not-a-mapping"}
    (tmp_path / STATE_FILE).write_text(json.dumps(payload))
    assert IndexState.load(tmp_path).commits == {}


def test_state_with_non_string_values_is_coerced(tmp_path: Path) -> None:
    payload = {"version": STATE_VERSION, "commits": {"repo1": 42}}
    (tmp_path / STATE_FILE).write_text(json.dumps(payload))
    assert IndexState.load(tmp_path).commits == {"repo1": "42"}


def test_save_leaves_no_temporary_behind(tmp_path: Path) -> None:
    IndexState(commits={"r": "sha"}).save(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [STATE_FILE]


def test_save_over_an_existing_state_replaces_it(tmp_path: Path) -> None:
    IndexState(commits={"repo1": "one"}).save(tmp_path)
    IndexState(commits={"repo2": "two"}).save(tmp_path)
    assert IndexState.load(tmp_path).commits == {"repo2": "two"}


# --- the loop step 22 will run: index, record, ask again ------------------


def test_state_drives_the_next_diff(repo: Path, git: GitRunner, tmp_path: Path) -> None:
    index_dir = tmp_path / "index"

    first = diff_since(repo, since=IndexState.load(index_dir).commits.get("repo1"))
    assert sorted(first.changed) == ["README.md", "a.py"]
    IndexState.load(index_dir).with_commit("repo1", first.head).save(index_dir)

    # Nothing happened in between: the second run must find no work.
    second = diff_since(repo, since=IndexState.load(index_dir).commits.get("repo1"))
    assert second.changed == ()
    assert second.deleted == ()

    # One file changes: only that file comes back.
    (repo / "a.py").write_text("print('changed')\n")
    commit_all(git, repo, "third")
    third = diff_since(repo, since=IndexState.load(index_dir).commits.get("repo1"))
    assert third.changed == ("a.py",)
    assert third.head != first.head
