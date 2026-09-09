"""Git state for incremental indexing: what changed since the last run.

Second stage of Этап 8 (step 21). `delete_chunks` (step 20) gave the store
the ability to forget; this module answers the question that makes the
ability useful — *which* files changed, so `Pipeline.index` can re-chunk
only those and delete what disappeared. Wiring it into the pipeline is
step 22; nothing here imports the pipeline or the config.

Git-only, by decision: a repo that is not a git repository is a
configuration error with a message, not a silent fall back to a full walk.
One code path, one model of state — a fallback would double both.

Two things live here, and they are deliberately different in kind:

- **The diff** (`diff_since`) is derived: git owns the truth, we only ask.
- **The state** (`IndexState`) is ours: the commit each repo was last
  indexed at. It is a *cache*, not intent — the config says which repos to
  index, this says how far we got. That distinction drives the error
  policy below.

Where the state lives is the caller's business, but the intended home is
`Config.index_dir` — always a local filesystem path, even when the vectors
themselves sit in S3 (`[store] uri`). Per-machine on purpose: two hosts
sharing one S3 index may sit on different branches, and a shared "last
indexed commit" would make each one's diff meaningless. Local state costs
at worst redundant work, which dedup by `chunk_id` absorbs; shared state
would cost silently skipped work, which nothing catches.

Error policy follows the intent/cache split. A config that cannot be
parsed is fatal (see `wsindex.config`) — it is the user's intent and
guessing would index the wrong thing. A state file that cannot be parsed
is *not*: it is a cache, so a corrupt or future-versioned one degrades to
"nothing indexed yet" and the next run rebuilds it. Recoverable by
construction, and it costs only time.

Subprocess hygiene, since every call here shells out to git:

- Argument lists, never `shell=True`: a repo path with a space or a `;`
  is a filename, not a command.
- Plumbing over porcelain where a stable format exists (`rev-parse`,
  `diff --name-status`, `ls-files`). Porcelain output is meant for humans
  and may change between git releases. The one exception is
  `status --porcelain`, which is explicitly documented as the stable
  machine format and is the only way to see untracked files as well as
  modified ones (see `has_uncommitted_changes`).
- `-z` everywhere paths come back: NUL-separated records are immune to
  git's quoting of unusual filenames (`"a\\nb"`), which the default
  output would apply and we would then have to unquote by hand.
- Paths are decoded with `surrogateescape`: git stores bytes, and a
  filename that is not valid UTF-8 must survive the round trip instead of
  crashing the run.
- `GIT_OPTIONAL_LOCKS=0`: every command here is read-only, so none of them
  should take `.git/index.lock` and race a git the user is running.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

STATE_FILE = "state.json"
STATE_VERSION = 1
"""Schema version of `state.json`. A file with an unknown version is
ignored (see `IndexState.load`), which is safe because state is a cache."""

GIT_TIMEOUT = 120.0
"""Seconds any single git command may take before it is killed. A diff
between two commits is near-instant; this only bounds a pathological
case (a network filesystem, a hung filter driver) instead of hanging
the whole index run forever."""


class NotAGitRepositoryError(ValueError):
    """The configured repo path is not the root of a git repository.

    A ValueError subclass on purpose: the CLI already turns ValueError
    into a human error with exit code 1, so this stays catchable
    precisely without every caller learning a new exception type.
    """


class GitCommandError(RuntimeError):
    """A git command ran and exited non-zero."""


class GitUnavailableError(GitCommandError):
    """git could not be run at all: not installed, or it hung.

    Split from its parent so `ensure_repo_root` can tell "git says this
    is not a repository" from "git never got to say anything". Without
    the distinction a machine without git installed would be told to run
    `git init`, which is advice for a different problem entirely.
    """


@dataclass(frozen=True, kw_only=True)
class RepoDiff:
    """What changed in one repo between two commits.

    Paths are repo-relative and POSIX-separated — the same shape as
    `WalkedFile.rel_path`, so step 22 can match them against stored
    chunks without translating.

    A commit-to-commit diff sees *committed* changes only: an edit that
    was never committed, or a brand new file, does not appear in one.
    That is why `has_uncommitted_changes` exists — the caller checks it
    before trusting a diff, and asks for the full listing instead when
    the tree is dirty.

    Attributes:
        changed: Files to re-chunk (added, modified, type-changed, and
            the destination side of a rename or copy).
        deleted: Files whose chunks must go (deleted, and the source
            side of a rename).
        head: The commit the diff ends at; what to record as the new
            state once indexing of this diff succeeded.
        full: True when `changed` is a complete listing of the project
            rather than a delta. The caller cannot infer this from what
            it asked for — passing a `since` that no longer resolves also
            produces a full listing — and the difference matters: after a
            delta only the listed paths may be reconciled, after a full
            listing everything not listed is stale.
    """

    changed: tuple[str, ...]
    deleted: tuple[str, ...]
    head: str
    full: bool


@dataclass(frozen=True, kw_only=True)
class IndexState:
    """The commit each repo was last indexed at.

    Frozen, like the rest of the model: `with_commit` returns a new
    state rather than mutating this one, so a half-finished index run
    cannot leave a partially-updated object behind.

    Attributes:
        commits: Repo id -> commit sha. A repo absent from the mapping
            has never been indexed; that is the normal first-run state,
            not an error.
    """

    commits: dict[str, str]

    @classmethod
    def load(cls, index_dir: Path) -> IndexState:
        """Read `state.json`, or return an empty state if there is none.

        Never raises on a bad file. Missing, unreadable, malformed and
        future-versioned all mean the same thing — "we do not know how
        far we got" — and the honest answer to that is to index
        everything again. Being strict here would turn a corrupted
        cache into a dead workspace.

        Args:
            index_dir: Directory holding `state.json`; normally
                `Config.index_dir`.

        Returns:
            The stored state, or an empty one.
        """
        path = index_dir / STATE_FILE
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # OSError: absent, unreadable, a directory. ValueError:
            # JSONDecodeError (a subclass) — truncated by a crash mid-write.
            return cls(commits={})
        if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
            return cls(commits={})
        commits = raw.get("commits")
        if not isinstance(commits, dict):
            return cls(commits={})
        # Values are re-typed rather than trusted: a hand-edited file
        # could hold a number where a sha belongs.
        return cls(commits={str(k): str(v) for k, v in commits.items()})

    def with_commit(self, repo_id: str, commit: str) -> IndexState:
        """A copy of this state with `repo_id` recorded at `commit`."""
        return IndexState(commits={**self.commits, repo_id: commit})

    def save(self, index_dir: Path) -> Path:
        """Write `state.json`, creating `index_dir` if needed.

        Written to a temporary file and renamed over the target, so a
        crash mid-write leaves the previous state intact instead of a
        truncated file. `Path.replace` is atomic within one filesystem,
        and the temporary sits in the same directory to guarantee that.

        Args:
            index_dir: Directory to write into.

        Returns:
            The path written.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / STATE_FILE
        payload = {"version": STATE_VERSION, "commits": self.commits}
        tmp = path.with_name(f"{STATE_FILE}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path


def run_git(root: Path, *args: str) -> bytes:
    """Run one git command in `root` and return raw stdout.

    The single place in the package that knows how to invoke git safely:
    argv (never a shell), a timeout, the read-only lock hint, and the
    mapping of an environment failure onto a typed error. `git_sync` runs
    its own commands through it for exactly that reason.

    Bytes, not text: paths come back from git as bytes and are decoded
    by the caller with `surrogateescape`, which `subprocess`'s own text
    mode would not do.

    Args:
        root: Working directory for the command.
        *args: Arguments after `git`.

    Returns:
        Raw stdout.

    Raises:
        GitUnavailableError: git is not installed, or it timed out.
        GitCommandError: git ran and exited non-zero.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=GIT_TIMEOUT,
            # Inherit the environment (git needs PATH and HOME to find
            # its config) and add the read-only hint.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except FileNotFoundError as exc:
        raise GitUnavailableError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        timeout = f"git {args[0]} timed out after {GIT_TIMEOUT:.0f}s in {root}"
        raise GitUnavailableError(timeout) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GitCommandError(f"git {args[0]} failed in {root}: {stderr}")
    return completed.stdout


def decode_path(raw: bytes) -> str:
    """Decode a path git handed us, preserving bytes that are not UTF-8."""
    return raw.decode("utf-8", errors="surrogateescape")


def ensure_repo_root(root: Path) -> Path:
    """Check that `root` is the top level of a git repository.

    The *top level*, not merely inside one: `git diff` reports paths
    relative to the repository root, while everything downstream (chunk
    ids, search output) is relative to the configured repo path. Letting
    a subdirectory through would silently produce paths that match
    nothing. Indexing part of a monorepo is a real want, but it needs
    path rebasing and `git diff -- <dir>`; until that exists, saying so
    beats guessing.

    Args:
        root: Configured repository path.

    Returns:
        The resolved repository root (equal to `root`, resolved).

    Raises:
        NotAGitRepositoryError: `root` is not a git repository at all,
            or is a subdirectory of one.
        GitUnavailableError: git could not be run at all.
    """
    if not root.is_dir():
        raise NotAGitRepositoryError(f"not a directory: {root}")
    try:
        raw = run_git(root, "rev-parse", "--show-toplevel")
    except GitUnavailableError:
        # Caught before the parent class below: "git is missing" must not
        # be reported as "this is not a repository".
        raise
    except GitCommandError as exc:
        # `rev-parse` outside a work tree exits 128; that is the only
        # expected failure, and it is the one this function exists to name.
        raise NotAGitRepositoryError(
            f"not a git repository: {root} — wsindex indexes git repos only "
            f"(run `git init` there, or point the config at a real clone)"
        ) from exc
    top_level = Path(decode_path(raw).strip()).resolve()
    if top_level != root.resolve():
        raise NotAGitRepositoryError(
            f"{root} is inside a git repository but is not its root ({top_level}) — "
            f"point the config at the repository root"
        )
    return top_level


def head_commit(root: Path) -> str:
    """The commit `root` currently points at.

    Args:
        root: Repository root (validated with `ensure_repo_root`).

    Returns:
        Full 40-character sha of HEAD.

    Raises:
        NotAGitRepositoryError: `root` is not a repository root.
        GitCommandError: The repository has no commits yet, or git failed.
    """
    ensure_repo_root(root)
    return decode_path(run_git(root, "rev-parse", "HEAD")).strip()


def has_uncommitted_changes(root: Path) -> bool:
    """True when the working tree differs from HEAD in any way.

    Untracked files count. That is the whole point: `walk_repo` indexes
    what is on disk, but `diff_since` can only see what git tracks, so a
    new file that was never committed is invisible to the diff and would
    be missed by an incremental run. The caller checks this first and
    falls back to a full pass when it is True.

    `status --porcelain` rather than a plumbing command because it is the
    one documented-stable format that reports tracked *and* untracked
    changes in a single call; `diff-index` would miss the untracked half.

    Args:
        root: Repository root.

    Returns:
        True if anything is modified, staged, or untracked.

    Raises:
        NotAGitRepositoryError: `root` is not a repository root.
        GitCommandError: git failed.
    """
    ensure_repo_root(root)
    # --untracked-files=normal is the default, but naming it keeps the
    # answer independent of the user's `status.showUntrackedFiles`.
    out = run_git(root, "status", "--porcelain", "-z", "--untracked-files=normal")
    return bool(out.strip(b"\x00"))


def _parse_name_status(out: bytes) -> tuple[list[str], list[str]]:
    """Split `git diff --name-status -z` output into (changed, deleted).

    The `-z` record layout is the subtle part: most entries are a status
    followed by one path, but a rename or copy (`R100`, `C75`) is a status
    followed by *two* — source then destination. Consuming one path for
    every status would desynchronize the whole stream after the first
    rename, quietly turning later paths into statuses.

    Args:
        out: Raw stdout of the diff.

    Returns:
        (changed, deleted) as lists of repo-relative POSIX paths.

    Raises:
        GitCommandError: A status code that cannot appear in a
            commit-to-commit diff (an unmerged or unknown entry), or a
            record that ends mid-way.
    """
    fields = out.split(b"\x00")
    if fields and fields[-1] == b"":
        fields.pop()  # trailing NUL terminates the last record
    changed: list[str] = []
    deleted: list[str] = []
    i = 0
    while i < len(fields):
        status = decode_path(fields[i])
        code = status[:1]
        # Rename/copy carry a similarity score (R100) and two paths.
        needed = 2 if code in ("R", "C") else 1
        # The last path of the record sits at fields[i + needed], so that
        # index must exist; anything else means the stream was cut short.
        if i + needed >= len(fields):
            raise GitCommandError(f"truncated diff record: status {status!r} without its path(s)")
        if code == "D":
            deleted.append(decode_path(fields[i + 1]))
        elif code in ("A", "M", "T"):
            changed.append(decode_path(fields[i + 1]))
        elif code == "R":
            deleted.append(decode_path(fields[i + 1]))
            changed.append(decode_path(fields[i + 2]))
        elif code == "C":
            changed.append(decode_path(fields[i + 2]))
        else:
            # U (unmerged) and X (internal error) cannot occur between two
            # commits; seeing one means our assumptions are wrong, and
            # guessing would corrupt the index.
            raise GitCommandError(f"unexpected diff status {status!r}")
        i += 1 + needed
    return changed, deleted


def diff_since(root: Path, *, since: str | None) -> RepoDiff:
    """What changed in `root` between `since` and HEAD.

    A `since` of None means "never indexed": every file git considers
    part of the project comes back as changed, so the caller has one
    shape to handle instead of branching on first-run versus incremental.

    An unknown `since` (history was rewritten, the commit was garbage
    collected, the state file outlived a re-clone) is not an error either
    — it degrades to the same full listing. Same reasoning as a corrupt
    state file: this is a cache, and the recoverable answer is to index
    everything again.

    Args:
        root: Repository root.
        since: Commit the repo was last indexed at, or None.

    Returns:
        The diff, with `head` set to the current commit.

    Raises:
        NotAGitRepositoryError: `root` is not a repository root.
        GitCommandError: git failed, or returned something unparsable.
    """
    ensure_repo_root(root)
    head = decode_path(run_git(root, "rev-parse", "HEAD")).strip()
    if since is None or not _commit_exists(root, since):
        # `--cached --others --exclude-standard` is "every file git
        # considers part of this project right now": tracked, plus
        # untracked ones that .gitignore does not exclude. Not plain
        # `ls-files`, which would miss a module the user has written but
        # not yet committed — the single most likely thing they want
        # indexed. Not a filesystem walk either, which would happily
        # index build output and local scratch files that .gitignore
        # exists to hide.
        listed = run_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        files = [decode_path(f) for f in listed.split(b"\x00") if f]
        return RepoDiff(changed=tuple(files), deleted=(), head=head, full=True)
    if since == head:
        return RepoDiff(changed=(), deleted=(), head=head, full=False)
    # Two explicit revisions, not `a..b`: the range syntax means something
    # different for `git log`, and spelling both out cannot be misread.
    out = run_git(root, "diff", "--name-status", "-z", since, head)
    changed, deleted = _parse_name_status(out)
    return RepoDiff(changed=tuple(changed), deleted=tuple(deleted), head=head, full=False)


def _commit_exists(root: Path, commit: str) -> bool:
    """True when `commit` still resolves to a commit object in `root`.

    `^{commit}` is the peel syntax: it makes the check fail for a sha
    that exists but is a tree or a blob, which a hand-edited state file
    could hold.
    """
    try:
        run_git(root, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    except GitCommandError:
        return False
    return True
