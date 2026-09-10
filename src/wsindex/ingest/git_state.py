"""Git state for incremental indexing: what changed since the last run.

Two things live here, deliberately different in kind:

- **The diff** (`diff_since`) is derived: git owns the truth, we only ask.
- **The state** (`IndexState`) is ours — the commit each repo was last
  indexed at, plus a fingerprint of the markup that produced it. It is a
  *cache*, not intent, which is why a missing or unreadable one costs a
  full pass rather than an error.

Git-only, by decision: a repo that is not a git repository is a
configuration error with a message, not a silent fall back to a full
walk. One code path, one model of state.

Nothing here imports the pipeline or the config.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from wsindex.paths import make_index_dir

STATE_FILE = "state.json"
STATE_VERSION = 1
"""Schema version of `state.json`. A file with an unknown version is
ignored (see `IndexState.load`), which is safe because state is a cache."""

_SHA = re.compile(r"[0-9a-f]{40}")
"""What every commit this package records looks like: `git rev-parse
HEAD` output, full and lowercase. Used to check values read back from
disk before they are spoken to git as revisions."""


def is_sha(value: str) -> bool:
    """True when `value` is a full commit sha and nothing else.

    Deliberately narrow. Abbreviated shas, `HEAD~2` and branch names are
    all things git would accept and none of them are things this package
    writes, so accepting them would only widen what a state file can say.
    """
    return _SHA.fullmatch(value) is not None


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
    `WalkedFile.rel_path`, so the pipeline can match them against stored
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
        since: The commit the diff began at, or None when it is a full
            listing. The counterpart to `head`: a caller that wants the
            commits gained in this range needs both ends, and only
            `diff_since` knows which `since` was actually usable.
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
    since: str | None
    head: str
    full: bool


@dataclass(frozen=True, kw_only=True)
class IndexState:
    """The commit each repo was last indexed at, and under which markup.

    Frozen, like the rest of the model: `with_commit` returns a new
    state rather than mutating this one, so a half-finished index run
    cannot leave a partially-updated object behind.

    Attributes:
        commits: Repo id -> commit sha. A repo absent from the mapping
            has never been indexed; that is the normal first-run state,
            not an error.
        markup: Repo id -> a fingerprint of the per-repo `ignore` and
            `formats` in force at that commit. A commit alone does not
            say what was indexed: change `formats` and the same tree
            yields different files, while git reports nothing changed.
            Without this, editing the markup and re-indexing did exactly
            nothing — measured, not imagined. A repo absent here is one
            whose markup is unknown, which costs one full pass.
        lost: A state file was there and could not be used. Empty
            `commits` alone cannot say that: a first index and a
            corrupted cache produce the same mapping and cost the same
            full pass, but only one of them is worth telling somebody
            about, and the note used to name three causes that were all
            wrong in that case.
    """

    commits: dict[str, str]
    markup: dict[str, str] = field(default_factory=dict)
    lost: bool = False

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
        present = path.exists()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # OSError: absent, unreadable, a directory. ValueError:
            # JSONDecodeError (a subclass) — truncated by a crash mid-write.
            # `present` is what tells the first of those from the rest.
            return cls(commits={}, lost=present)
        if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
            return cls(commits={}, lost=True)
        commits = raw.get("commits")
        if not isinstance(commits, dict):
            return cls(commits={}, lost=True)
        markup = raw.get("markup")
        # Values are re-typed rather than trusted: a hand-edited file
        # could hold a number where a sha belongs. Shas are also checked
        # for *shape*, because from here they go into a git command line
        # as revisions — and `git log --output=…..HEAD` would be read as
        # an option, not a range. Nothing writes such a value today; the
        # check is what makes that stay true. A rejected entry looks
        # exactly like an unknown commit, which this file already
        # degrades to a full pass for.
        return cls(
            commits={str(k): str(v) for k, v in commits.items() if is_sha(str(v))},
            markup=(
                {str(k): str(v) for k, v in markup.items()} if isinstance(markup, dict) else {}
            ),
        )

    def with_commit(self, repo_id: str, commit: str, *, markup: str = "") -> IndexState:
        """A copy of this state with `repo_id` recorded at `commit`.

        Args:
            repo_id: The repo just indexed.
            commit: The commit its working tree was at.
            markup: Fingerprint of the `ignore`/`formats` used, so the
                next run can tell a changed policy from an unchanged tree.
        """
        return IndexState(
            commits={**self.commits, repo_id: commit},
            markup={**self.markup, repo_id: markup},
        )

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
        make_index_dir(index_dir)
        path = index_dir / STATE_FILE
        payload = {"version": STATE_VERSION, "commits": self.commits, "markup": self.markup}
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

    Untracked files count. That is the whole point: the full listing
    includes them (`ls-files --others`), but a commit-to-commit diff can
    only see what git tracks, so a new file that was never committed is
    invisible to the diff and would be missed by an incremental run.
    The caller checks this first and falls back to a full pass when it
    is True.

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
        return RepoDiff(changed=tuple(files), deleted=(), since=None, head=head, full=True)
    if since == head:
        return RepoDiff(changed=(), deleted=(), since=since, head=head, full=False)
    # Two explicit revisions, not `a..b`: the range syntax means something
    # different for `git log`, and spelling both out cannot be misread.
    out = run_git(root, "diff", "--name-status", "-z", since, head)
    changed, deleted = _parse_name_status(out)
    return RepoDiff(
        changed=tuple(changed), deleted=tuple(deleted), since=since, head=head, full=False
    )


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
