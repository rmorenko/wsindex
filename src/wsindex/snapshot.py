"""Materializing external documents into a git snapshot repository.

What a connector fetches does not go into the store. It is written to
disk as markdown, in a directory that is itself a git repository, and
every sync is one commit. The store then learns about it the ordinary
way — the snapshot is a `[[repos]]` entry like any other.

That indirection buys four things at once, and it is why the fetched
text never touches the store directly:

- **The git-only invariant survives.** Everything wsindex indexes is a
  git working copy, so incremental indexing works here without a single
  change: a document that did not change
  produces no commit, so `diff_since` reports nothing and the sync costs
  no embeddings.
- **`file:line` stays honest.** A hit points at a line of a file that
  exists, in a snapshot that can be opened and read. The source's url is
  in the frontmatter, so the live document is one hop away, but the
  citation is not a claim about a page that may have changed since.
- **The git log of the snapshot is the history of the source.** A wiki
  that keeps no readable history, or a tracker whose history is a list
  of field changes, becomes `git log -p`.
- **It is reversible.** The snapshot is files; deleting it costs
  nothing, and nothing else in the workspace has to know it existed.

The probe behind the third point
--------------------------------
"Every sync is a commit" is worth nothing if an unchanged document comes
back different — the log would then record noise at whatever interval
the sync runs, which is worse than no history at all. So it was
measured: four real documents fetched twice, several seconds apart — a
GitHub issue, a raw markdown file, and two HTML pages. All four came
back byte-identical, text and metadata alike.

So stability is a property of the *sources*, and the one thing that
could break it is us. That is why the frontmatter carries no fetch
timestamp: a `fetched_at` field would make every sync a diff, turn the
log into a heartbeat, and re-embed every document on every run. What is
recorded is what the source said about the document, and nothing about
the act of fetching it.

Paths are derived, not stored
-----------------------------
A url maps to a file path by a pure function (`document_path`), so the
set of files a config *should* produce is known without touching the
network. That is what makes pruning safe: a document dropped from the
config is deleted, while a document whose fetch failed this run keeps
its file. A network blip must not read as "the page was deleted" in a
history whose whole purpose is to say when things changed.

A url is external input that decides a filename, so `document_path`
defends itself — see its docstring for what the probe found there.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import blake2b
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from wsindex.connectors import ConnectorError, ConnectorSpec, Document, route
from wsindex.ingest.git_state import GitCommandError, decode_path, run_git

MAX_SEGMENT_BYTES = 100
"""Longest path segment written, before a disambiguating hash is added.
Filesystems cap a name at 255 bytes; the probe found a real 300-character
url, so this is not hypothetical."""

_UNSAFE = re.compile(r"[^\w.\-]", re.UNICODE)
"""Everything a path segment may not contain. `\\w` is unicode-aware, so
a Cyrillic or CJK page name survives intact — only punctuation that
means something to a shell, a filesystem or a url is replaced."""

_COMMIT_IDENTITY = ("-c", "user.name=wsindex", "-c", "user.email=wsindex@localhost")
"""Who authors a snapshot commit. Not the person running sync: they did
not write the wiki, and attributing it to them would poison `git blame`
on the one repository where blame answers "who edited the source". It
also means a snapshot can be committed on a machine with no git identity
configured at all."""


@dataclass(frozen=True, kw_only=True)
class SnapshotReport:
    """What one materialization pass did.

    Attributes:
        added: Documents written that had no file before.
        updated: Documents whose file changed.
        unchanged: Documents that came back identical — the common case,
            and the reason a sync is usually free downstream.
        removed: Files deleted because no configured url produces them.
        failed: `(url, reason)` for each document that could not be
            fetched. Reported rather than raised: one unreachable page
            must not strand the rest of the snapshot.
        commit: The commit this pass created, or None when nothing
            changed. None is the expected outcome of a repeat sync, and
            the only honest way to ask whether anything moved — the
            counters above describe intent, git describes the tree.
    """

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: tuple[tuple[str, str], ...] = ()
    commit: str | None = None

    def summary(self) -> str:
        """One line for a terminal: what moved, or that nothing did."""
        parts = [
            f"{self.added} added" if self.added else "",
            f"{self.updated} updated" if self.updated else "",
            f"{self.removed} removed" if self.removed else "",
        ]
        moved = ", ".join(part for part in parts if part)
        if not moved:
            moved = f"up to date ({self.unchanged} documents)"
        if self.failed:
            moved += f"; {len(self.failed)} failed"
        return moved


def _segment(raw: str) -> str:
    """One url component as one safe path segment, or empty to drop it.

    Empty for `.` and `..`, which is how traversal is handled: the
    segment simply does not exist, so there is no path to escape with.
    """
    cleaned = _UNSAFE.sub("-", raw).strip("-.")
    if not cleaned:
        return ""
    encoded = cleaned.encode("utf-8")
    if len(encoded) > MAX_SEGMENT_BYTES:
        # Truncate by bytes, not characters — a filesystem's limit is in
        # bytes, and a name of CJK characters is three times its length.
        # The hash is of the original, so two urls that share a prefix
        # stay two files.
        head = encoded[:MAX_SEGMENT_BYTES].decode("utf-8", errors="ignore")
        cleaned = f"{head}-{blake2b(raw.encode('utf-8'), digest_size=4).hexdigest()}"
    return cleaned


def document_path(url: str) -> PurePosixPath:
    """Where a url's document lives inside the snapshot.

    A pure function of the url, which is what lets the snapshot be
    reconciled without the network (see the module docstring). The shape
    mirrors the source — `github.com/org/repo/issues/7.md` — so the
    directory tree reads like the site it came from and the doc chunker
    gets a path worth showing in a search hit.

    Four defences, each answering something measured on real and hostile
    urls:

    - **Traversal.** `https://host/../../etc/passwd` and its
      percent-encoded twin both reach `_segment`, which drops `..` and
      turns `%2e%2e` into an ordinary name. Nothing is percent-decoded on
      the way, so there is no second decoding pass to be tricked.
    - **Credentials.** `hostname` rather than `netloc`: a url of the form
      `https://user:token@host/page` would otherwise write the token into
      a directory name — and this directory is a git repository someone
      may push.
    - **Collisions.** The query string is part of the name.
      `wiki?page=Home` and `wiki?page=Other` are two documents, and
      dropping the query would have quietly made them one file.
    - **Length.** A 300-character segment is truncated with a hash of the
      original appended, so it stays under the filesystem's limit and
      stays unique.

    The fragment is dropped on purpose: `#section` names a place inside a
    document, not another document.

    Args:
        url: The document's url, as configured.

    Returns:
        A relative POSIX path ending in `.md`.
    """
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        # A malformed port is not worth an exception here: the url still
        # names a document, and the host segment is only a directory name.
        port = None
    host = (parts.hostname or "").lower()
    host_segment = _segment(f"{host}-{port}" if port else host) or "unknown-host"

    segments = [safe for safe in (_segment(part) for part in parts.path.split("/")) if safe]
    if parts.query:
        query = _segment(parts.query)
        if query:
            # Appended to the last segment rather than made its own
            # directory: `wiki?page=Home` is one document named by the
            # whole thing, not a `wiki/` containing a `page=Home`.
            segments[-1:] = [f"{segments[-1]}-{query}" if segments else query]
    if not segments or parts.path.endswith("/"):
        # A directory-shaped url still names a page. `host/uv/` becomes
        # `uv/index.md`, which leaves `host/uv` free to be `uv.md` — two
        # urls, two files, no collision.
        segments.append("index")

    name = segments[-1]
    leaf = name if name.endswith(".md") else f"{name}.md"
    return PurePosixPath(host_segment, *segments[:-1], leaf)


def render(document: Document) -> str:
    """One fetched document as the markdown that gets committed.

    Frontmatter, then the title as an H1, then the body:

        ---
        url: "https://github.com/org/repo/issues/7"
        title: "Add basic CI"
        source: "github"
        ---

        # Add basic CI

        Run tests, formatting, linting.

    **Frontmatter, not a sidecar file.** The metadata belongs to the
    document, and a second file has to be kept in step with the first —
    two paths to derive, two things to prune, and a diff that shows half
    the change. In frontmatter it travels with the text through every
    tool that already handles the snapshot: git shows a changed title as
    a changed line, and the chunker indexes the url along with the
    preamble, so searching for the url finds the document.

    Values are JSON-quoted, which is also valid YAML — a title
    containing a colon or a quote is the normal case, not the exception.

    The title is repeated as an H1 because the doc chunker splits on
    headings: without it a body with no headings is one unnamed blob, and
    with it every document has at least one named section.

    Line endings are normalized to `\\n`. A real GitHub issue body turned
    out to carry `\\r\\n` — one line of it, enough to give the file two
    kinds of ending, since the frontmatter this builds has only one.
    Sources are not consistent about this and a snapshot has to be, or
    every reader downstream inherits the inconsistency.

    Args:
        document: What the connector returned.

    Returns:
        File contents, ending in a newline.
    """
    front: dict[str, str] = {"url": document.url}
    if document.title:
        front["title"] = document.title
    # Sorted, so a source that reorders its JSON does not produce a diff.
    front.update(dict(sorted(document.metadata.items())))

    lines = ["---"]
    lines += [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in front.items()]
    lines += ["---", ""]
    body = document.text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if document.title and not body.startswith(f"# {document.title}"):
        lines += [f"# {document.title}", ""]
    lines.append(body)
    return "\n".join(lines).rstrip("\n") + "\n"


def _ensure_repo(root: Path) -> None:
    """Make `root` a git repository, or confirm it already is one.

    Unlike every other path in the config, this one is the tool's to
    create: a snapshot repository has no remote to clone from and no
    working copy the user maintains. It is still not the tool's to *take
    over*, so a non-empty directory that is not a repository is refused
    rather than initialized around whatever is in it.

    Raises:
        ValueError: `root` holds files but is not a git repository.
    """
    if (root / ".git").exists():
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"{root} is not a git repository but is not empty — refusing to take it over"
        )
    root.mkdir(parents=True, exist_ok=True)
    run_git(root, "init", "--quiet")


def _tracked(root: Path) -> set[PurePosixPath]:
    """Files git already has in the snapshot.

    Tracked only: something untracked in the directory was not written by
    a previous sync, and deleting it would be deleting someone's file
    with no way to get it back.
    """
    raw = run_git(root, "ls-files", "-z")
    return {PurePosixPath(decode_path(name)) for name in raw.split(b"\0") if name}


def _commit(root: Path, message: str, paths: Sequence[PurePosixPath]) -> str | None:
    """Commit the named paths, or return None when they hold no change.

    Named paths rather than `add -A .`, which would sweep in anything
    else in the directory. That is the difference between "untracked
    files are never deleted" holding for one run and holding always: a
    stray file swept into a commit becomes tracked, and the next run
    prunes it as a document no url produces.

    Every expected path is staged, not only the ones this pass rewrote,
    so a run interrupted between writing and committing is finished by
    the next one instead of leaving the snapshot permanently dirty — a
    dirty tree costs `index` a full pass on every run afterwards.

    Args:
        root: The snapshot repository.
        message: Commit subject.
        paths: Repo-relative paths to stage; those that neither exist nor
            are tracked are dropped, because git rejects a pathspec that
            matches nothing.

    Returns:
        The new commit's sha, or None when nothing was staged — what a
        sync of unchanged documents produces.
    """
    if not paths:
        return None
    run_git(root, "add", "-A", "--", *(str(path) for path in paths))
    try:
        # Exit 1 means "there are staged changes"; the inverted spelling
        # is git's, and the exception is how run_git reports exit 1.
        run_git(root, "diff", "--cached", "--quiet")
    except GitCommandError:
        run_git(root, *_COMMIT_IDENTITY, "commit", "--quiet", "-m", message)
        return decode_path(run_git(root, "rev-parse", "HEAD")).strip()
    return None


def materialize(root: Path, *, urls: list[str], specs: list[ConnectorSpec]) -> SnapshotReport:
    """Fetch every configured url into the snapshot at `root` and commit.

    The whole pass is reconciliation, not accumulation: what the config
    names is what the snapshot contains. Documents no longer configured
    are deleted, so removing a url from the config removes it from the
    index on the next run — with the deletion visible in the log rather
    than silent.

    A failure is per document. An unreachable page is reported and its
    existing file left exactly as it was: the snapshot's history must
    mean "the source changed", and a deletion caused by a timeout would
    be a lie in that history.

    Args:
        root: The snapshot repository's directory. Created and
            `git init`-ed when it does not exist.
        urls: Documents this snapshot holds, from the repo's config
            entry.
        specs: `Config.connectors`, for routing.

    Returns:
        What was written, and what could not be.

    Raises:
        ValueError: `root` exists, holds files, and is not a repository.
        GitCommandError: git could not run, or a command failed.
    """
    _ensure_repo(root)
    expected: dict[PurePosixPath, str] = {}
    added = updated = unchanged = 0
    failed: list[tuple[str, str]] = []

    for url in urls:
        relative = document_path(url)
        clash = expected.get(relative)
        if clash is not None:
            # Two urls, one file. Silently letting the second win would
            # make the snapshot depend on config order and lose a
            # document without saying so.
            failed.append((url, f"maps to the same file as {clash} ({relative})"))
            continue
        expected[relative] = url

        connector = route(url, specs)
        if connector is None:
            failed.append((url, "no connector claims it"))
            continue
        try:
            document = connector.fetch(url)
        except ConnectorError as exc:
            failed.append((url, str(exc)))
            continue

        target = root / relative
        # Bytes, not text. `read_text` translates line endings, so a
        # document carrying a single `\r\n` — a real GitHub issue does —
        # never compares equal to what was just written from it: the file
        # would be rewritten and reported as updated on every sync
        # forever, while git correctly saw no change and committed
        # nothing. Comparing what is actually on disk is the only version
        # of this check that agrees with git.
        rendered = render(document).encode("utf-8")
        if target.exists() and target.read_bytes() == rendered:
            unchanged += 1
            continue
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(rendered)
        updated += existed
        added += not existed

    tracked = _tracked(root)
    stale = sorted(tracked - set(expected))
    for relative in stale:
        (root / relative).unlink(missing_ok=True)

    report = SnapshotReport(
        added=added,
        updated=updated,
        unchanged=unchanged,
        removed=len(stale),
        failed=tuple(failed),
    )
    # Everything the config accounts for, deletions included — see
    # `_commit` for why the set is wider than what this pass rewrote.
    # git, not the counters above, decides whether that amounts to a
    # commit: its None is what "nothing happened" means here.
    staged = [path for path in expected if path in tracked or (root / path).exists()]
    return replace(report, commit=_commit(root, f"sync: {report.summary()}", staged + stale))


__all__ = ["SnapshotReport", "document_path", "materialize", "render"]
