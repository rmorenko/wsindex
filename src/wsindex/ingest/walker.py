"""Which files are worth indexing, one file at a time.

First stage of the indexing pipeline (ARCH §4): produces the WalkedFile
entries the chunker dispatcher consumes. Reads only file names, sizes and
a small binary-sniff prefix — never whole file contents.

No traversal lives here any more. Since step 25b the pipeline gets its
file list from git (`ls-files --cached --others --exclude-standard`)
rather than from a filesystem walk, so what is left is the *policy*: does
this one path deserve indexing. A `walk_repo` generator survived that
change for a while as a tested primitive nobody called, which is how a
second definition of the policy starts to drift from the first.

Which files count as indexable is not decided here. Suffixes and exact
names come from the language registry, so registering a `LanguageSpec`
is enough to make this stage select the files — see
`wsindex.ingest.languages`. What stays here is the policy that has
nothing to do with language: pruned directories, the size ceiling, the
binary sniff.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from wsindex.ingest.languages import REGISTRY
from wsindex.model import Kind

# Directory-pruning policy in one place: any hidden directory (dot-prefix)
# is skipped, plus a short list of well-known build/artifact dirs that
# are not hidden. Hidden ones like `.git`/`.venv`/`.pytest_cache` are
# caught by the `startswith(".")` rule automatically — do not list them
# here, the two rules would drift.
IGNORED_DIRS: frozenset[str] = frozenset(["node_modules", "target", "__pycache__"])
MAX_FILE_SIZE = 1024 * 1024


def _skip_dir(name: str) -> bool:
    """One place to answer 'should the walker descend into this directory?'.

    Splitting the policy across an inline filter and a constant made the
    two rules easy to drift apart (an added ignored dir with a dot would
    duplicate the hidden-dir rule silently). Colocating them is cheap.
    """
    return name.startswith(".") or name in IGNORED_DIRS


@dataclass(frozen=True, kw_only=True)
class WalkedFile:
    """A file selected for indexing.

    `rel_path` is the file's identity in the index: it enters `chunk_id`,
    drives dedup, and is what the user sees in search results. POSIX and
    repo-relative on purpose — the same file cloned into different roots
    (or different OSes) must yield the same id, otherwise dedup breaks.
    Runtime absolute paths are the caller's business: they hold `root`
    already, so `root / rel_path` reconstructs the on-disk location when
    needed — no reason to duplicate it in this type.

    Attributes:
        rel_path: POSIX path relative to the repo root; the file's identity.
        lang: Detected language ("python", "rust", ...).
        kind: Broad category (CODE / DOC / CONFIG).
    """

    rel_path: str
    lang: str
    kind: Kind


def detect_lang_kind(path: Path) -> tuple[str, Kind] | None:
    """Map a file to (lang, kind) by suffix or exact name; None = skip.

    The table this used to hold is now `wsindex.ingest.languages.REGISTRY`,
    so a plugin that registers a `LanguageSpec` starts getting its files
    walked without touching this module. That was the point: the walker
    was the one stage a language could not reach from outside.
    """
    spec = REGISTRY.match(path)
    return (spec.name, spec.kind) if spec is not None else None


def is_binary(path: Path) -> bool:
    """Git's heuristic: a NUL byte in the first 8 KiB means binary."""
    with path.open("rb") as f:
        prefix = f.read(8192)
    return b"\x00" in prefix


def inspect_file(root: Path, rel_path: str) -> WalkedFile | None:
    """Apply the indexing policy to one named file; None means "skip it".

    The per-file half of `walk_repo`, split out so an incremental run can
    ask about the handful of paths git reported as changed without
    walking the tree. Both callers must agree on what counts as
    indexable — if they drifted, a file would be indexed by a full pass
    and ignored by an incremental one (or the reverse), and the index
    would depend on which run happened to touch it.

    Unlike `walk_repo` this receives a path from the outside, so it
    cannot assume the file is there: a path may have been deleted between
    git reporting it and this call, or point at a directory. Both mean
    "nothing to index", not an error.

    Args:
        root: Repository root.
        rel_path: POSIX path relative to `root`.

    Returns:
        The WalkedFile, or None when the policy excludes it.
    """
    # Directory pruning, which walk_repo does by not descending, has to be
    # re-checked here: git happily reports a tracked file under
    # `node_modules/`, and the two callers must select the same set.
    parts = PurePosixPath(rel_path).parts
    if any(_skip_dir(part) for part in parts[:-1]):
        return None
    abs_path = root / rel_path
    # Cheapest check first (name only), then stat, then open+read.
    found = detect_lang_kind(abs_path)
    if found is None:
        return None
    try:
        if not abs_path.is_file():
            return None
        if abs_path.stat().st_size > MAX_FILE_SIZE:
            return None
        if is_binary(abs_path):
            return None
    except OSError:
        # Unreadable, a broken symlink, a race with a concurrent delete:
        # all of them mean the same thing to an indexer.
        return None
    lang, kind = found
    return WalkedFile(rel_path=PurePosixPath(rel_path).as_posix(), lang=lang, kind=kind)
