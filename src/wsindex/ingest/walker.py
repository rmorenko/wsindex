"""Repository walking: find indexable files, skip junk, detect lang/kind.

First stage of the indexing pipeline (ARCH §4): yields WalkedFile entries
that the chunker dispatcher consumes. Reads only file names, sizes and a
small binary-sniff prefix — never whole file contents.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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


# Decision tables instead of if/elif chains: adding a format is one data row.
# Only the ARCH §1 corpus is listed on purpose — every extra format is a
# future obligation for the chunkers.
_SUFFIX_MAP: dict[str, tuple[str, Kind]] = {
    ".py": ("python", Kind.CODE),
    ".rs": ("rust", Kind.CODE),
    ".ts": ("typescript", Kind.CODE),
    ".java": ("java", Kind.CODE),
    ".toml": ("toml", Kind.CONFIG),
    ".yaml": ("yaml", Kind.CONFIG),
    ".yml": ("yaml", Kind.CONFIG),
    ".json": ("json", Kind.CONFIG),
    ".rst": ("rst", Kind.DOC),
    ".txt": ("text", Kind.DOC),
    ".md": ("markdown", Kind.DOC),
}

# Files matched by exact name (they have no useful extension).
_FILENAME_MAP: dict[str, tuple[str, Kind]] = {
    "Dockerfile": ("dockerfile", Kind.CONFIG),
}


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
    """Map a file to (lang, kind) by suffix or exact name; None = skip."""
    return _SUFFIX_MAP.get(path.suffix.lower()) or _FILENAME_MAP.get(path.name)


def is_binary(path: Path) -> bool:
    """Git's heuristic: a NUL byte in the first 8 KiB means binary."""
    with path.open("rb") as f:
        prefix = f.read(8192)
    return b"\x00" in prefix


def walk_repo(root: Path) -> Iterator[WalkedFile]:
    """Lazily yield indexable files under root, in deterministic order."""
    for dirpath, dir_names, filenames in os.walk(root):
        # Slice assignment mutates the list os.walk holds, so pruned dirs are
        # never descended into (plain `dir_names = ...` would rebind the local
        # name and silently disable the filter). Sorting makes traversal
        # order reproducible across OSes.
        dir_names[:] = sorted(d for d in dir_names if not _skip_dir(d))
        for f_name in sorted(filenames):
            abs_path = Path(dirpath) / f_name
            # Cheapest check first (name only), then stat, then open+read.
            found = detect_lang_kind(abs_path)
            if found is None:
                continue
            if abs_path.stat().st_size > MAX_FILE_SIZE:
                continue
            if is_binary(abs_path):
                continue
            lang, kind = found
            yield WalkedFile(
                rel_path=abs_path.relative_to(root).as_posix(),
                lang=lang,
                kind=kind,
            )
