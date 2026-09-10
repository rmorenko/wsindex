"""Which files are worth indexing, one file at a time.

Reads only names, sizes and a small binary-sniff prefix — never whole
contents. No traversal lives here: the pipeline gets its file list from
git, so what is left is the *policy*.

Which files count as indexable is not decided here either. Suffixes and
exact names come from the language registry, so registering a
`LanguageSpec` is enough to make this stage select the files.

The policy is *per repository*, which is why the tables arrive as
arguments and the constants below are only defaults: in an Angular repo a
`.component.html` is source, in a Python repo an `.html` is generated
noise, and no global table is right for both.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from wsindex.ingest.languages import REGISTRY
from wsindex.model import Kind

# Directory-pruning policy in one place: any hidden directory (dot-prefix)
# is skipped, plus a short list of well-known build/artifact dirs that
# are not hidden. Hidden ones like `.git`/`.venv`/`.pytest_cache` are
# caught by the `startswith(".")` rule automatically — do not list them
# here, the two rules would drift (which is why `.next` is absent below
# despite being exactly this kind of directory).
IGNORED_DIRS: frozenset[str] = frozenset(
    [
        "node_modules",
        "target",
        "__pycache__",
        # Build output. Indexing it means indexing the same code twice,
        # once as source and once minified — and a `dist/` full of
        # generated `.json` is noise no query wants back.
        "dist",
        "build",
        "out",
        # Coverage reports: generated HTML and JSON, by the megabyte.
        "coverage",
        "htmlcov",
    ]
)
MAX_FILE_SIZE = 1024 * 1024


def _skip_dir(name: str) -> bool:
    """One place to answer 'should the walker descend into this directory?'.

    Splitting the policy across an inline filter and a constant made the
    two rules easy to drift apart (an added ignored dir with a dot would
    duplicate the hidden-dir rule silently). Colocating them is cheap.

    Not overridable per repo, and that is a real limitation rather than
    an oversight: a repository whose `build/` holds source has no way to
    say so. The alternative is an include/exclude engine with ordering
    rules, which is the thing this design refuses to become — `ignore`
    narrows, nothing widens.
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


def detect_lang_kind(
    path: Path, formats: Mapping[str, tuple[str, Kind]] | None = None
) -> tuple[str, Kind] | None:
    """Map a file to (lang, kind) by suffix or exact name; None = skip.

    The table this used to hold is now `wsindex.ingest.languages.REGISTRY`,
    so a plugin that registers a `LanguageSpec` starts getting its files
    walked without touching this module. That was the point: the walker
    was the one stage a language could not reach from outside.

    `formats` is the repository's own answer, and it wins over the
    registry — the whole point of the override is to say "here, `.html`
    is source" or "here, `.txt` is documentation" about one repo. A
    suffix mapped to a language with no grammar is not a problem: the
    chunker falls back to text windows, which is what makes markup the
    cheap way to cover a long tail of formats (`.sql`, `.proto`) that do
    not each deserve a grammar.

    Args:
        path: The file; only its name is read.
        formats: Repo-level suffix -> (lang, kind), lowercase suffixes
            with the dot. None means the registry alone.

    Returns:
        `(lang, kind)`, or None when nothing claims the file.
    """
    if formats:
        override = formats.get(path.suffix.lower())
        if override is not None:
            return override
    spec = REGISTRY.match(path)
    return (spec.name, spec.kind) if spec is not None else None


def is_binary(path: Path) -> bool:
    """Git's heuristic: a NUL byte in the first 8 KiB means binary."""
    with path.open("rb") as f:
        prefix = f.read(8192)
    return b"\x00" in prefix


def inspect_file(
    root: Path,
    rel_path: str,
    *,
    ignore: Sequence[str] = (),
    formats: Mapping[str, tuple[str, Kind]] | None = None,
) -> WalkedFile | None:
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
        ignore: Globs from this repo's config, matched against the whole
            relative path. `*` crosses directory separators, so `dist/*`
            means everything under `dist/` at any depth and `*.min.js`
            means what it looks like.
        formats: This repo's suffix overrides; see `detect_lang_kind`.

    Returns:
        The WalkedFile, or None when the policy excludes it.
    """
    # Directory pruning, which walk_repo does by not descending, has to be
    # re-checked here: git happily reports a tracked file under
    # `node_modules/`, and the two callers must select the same set.
    parts = PurePosixPath(rel_path).parts
    if any(_skip_dir(part) for part in parts[:-1]):
        return None
    if any(fnmatchcase(rel_path, pattern) for pattern in ignore):
        return None
    abs_path = root / rel_path
    # Cheapest check first (name only), then stat, then open+read.
    found = detect_lang_kind(abs_path, formats)
    if found is None:
        return None
    try:
        if abs_path.is_symlink():
            # A symlink is a name, not a file. `is_file()` follows it, so
            # a repository containing `notes.md -> ~/.ssh/id_rsa` had the
            # key's *contents* indexed — measured, and git tracks
            # symlinks, so cloning someone's repository let them choose
            # which of your files went into your index. A link whose
            # target is inside the repo is no better: the target is
            # walked on its own, and indexing it twice would put one text
            # at two paths.
            return None
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
