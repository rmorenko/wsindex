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
from enum import StrEnum
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


class Skip(StrEnum):
    """Why a file is not in the index.

    Seven ways to be left out, and until this existed a reader could tell
    them apart only by guessing. "Why is my `.tf` file not searchable" is
    the likeliest question this tool gets asked, and the answer decides
    what to do next — add a `formats` entry, move the file, or look at
    permissions. `wsindex explain` turns these into that answer.

    One of them is not policy at all: `UNREADABLE` means a file git
    considers part of the project could not be opened. That is worth
    reporting from an ordinary index run, which the other six are not.
    """

    HIDDEN_DIR = "hidden-dir"
    IGNORED = "ignored"
    UNKNOWN_SUFFIX = "unknown-suffix"
    SYMLINK = "symlink"
    NOT_A_FILE = "not-a-file"
    TOO_LARGE = "too-large"
    BINARY = "binary"
    UNREADABLE = "unreadable"


SKIP_REASONS: dict[Skip, str] = {
    Skip.HIDDEN_DIR: "inside a directory the walker never descends into",
    Skip.IGNORED: "excluded by this repo's `ignore`",
    Skip.UNKNOWN_SUFFIX: "no language claims this suffix; add a `formats` entry for it",
    Skip.SYMLINK: "a symlink; whatever it points at is indexed on its own, if it belongs",
    Skip.NOT_A_FILE: "not a regular file",
    Skip.TOO_LARGE: f"larger than {MAX_FILE_SIZE // 1024 // 1024} MB",
    Skip.BINARY: "binary; a NUL byte in the first 8 KiB, which is git's own test",
    Skip.UNREADABLE: "could not be read; check the permissions",
}
"""One sentence per reason, said to a person. Apart from the enum so the
size comes from `MAX_FILE_SIZE` rather than from a number typed twice."""


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

    The convenience wrapper over `examine`, for the callers that only
    need a yes or a no. Both paths through the pipeline — the full pass
    and the incremental one — decide with the same function, because if
    they drifted a file would be indexed by one and ignored by the other,
    and the index would depend on which run happened to touch it.

    Every path arrives from outside, so nothing here may assume the file
    is there: it may have been deleted between git listing it and this
    call, or be a directory. Both mean "nothing to index", not an error.

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
    found = examine(root, rel_path, ignore=ignore, formats=formats)
    return found if isinstance(found, WalkedFile) else None


def examine(
    root: Path,
    rel_path: str,
    *,
    ignore: Sequence[str] = (),
    formats: Mapping[str, tuple[str, Kind]] | None = None,
) -> WalkedFile | Skip:
    """The same decision as `inspect_file`, with the reason kept.

    Split out because "no" is not one answer: an index run wants to
    report a file it could not *read*, and `wsindex explain` wants to
    tell a person which of the seven rules caught theirs. Both used to be
    impossible, since every rule returned the same None.

    Args:
        root: Repository root.
        rel_path: POSIX path relative to `root`.
        ignore: Globs from this repo's config; see `inspect_file`.
        formats: This repo's suffix overrides; see `detect_lang_kind`.

    Returns:
        The WalkedFile, or the `Skip` that excluded it.
    """
    # Directory pruning has to be re-checked here: git happily reports a
    # tracked file under `node_modules/`.
    parts = PurePosixPath(rel_path).parts
    if any(_skip_dir(part) for part in parts[:-1]):
        return Skip.HIDDEN_DIR
    if any(fnmatchcase(rel_path, pattern) for pattern in ignore):
        return Skip.IGNORED
    abs_path = root / rel_path
    # Cheapest check first (name only), then stat, then open+read.
    found = detect_lang_kind(abs_path, formats)
    if found is None:
        return Skip.UNKNOWN_SUFFIX
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
            return Skip.SYMLINK
        if not abs_path.is_file():
            # A broken symlink lands here too, and so does a file deleted
            # between git listing it and this call. Neither is anybody's
            # problem: there is nothing there to index.
            return Skip.NOT_A_FILE
        if abs_path.stat().st_size > MAX_FILE_SIZE:
            return Skip.TOO_LARGE
        if is_binary(abs_path):
            return Skip.BINARY
    except OSError:
        # Not folded into NOT_A_FILE, which is what this used to be. A
        # file git tracks and the filesystem refuses to open *should*
        # have been indexed and was not, and an index run that says
        # `files: 1` when there were two has told the reader something
        # untrue.
        return Skip.UNREADABLE
    lang, kind = found
    return WalkedFile(rel_path=PurePosixPath(rel_path).as_posix(), lang=lang, kind=kind)
