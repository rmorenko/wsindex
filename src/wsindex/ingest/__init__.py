"""Ingest stage: getting sources in, and knowing what changed in them.

Three concerns, in the order a workspace meets them: `git_sync` brings a
working copy up to date, `walker` and `chunker` turn it into chunks, and
`git_state` remembers how far indexing got so the next run can do less.

`languages` cuts across all of them. A `LanguageSpec` is the whole
contract for teaching wsindex a new language — how to recognize its
files, and how to split them — and registering one is enough for the
walker to start selecting those files and the chunker to start routing
them. The spec types are re-exported here because that is the surface a
plugin imports; the extractor toolkit it writes `spans` with lives in
`wsindex.ingest.ast`.

Assembling the stage is this module's last act: once the built-in
languages and the plugin loader are both importable, `load_plugins` runs
so that an installed plugin is live for anything that imports wsindex.
Doing it here rather than in `languages` keeps that module free of the
discovery machinery — and avoids the import cycle, since `plugins` needs
`LanguageSpec`.
"""

from wsindex.ingest.chunker import chunk_file
from wsindex.ingest.git_state import (
    GitCommandError,
    GitUnavailableError,
    IndexState,
    NotAGitRepositoryError,
    RepoDiff,
    diff_since,
    has_uncommitted_changes,
    head_commit,
)
from wsindex.ingest.git_sync import SyncOutcome, sync_repo
from wsindex.ingest.languages import (
    REGISTRY,
    GrammarSpec,
    LanguageRegistry,
    LanguageSpec,
    SpanExtractor,
)
from wsindex.ingest.plugins import ENTRY_POINT_GROUP, PluginLoadWarning, load_plugins
from wsindex.ingest.walker import WalkedFile, inspect_file, walk_repo

__all__ = [
    "ENTRY_POINT_GROUP",
    "REGISTRY",
    "GitCommandError",
    "GitUnavailableError",
    "GrammarSpec",
    "IndexState",
    "LanguageRegistry",
    "LanguageSpec",
    "NotAGitRepositoryError",
    "PluginLoadWarning",
    "RepoDiff",
    "SpanExtractor",
    "SyncOutcome",
    "WalkedFile",
    "chunk_file",
    "diff_since",
    "has_uncommitted_changes",
    "head_commit",
    "inspect_file",
    "load_plugins",
    "sync_repo",
    "walk_repo",
]


load_plugins(REGISTRY)
