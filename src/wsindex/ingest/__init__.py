"""Ingest stage: repository walking, file chunking, and git state.

The public API is what the pipeline consumes — the two indexing entry
points (`walk_repo`, `chunk_file`) plus the git layer that tells an
incremental run what changed. The ast/ and text_chunker submodules are
implementation details reached via chunker dispatch, not by outside
callers.
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
from wsindex.ingest.walker import WalkedFile, inspect_file, walk_repo

__all__ = [
    "GitCommandError",
    "GitUnavailableError",
    "IndexState",
    "NotAGitRepositoryError",
    "RepoDiff",
    "WalkedFile",
    "chunk_file",
    "diff_since",
    "has_uncommitted_changes",
    "head_commit",
    "inspect_file",
    "walk_repo",
]
