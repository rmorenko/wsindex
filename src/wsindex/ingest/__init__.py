"""Ingest stage: getting sources in, and knowing what changed in them.

Three concerns, in the order a workspace meets them: `git_sync` brings a
working copy up to date, `walker` and `chunker` turn it into chunks, and
`git_state` remembers how far indexing got so the next run can do less.
The ast/ and text_chunker submodules are implementation details reached
via chunker dispatch, not by outside callers.
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
from wsindex.ingest.walker import WalkedFile, inspect_file, walk_repo

__all__ = [
    "GitCommandError",
    "GitUnavailableError",
    "IndexState",
    "NotAGitRepositoryError",
    "RepoDiff",
    "SyncOutcome",
    "WalkedFile",
    "chunk_file",
    "diff_since",
    "has_uncommitted_changes",
    "head_commit",
    "inspect_file",
    "sync_repo",
    "walk_repo",
]
