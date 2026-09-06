"""Ingest stage: repository walking and file chunking.

The public API is the two entry points the pipeline uses; the ast/ and
text_chunker submodules are implementation details reached via chunker
dispatch, not by outside callers.
"""

from wsindex.ingest.chunker import chunk_file
from wsindex.ingest.walker import WalkedFile, walk_repo

__all__ = ["WalkedFile", "chunk_file", "walk_repo"]
