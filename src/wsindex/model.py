"""Core data model of the index: Chunk (unit of indexing) and Hit (search result).

These types are the shared contract between all pipeline layers (ARCH §6):
chunkers produce Chunks, the embedder reads Chunk.text, stores keep chunk
metadata next to the vector and return Hits.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Kind(StrEnum):
    """Artifact category; drives the chunker choice (AST vs text, ARCH §4).

    StrEnum (not plain Enum) on purpose: members behave as regular strings,
    so `Kind.CODE == "code"` holds and metadata serializes without `.value`.
    """

    CODE = "code"
    CONFIG = "config"
    DOC = "doc"


@dataclass(frozen=True, kw_only=True)
class Chunk:
    """One meaningful fragment of a file (function, config table, doc section).

    Frozen: a chunk describes an exact piece of source text and must never
    mutate after creation. The `id` field is derived from `text` and `path`,
    so it is excluded from __init__ and computed in __post_init__ instead —
    this makes it impossible to construct a Chunk with a wrong id.

    Attributes:
        id: Deterministic sha256 of (text, path); see `chunk_id`.
        repo: Repo id from the config; doubles as the dataset name.
        path: File path relative to the repo root, POSIX separators.
        lang: Language/format name, e.g. "python", "toml", "markdown".
        kind: Artifact category; drove the chunker choice.
        symbol: Definition the chunk covers ("f", "Cls.method"); None for
            gap chunks between definitions.
        node_type: tree-sitter node type behind the chunk; None for
            non-AST chunkers and gap chunks.
        start_line: First line of the fragment, 1-based inclusive.
        end_line: Last line of the fragment, 1-based inclusive.
        text: Verbatim slice of the source, the embedding input.
    """

    # init=False: not a constructor argument, filled in __post_init__.
    id: str = field(init=False)
    repo: str
    path: str
    lang: str
    kind: Kind
    symbol: str | None
    node_type: str | None
    start_line: int
    end_line: int
    text: str

    def __post_init__(self) -> None:
        # object.__setattr__ is the documented way to assign a field of a
        # frozen dataclass during initialization (plain `self.id = ...`
        # would raise FrozenInstanceError).
        object.__setattr__(self, "id", self.chunk_id(self.text, path=self.path))

    def to_metadata(self) -> dict[str, Any]:
        """Serialize for the vector store.

        Returns:
            All fields (including `id`) as a flat dict — the payload every
            store keeps next to the vector and returns inside Hit.metadata.
        """
        return asdict(self)

    @staticmethod
    def chunk_id(text: str, *, path: str) -> str:
        """Deterministic chunk id: sha256 over (text, path).

        The same fragment at the same path always yields the same id, which
        makes reindexing reproducible and lets the incremental mode skip
        chunks that are already stored (ARCH §5.1). Public on purpose: the
        pipeline computes ids before constructing Chunks.

        Each part is hashed together with its length so that pairs like
        ("ab", "c") and ("a", "bc") do not collide after concatenation.

        Args:
            text: Verbatim chunk text.
            path: Repo-relative path of the file the text came from.

        Returns:
            64-character sha256 hex digest, stable across runs and machines.
        """
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(len(text).to_bytes(8, "big"))
        h.update(path.encode("utf-8"))
        h.update(len(path).to_bytes(8, "big"))
        return h.hexdigest()


@dataclass(frozen=True, kw_only=True)
class Hit:
    """Backend-independent search result (ARCH §6.4).

    Concrete stores normalize their native responses into this type, so the
    pipeline reads chunk fields only from `metadata` and never depends on
    a concrete backend.

    Attributes:
        score: Similarity of the chunk to the query; higher is better,
            comparable across datasets of one workspace.
        metadata: The stored `Chunk.to_metadata()` dict of the found chunk.
        native_id: Backend-native record id (chunk id for LanceDBStore);
            service field, not used for ranking or output.
    """

    score: float
    metadata: dict[str, Any]
    native_id: str | None = None
