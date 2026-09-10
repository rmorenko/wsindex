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
    COMMIT = "commit"
    """A commit message. Its own kind rather than a DOC: it is not a file,
    it has no path on disk, and folding it into DOC would make `--kind doc`
    mean two different things and leave no way to search without it."""


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
class SearchFilter:
    """Store-side filters applied BEFORE the vector KNN (ADR-7).

    Prefilter, not postfilter: a selective filter over a postfilter of
    top-k could leave two hits instead of k. Repo scope is NOT here —
    repo maps to dataset, so narrowing by repo happens by choosing which
    datasets to ask, not by predicate.

    Attributes:
        lang: Restrict to these languages (OR within the field); empty
            tuple means "any language".
        kind: Restrict to these Kind values (OR within the field); empty
            tuple means "any kind".
        path: Path glob (`*`, `?` wildcards) matched against the chunk
            path; None means "any path". Chunks whose path is exactly
            equal to a literal without wildcards match too.
        symbol: Substring matched against the chunk symbol; only chunks
            with a non-null symbol pass this filter.
    """

    lang: tuple[str, ...] = ()
    kind: tuple[Kind, ...] = ()
    path: str | None = None
    symbol: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when no fields are set — the caller should pass None instead."""
        return not (self.lang or self.kind or self.path or self.symbol)


@dataclass(frozen=True, kw_only=True)
class SourceFile:
    """The file being chunked: what identifies it, and what it is.

    These four values always travel together — every chunker takes them,
    every `Chunk` carries them — so they are one thing rather than four
    parameters repeated across seven signatures. Grouping them also
    closes a class of mistake: `lang` and `kind` are both string-like,
    and separate parameters can be swapped without a word from the type
    checker.

    Attributes:
        repo: Repo id; also the dataset name in the store.
        path: POSIX path relative to the repo root. With `repo` it is the
            chunk's identity, so the same file cloned elsewhere still
            yields the same ids.
        lang: Language, as the walker identified it.
        kind: Artifact category (code, config, doc, commit).
    """

    repo: str
    path: str
    lang: str
    kind: Kind

    def chunk(
        self,
        *,
        text: str,
        start_line: int,
        end_line: int,
        symbol: str | None = None,
        node_type: str | None = None,
    ) -> "Chunk":
        """One chunk of this file, with the file's own fields filled in.

        Args:
            text: The chunk's text, a verbatim slice of the file.
            start_line: First line, 1-based inclusive.
            end_line: Last line, 1-based inclusive.
            symbol: Name of what this chunk is, when it has one.
            node_type: Grammar node the chunk came from, when it has one.

        Returns:
            The chunk.
        """
        return Chunk(
            repo=self.repo,
            path=self.path,
            lang=self.lang,
            kind=self.kind,
            symbol=symbol,
            node_type=node_type,
            start_line=start_line,
            end_line=end_line,
            text=text,
        )


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

    # The chunk's own fields, named. `metadata` stays the whole stored
    # payload — a store may keep more than the contract promises — but a
    # reader asking where a hit is should not have to know the spelling
    # of a dict key, in six modules, with a cast at every site.

    @property
    def repo(self) -> str:
        """Repo id the chunk belongs to."""
        return str(self.metadata["repo"])

    @property
    def path(self) -> str:
        """Repo-relative POSIX path of the file."""
        return str(self.metadata["path"])

    @property
    def start_line(self) -> int:
        """First line of the chunk, 1-based inclusive."""
        return int(self.metadata["start_line"])

    @property
    def end_line(self) -> int:
        """Last line of the chunk, 1-based inclusive."""
        return int(self.metadata["end_line"])

    @property
    def lang(self) -> str:
        """Language the chunker identified, or empty when it had none."""
        return str(self.metadata.get("lang", ""))

    @property
    def kind(self) -> str:
        """Artifact category as a plain string: code, config, doc, commit."""
        return str(self.metadata.get("kind", ""))

    @property
    def symbol(self) -> str | None:
        """What the chunk is called, when it is called anything."""
        found = self.metadata.get("symbol")
        return str(found) if found else None

    @property
    def text(self) -> str:
        """The chunk itself, verbatim."""
        return str(self.metadata["text"])

    @property
    def location(self) -> str:
        """`repo/path:start-end` — how every interface cites a hit."""
        return f"{self.repo}/{self.path}:{self.start_line}-{self.end_line}"

    def to_json(self) -> dict[str, Any]:
        """The fields an outside caller needs, flat and JSON-ready.

        One definition for every interface that answers with a hit — the
        HTTP API and the MCP tools both return this, so a caller that has
        seen one recognizes the other. `metadata` holds whatever the
        store kept; this is the part that is promised.

        `id` leads, because it is the only field that *names* the chunk.
        `repo/path:lines` locates it for a person, but one file yields
        many chunks and re-indexing moves the boundaries; the id is
        deterministic, is what dedup and every link are keyed on, and was
        the one thing an outside caller could not say back. A surprising
        hit was impossible to report precisely without it.
        """
        return {
            "id": self.native_id,
            "repo": self.repo,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "lang": self.lang,
            "kind": self.kind,
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "text": self.text,
        }
