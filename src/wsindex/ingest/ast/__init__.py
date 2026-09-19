"""AST chunking package: grammar registries plus the two chunk functions.

The language-agnostic mechanism lives in `core`; per-language policy is one
small module per language (`python`, `rust`, `typescript`, `java`,
`configs`). Adding a language = a new module with a `spans` extractor plus
one row in each registry below. Grammars are optional: the registries hold
whichever grammars imported successfully, and the dispatcher falls back to
the text chunker for everything else. Files with syntax errors are chunked
from whatever the error-tolerant parser recognized.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from wsindex.ingest.ast import configs, java, python, rust, typescript
from wsindex.ingest.ast.core import HAS_TREE_SITTER, Span, ast_chunks, load_parsers
from wsindex.model import Chunk, Kind

__all__ = [
    "CODE_PARSERS",
    "CONFIG_PARSERS",
    "HAS_TREE_SITTER",
    "chunk_code_ast",
    "chunk_config",
]

if TYPE_CHECKING:
    from tree_sitter import Node, Parser

CODE_PARSERS: dict[str, Parser] = {}

CONFIG_PARSERS: dict[str, Parser] = {}

if HAS_TREE_SITTER:
    CODE_PARSERS = load_parsers(
        (
            ("python", "tree_sitter_python", "language"),
            ("rust", "tree_sitter_rust", "language"),
            ("typescript", "tree_sitter_typescript", "language_typescript"),
            ("java", "tree_sitter_java", "language"),
        )
    )

    CONFIG_PARSERS = load_parsers(
        (
            ("toml", "tree_sitter_toml", "language"),
            ("yaml", "tree_sitter_yaml", "language"),
            ("json", "tree_sitter_json", "language"),
            ("dockerfile", "tree_sitter_dockerfile", "language"),
        )
    )

_CODE_EXTRACTORS: dict[str, Callable[[Node, list[str], list[bool]], list[Span]]] = {
    "python": python.spans,
    "rust": rust.spans,
    "typescript": typescript.spans,
    "java": java.spans,
}

_CONFIG_EXTRACTORS: dict[str, Callable[[Node, list[str], list[bool]], list[Span]]] = {
    "toml": configs.toml_spans,
    "yaml": configs.yaml_spans,
    "json": configs.json_spans,
    "dockerfile": configs.dockerfile_spans,
}


def chunk_config(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Chunk a config file: one chunk per TOML table, top-level YAML/JSON
    key or Dockerfile stage; comments and leftovers become gap chunks.
    """
    return ast_chunks(
        parsers=CONFIG_PARSERS,
        extractors=_CONFIG_EXTRACTORS,
        text=text,
        repo=repo,
        path=path,
        lang=lang,
        kind=kind,
    )


def chunk_code_ast(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Chunk a source file: one chunk per function, method or type
    definition; the module-level remainder becomes gap chunks.
    """
    return ast_chunks(
        parsers=CODE_PARSERS,
        extractors=_CODE_EXTRACTORS,
        text=text,
        repo=repo,
        path=path,
        lang=lang,
        kind=kind,
    )
