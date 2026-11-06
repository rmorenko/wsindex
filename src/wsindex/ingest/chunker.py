from typing import assert_never

from wsindex.ingest.ast import (
    CODE_PARSERS,
    CONFIG_PARSERS,
    chunk_code_ast,
    chunk_config,
)
from wsindex.ingest.text_chunker import chunk_text
from wsindex.model import Chunk, Kind


def chunk_file(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Route a file to a chunker by its kind — the pipeline's single entry point.

    DOC uses the text chunker; CODE and CONFIG get AST chunks when a
    grammar for the language is available and plain windows otherwise.
    """
    match kind:
        case Kind.DOC:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CODE if lang in CODE_PARSERS:
            return chunk_code_ast(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CODE:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CONFIG if lang in CONFIG_PARSERS:
            return chunk_config(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CONFIG:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(kind)
