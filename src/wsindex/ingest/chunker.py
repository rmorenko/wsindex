from typing import assert_never

from wsindex.ingest.text_chunker import chunk_text
from wsindex.model import Chunk, Kind


def chunk_file(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Route a file to a chunker by its kind — the pipeline's single entry point.

    CODE and CONFIG fall back to the text chunker until the AST chunkers
    land; the branches are separate so each can be swapped alone.
    """
    match kind:
        case Kind.DOC:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CODE:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CONFIG:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(kind)
