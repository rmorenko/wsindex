from typing import assert_never

from wsindex.ingest.ast_chunker import HAS_TREE_SITTER, chunk_python
from wsindex.ingest.text_chunker import chunk_text
from wsindex.model import Chunk, Kind


def chunk_file(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Route a file to a chunker by its kind — the pipeline's single entry point.

    DOC and CONFIG use the text chunker (config AST is step 14); CODE goes
    through a second, per-language dispatch in `chunk_code`.
    """
    match kind:
        case Kind.DOC:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CODE:
            return chunk_code(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CONFIG:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(kind)


def chunk_code(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Per-language dispatch for CODE files.

    Python gets AST chunks when the `ast` extra is installed; any other
    language — and Python on a base install — degrades to plain windows.
    """
    match lang:
        case "python" if HAS_TREE_SITTER:
            return chunk_python(text, repo=repo, path=path, lang=lang, kind=kind)
        case _:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
