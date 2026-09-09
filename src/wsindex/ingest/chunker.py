"""Routing a file to a chunker — the pipeline's single entry point."""

from typing import assert_never

from wsindex.ingest.ast import ast_chunks
from wsindex.ingest.languages import REGISTRY
from wsindex.ingest.text_chunker import chunk_text
from wsindex.model import Chunk, Kind


def chunk_file(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Route a file to a chunker by its kind — the pipeline's single entry point.

    DOC uses the text chunker. CODE and CONFIG go through the AST path
    when the registry has a parser and an extractor for the language, and
    fall back to plain windows otherwise — which is the normal state on a
    base install, where the optional `ast` extra is absent.

    Code and configs share one branch on purpose. They used to be two,
    each with its own parser table, but a `LanguageSpec` carries the
    grammar and the extractor together (see `wsindex.ingest.languages`),
    so "is there an AST path for this language?" is now one question
    rather than one per kind.

    Args:
        text: File contents.
        repo: Repo id recorded on every chunk.
        path: Repo-relative path recorded on every chunk.
        lang: Language name, as the walker identified it.
        kind: Artifact category, as the walker identified it.

    Returns:
        The file's chunks, in file order.
    """
    match kind:
        case Kind.DOC:
            return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
        case Kind.CODE | Kind.CONFIG:
            parser = REGISTRY.parser(lang)
            extractor = REGISTRY.extractor(lang)
            if parser is None or extractor is None:
                return chunk_text(text, repo=repo, path=path, lang=lang, kind=kind)
            return ast_chunks(
                parser=parser,
                extractor=extractor,
                text=text,
                repo=repo,
                path=path,
                lang=lang,
                kind=kind,
            )
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(kind)
