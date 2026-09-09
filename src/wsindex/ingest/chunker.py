"""Routing a file to a chunker — the pipeline's single entry point."""

from dataclasses import replace
from typing import assert_never

from wsindex.ingest.ast import ast_chunks, gap_spans, mark_covered
from wsindex.ingest.languages import REGISTRY, Section
from wsindex.ingest.text_chunker import chunk_text
from wsindex.model import Chunk, Kind, SourceFile


def chunk_file(text: str, source: SourceFile) -> list[Chunk]:
    """Route a file to a chunker by its kind — the pipeline's single entry point.

    DOC uses the text chunker. CODE and CONFIG go through the AST path
    when the registry has a parser and an extractor for the language, and
    fall back to plain windows otherwise — which is the normal state on a
    base install, where the optional `ast` extra is absent.

    A *container* language (a `.vue` or `.svelte` component) is neither:
    it is split into sections and each section is chunked as its own
    language, by this same function. See `_chunk_container`.

    Code and configs share one branch on purpose. They used to be two,
    each with its own parser table, but a `LanguageSpec` carries the
    grammar and the extractor together (see `wsindex.ingest.languages`),
    so "is there an AST path for this language?" is now one question
    rather than one per kind.

    Args:
        text: File contents.
        source: The file being chunked — its identity and its kind.

    Returns:
        The file's chunks, in file order.
    """
    match source.kind:
        case Kind.DOC | Kind.COMMIT:
            # COMMIT never reaches here in practice — `ingest.commits`
            # builds those chunks itself, because a commit message is one
            # unit and windowing it would scatter the reasoning `why`
            # exists to surface. Routed anyway so the match stays total.
            return chunk_text(text, source)
        case Kind.CODE | Kind.CONFIG:
            spec = REGISTRY.get(source.lang)
            parser = REGISTRY.parser(source.lang)
            if spec is not None and spec.sections is not None and parser is not None:
                lines = text.splitlines()
                return _chunk_container(
                    spec.sections(parser.parse(text.encode()).root_node, lines),
                    lines=lines,
                    source=source,
                )
            extractor = REGISTRY.extractor(source.lang)
            if parser is None or extractor is None:
                return chunk_text(text, source)
            return ast_chunks(text, source, parser=parser, extractor=extractor)
        case _:  # pragma: no cover - mypy proves this branch unreachable
            assert_never(source.kind)


def _chunk_container(
    sections: list[Section], *, lines: list[str], source: SourceFile
) -> list[Chunk]:
    """Chunk each section as its own language, back in file coordinates.

    The recursion is what makes a container cheap: a `<script lang="ts">`
    block is handed to `chunk_file` as TypeScript and gets the real
    TypeScript extractor, so nothing here knows anything about
    TypeScript. Teaching the box a new language teaches it to Vue script
    blocks at the same time.

    Two things are rewritten on the way back:

    - **Line numbers.** A section is chunked as if it were a file, so its
      chunks start at line 1. The section's own offset is added back, or
      every hit in a component would point at the top of the file.
    - **Language.** Chunks keep the *container's* name, not the section's.
      `lang` answers "what kind of file is this" everywhere else in the
      index — the walker assigns it from the suffix — and a `.vue` file
      whose chunks claimed to be TypeScript would be the one thing
      `--lang vue` could not find. The section's own shape survives in
      `node_type`, which the sub-chunker filled in.

    What is left over then goes through the same gap pass every language
    gets. A section covers what is *between* the tags, so `<script>` and
    `</script>` belong to no section at all — without the sweep those
    lines would silently fall out of the index, and the invariant that
    every non-blank line lands in exactly one chunk would hold everywhere
    but here.

    Args:
        sections: What the container's splitter returned.
        lines: The container's lines, for the gap pass.
        source: The container file; sections inherit its name and kind.

    Returns:
        Every section's chunks plus the leftovers, in file order.
    """
    chunks: list[Chunk] = []
    covered = [False] * (len(lines) + 1)
    for section in sections:
        inner = replace(source, lang=section.lang)
        for chunk in chunk_file(section.text, inner):
            offset = section.start_line - 1
            start, end = chunk.start_line + offset, chunk.end_line + offset
            mark_covered(covered, start=start, end=end)
            # `replace` rather than a fresh Chunk: `id` is derived from
            # (text, path) in __post_init__ and neither changes here, so
            # the id a section's chunk gets is the id it keeps — dedup
            # and incremental deletes still work on it.
            chunks.append(replace(chunk, lang=source.lang, start_line=start, end_line=end))
    chunks += [
        source.chunk(
            text="\n".join(lines[span.start_line - 1 : span.end_line]),
            start_line=span.start_line,
            end_line=span.end_line,
        )
        for span in gap_spans(
            lines, covered=covered, start=1, end=len(lines), symbol=None, node_type=None
        )
    ]
    return sorted(chunks, key=lambda chunk: chunk.start_line)
