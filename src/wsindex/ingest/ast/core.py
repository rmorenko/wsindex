"""Language-agnostic mechanism of AST chunking.

Definition nodes become spans, then a line-coverage pass turns everything
else into gap chunks, so each non-blank line lands in exactly one chunk.
Per-language policy (which nodes matter and where the symbol comes from)
lives in the sibling language modules; the registries that tie them
together live in the package __init__.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from wsindex.model import Chunk, SourceFile

try:
    from tree_sitter import Node, Parser

    HAS_TREE_SITTER = True
except ImportError:  # pragma: no cover - only reachable on a base install (CI matrix)
    HAS_TREE_SITTER = False


@dataclass(frozen=True, kw_only=True)
class Span:
    """A future chunk: 1-based inclusive line range plus Chunk metadata."""

    start_line: int
    end_line: int
    symbol: str | None
    node_type: str | None


class SpanExtractor(Protocol):
    """Per-language policy: which parts of a tree become their own chunks.

    Called once per file with the parse tree and the file's lines. The
    `covered` list is the shared bookkeeping that makes chunking total:
    mark the lines a span claims (`mark_covered`, or `def_span` which
    does it for you) and whatever is left becomes gap chunks, so every
    non-blank line lands in exactly one chunk.

    Lives here rather than with `LanguageSpec` so that `nested.extractor`
    can declare it as a return type: the registry imports the language
    modules, so the language modules cannot import the registry.
    """

    def __call__(self, root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
        """Extract spans from one parsed file.

        Args:
            root: Root node of the parsed file. The parser is
                error-tolerant, so this may describe a partial tree —
                return what you recognized and let the gap pass cover the
                rest rather than raising.
            lines: The file's lines, without terminators; 0-based, while
                span line numbers are 1-based inclusive.
            covered: One flag per line (index 0 unused), shared with the
                gap pass. Mark what you claim.

        Returns:
            The spans this language wants as chunks of their own.
        """
        ...


def line_span(node: Node) -> tuple[int, int]:
    """Node position as 1-based inclusive lines.

    A node that swallows the trailing newline reports end_point at column 0
    of the next row — then the last content line is the row itself.
    """
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1 if node.end_point[1] > 0 else node.end_point[0]
    return start, end


def mark_covered(covered: list[bool], *, start: int, end: int) -> None:
    for i in range(start, end + 1):
        covered[i] = True


def _uncovered_runs(covered: list[bool], *, start: int, end: int) -> list[tuple[int, int]]:
    """Contiguous runs of uncovered lines within [start, end], inclusive."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for i in range(start, end + 1):
        if not covered[i] and run_start is None:
            run_start = i
        elif covered[i] and run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, end))
    return runs


def gap_spans(
    lines: list[str],
    *,
    covered: list[bool],
    start: int,
    end: int,
    symbol: str | None,
    node_type: str | None,
) -> list[Span]:
    """Spans for uncovered runs in [start, end], blank edges trimmed.

    Marks produced lines as covered, so a later pass over an enclosing
    range cannot emit them twice. Blank-only runs yield nothing.
    """
    spans: list[Span] = []
    for run_start, run_end in _uncovered_runs(covered, start=start, end=end):
        while run_start <= run_end and not lines[run_start - 1].strip():
            run_start += 1
        while run_end >= run_start and not lines[run_end - 1].strip():
            run_end -= 1
        if run_start > run_end:
            continue
        mark_covered(covered, start=run_start, end=run_end)
        spans.append(
            Span(start_line=run_start, end_line=run_end, symbol=symbol, node_type=node_type)
        )
    return spans


def _assemble(spans: list[Span], *, lines: list[str], source: SourceFile) -> list[Chunk]:
    """Turn spans into Chunks in file order; text is a verbatim line slice."""
    return [
        source.chunk(
            text="\n".join(lines[span.start_line - 1 : span.end_line]),
            start_line=span.start_line,
            end_line=span.end_line,
            symbol=span.symbol,
            node_type=span.node_type,
        )
        for span in sorted(spans, key=lambda span: span.start_line)
    ]


def unwrap(node: Node, *, wrapper: str, inner: tuple[str, ...]) -> Node:
    """Return the definition inside a wrapper node, the node itself otherwise.

    Wrappers differ per grammar: python hides defs in decorated_definition,
    typescript — in export_statement.
    """
    if node.type == wrapper:
        for child in node.named_children:
            if child.type in inner:
                return child
    return node


def symbol_name(node: Node) -> str | None:
    """Identifier of a def/class; None on error-recovery trees with no name."""
    child = node.child_by_field_name("name")
    if child is None or child.text is None:
        return None
    return child.text.decode()


def def_span(outer: Node, *, covered: list[bool], symbol: str, node_type: str) -> Span:
    """Span of one definition; `outer` includes decorators or the export keyword."""
    start, end = line_span(outer)
    mark_covered(covered, start=start, end=end)
    return Span(start_line=start, end_line=end, symbol=symbol, node_type=node_type)


PARSE_ERROR = "parse-error"
"""`node_type` on the chunks left over when the grammar reported errors.

A value rather than a flag on the side: it reaches search output, the
index report and `wsindex explain` through the field every chunk already
has."""


def ast_chunks(
    text: str,
    source: SourceFile,
    *,
    parser: Parser,
    extractor: Callable[[Node, list[str], list[bool]], list[Span]],
) -> list[Chunk]:
    """Shared skeleton: parse, run the per-language extractor, fill the gaps.

    One language's parser and extractor rather than a table of them:
    picking the pair is the registry's job (`LanguageSpec`), and passing
    two dicts plus the key to look them up with only made it possible for
    the two to disagree.

    Args:
        text: File contents.
        source: The file these chunks belong to.
        parser: Grammar-backed parser for the file's language.
        extractor: That language's span policy.

    Returns:
        Chunks in file order, covering every non-blank line exactly once.
        When the grammar could not read the file, the leftover chunks
        carry `PARSE_ERROR` as their node type — see below.
    """
    lines = text.splitlines()
    covered = [False] * (len(lines) + 1)
    tree = parser.parse(text.encode())
    root = tree.root_node
    spans = extractor(root, lines, covered)
    # `has_error` was available all along and asked by nobody, so a file
    # the grammar could not read degraded quietly into text windows —
    # search kept working, worse, and there was no way to find out.
    # It happens for real reasons: syntax newer than the installed
    # grammar, or a `formats` entry pointing a suffix at the wrong
    # language.
    #
    # Marked on the gap chunks rather than reported out of band, because
    # `node_type` is already carried everywhere a chunk goes and is not
    # part of the chunk id — so this costs no second parse and moves no
    # id. Its limit is worth naming: an error confined inside a
    # definition still leaves that definition's own span intact, so this
    # catches the common case (a broken top level) and not every case.
    leftover = PARSE_ERROR if root.has_error else None
    spans += gap_spans(
        lines, covered=covered, start=1, end=len(lines), symbol=None, node_type=leftover
    )
    return _assemble(spans, lines=lines, source=source)
