"""Language-agnostic mechanism of AST chunking.

Definition nodes become spans, then a line-coverage pass turns everything
else into gap chunks, so each non-blank line lands in exactly one chunk.
Per-language policy (which nodes matter and where the symbol comes from)
lives in the sibling language modules; the registries that tie them
together live in the package __init__.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

from wsindex.model import Chunk, Kind

try:
    from tree_sitter import Language, Node, Parser

    HAS_TREE_SITTER = True
except ImportError:  # pragma: no cover - only reachable on a base install (CI matrix)
    HAS_TREE_SITTER = False


def _load_parsers(table: tuple[tuple[str, str, str], ...]) -> dict[str, Parser]:
    """Build a lang->Parser dict from whichever grammar modules import."""
    parsers: dict[str, Parser] = {}
    for lang, module_name, getter in table:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover - partial grammar install
            continue
        parsers[lang] = Parser(Language(getattr(module, getter)()))
    return parsers


@dataclass(frozen=True)
class _Span:
    """A future chunk: 1-based inclusive line range plus Chunk metadata."""

    start_line: int
    end_line: int
    symbol: str | None
    node_type: str | None


def _line_span(node: Node) -> tuple[int, int]:
    """Node position as 1-based inclusive lines.

    A node that swallows the trailing newline reports end_point at column 0
    of the next row — then the last content line is the row itself.
    """
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1 if node.end_point[1] > 0 else node.end_point[0]
    return start, end


def _cover(covered: list[bool], start: int, end: int) -> None:
    for i in range(start, end + 1):
        covered[i] = True


def _uncovered_runs(covered: list[bool], start: int, end: int) -> list[tuple[int, int]]:
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


def _gap_spans(
    lines: list[str],
    covered: list[bool],
    start: int,
    end: int,
    *,
    symbol: str | None,
    node_type: str | None,
) -> list[_Span]:
    """Spans for uncovered runs in [start, end], blank edges trimmed.

    Marks produced lines as covered, so a later pass over an enclosing
    range cannot emit them twice. Blank-only runs yield nothing.
    """
    spans: list[_Span] = []
    for run_start, run_end in _uncovered_runs(covered, start, end):
        while run_start <= run_end and not lines[run_start - 1].strip():
            run_start += 1
        while run_end >= run_start and not lines[run_end - 1].strip():
            run_end -= 1
        if run_start > run_end:
            continue
        _cover(covered, run_start, run_end)
        spans.append(
            _Span(start_line=run_start, end_line=run_end, symbol=symbol, node_type=node_type)
        )
    return spans


def _assemble(
    spans: list[_Span], lines: list[str], *, repo: str, path: str, lang: str, kind: Kind
) -> list[Chunk]:
    """Turn spans into Chunks in file order; text is a verbatim line slice."""
    return [
        Chunk(
            repo=repo,
            path=path,
            lang=lang,
            kind=kind,
            symbol=span.symbol,
            node_type=span.node_type,
            start_line=span.start_line,
            end_line=span.end_line,
            text="\n".join(lines[span.start_line - 1 : span.end_line]),
        )
        for span in sorted(spans, key=lambda span: span.start_line)
    ]


def _unwrap(node: Node, *, wrapper: str, inner: tuple[str, ...]) -> Node:
    """Return the definition inside a wrapper node, the node itself otherwise.

    Wrappers differ per grammar: python hides defs in decorated_definition,
    typescript — in export_statement.
    """
    if node.type == wrapper:
        for child in node.named_children:
            if child.type in inner:
                return child
    return node


def _name(node: Node) -> str | None:
    """Identifier of a def/class; None on error-recovery trees with no name."""
    child = node.child_by_field_name("name")
    if child is None or child.text is None:
        return None
    return child.text.decode()


def _def_span(outer: Node, covered: list[bool], *, symbol: str, node_type: str) -> _Span:
    """Span of one definition; `outer` includes decorators or the export keyword."""
    start, end = _line_span(outer)
    _cover(covered, start, end)
    return _Span(start_line=start, end_line=end, symbol=symbol, node_type=node_type)


def _ast_chunks(
    parsers: dict[str, Parser],
    extractors: dict[str, Callable[[Node, list[str], list[bool]], list[_Span]]],
    text: str,
    *,
    repo: str,
    path: str,
    lang: str,
    kind: Kind,
) -> list[Chunk]:
    """Shared skeleton: parse, run the per-language extractor, fill the gaps."""
    parser = parsers.get(lang)
    if parser is None:
        raise RuntimeError(f"no grammar for {lang!r} — run `uv sync --extra ast`")
    lines = text.splitlines()
    covered = [False] * (len(lines) + 1)
    root = parser.parse(text.encode()).root_node
    spans = extractors[lang](root, lines, covered)
    spans += _gap_spans(lines, covered, 1, len(lines), symbol=None, node_type=None)
    return _assemble(spans, lines, repo=repo, path=path, lang=lang, kind=kind)
