"""AST chunking for Python files: functions, methods and class headers.

Spans come from tree-sitter definition nodes (decorators included); every
line outside a definition joins a gap chunk, so each non-blank line of the
file lands in exactly one chunk. Files with syntax errors are chunked from
whatever the error-tolerant parser recognized. Requires the optional `ast`
extra; callers check HAS_TREE_SITTER and fall back to the text chunker.
"""

from __future__ import annotations

from dataclasses import dataclass

from wsindex.model import Chunk, Kind

try:
    import tree_sitter_python
    from tree_sitter import Language, Node, Parser

    HAS_TREE_SITTER = True
    _PARSER = Parser(Language(tree_sitter_python.language()))
except ImportError:
    HAS_TREE_SITTER = False


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


def _unwrap(node: Node) -> Node:
    """Return the def/class inside a decorated_definition, the node itself otherwise."""
    if node.type == "decorated_definition":
        for child in node.named_children:
            if child.type in ("function_definition", "class_definition"):
                return child
    return node


def _name(node: Node) -> str | None:
    """Identifier of a def/class; None on error-recovery trees with no name."""
    child = node.child_by_field_name("name")
    if child is None or child.text is None:
        return None
    return child.text.decode()


def _cover(covered: list[bool], start: int, end: int) -> None:
    for i in range(start, end + 1):
        covered[i] = True


def _def_span(outer: Node, covered: list[bool], symbol: str) -> _Span:
    """Span of one function or method; `outer` includes the decorators."""
    start, end = _line_span(outer)
    _cover(covered, start, end)
    return _Span(start_line=start, end_line=end, symbol=symbol, node_type="function_definition")


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


def chunk_python(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Chunk a Python source file by its AST.

    Functions and methods become chunks with a qualified `symbol`
    ("Cls.method") and `node_type`; class lines not taken by methods
    (header, docstring, attributes) carry the class name; the module-level
    remainder (docstring, imports, constants) becomes plain gap chunks.
    Oversized functions stay whole on purpose — accepted MVP debt.
    """
    if not HAS_TREE_SITTER:
        raise RuntimeError("tree-sitter is not installed — run `uv sync --extra ast`")
    lines = text.splitlines()
    covered = [False] * (len(lines) + 1)
    root = _PARSER.parse(text.encode()).root_node
    spans: list[_Span] = []
    for child in root.named_children:
        inner = _unwrap(child)
        if inner.type == "function_definition":
            name = _name(inner)
            if name is not None:
                spans.append(_def_span(child, covered, symbol=name))
        elif inner.type == "class_definition":
            cls_name = _name(inner)
            body = inner.child_by_field_name("body")
            if cls_name is None or body is None:
                continue  # error-recovery leftovers fall through to gap chunks
            cls_start, cls_end = _line_span(child)
            for stmt in body.named_children:
                method = _unwrap(stmt)
                if method.type != "function_definition":
                    continue
                method_name = _name(method)
                if method_name is not None:
                    spans.append(_def_span(stmt, covered, symbol=f"{cls_name}.{method_name}"))
            spans += _gap_spans(
                lines, covered, cls_start, cls_end, symbol=cls_name, node_type="class_definition"
            )
    spans += _gap_spans(lines, covered, 1, len(lines), symbol=None, node_type=None)
    spans.sort(key=lambda span: span.start_line)
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
        for span in spans
    ]
