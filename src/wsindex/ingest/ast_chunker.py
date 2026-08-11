"""AST chunking: Python code plus TOML/YAML/JSON/Dockerfile configs.

The mechanism is shared and language-agnostic: definition nodes become
spans, then a line-coverage pass turns everything else into gap chunks, so
each non-blank line lands in exactly one chunk. Per-language policy (which
nodes matter and where the symbol comes from) lives in small extractors.
Grammars are optional: HAS_TREE_SITTER gates Python, CONFIG_PARSERS holds
whichever config grammars imported successfully; callers fall back to the
text chunker for everything else. Files with syntax errors are chunked
from whatever the error-tolerant parser recognized.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

from wsindex.model import Chunk, Kind

try:
    import tree_sitter_python
    from tree_sitter import Language, Node, Parser

    HAS_TREE_SITTER = True
    _PARSER = Parser(Language(tree_sitter_python.language()))
except ImportError:  # pragma: no cover - only reachable on a base install (CI matrix)
    HAS_TREE_SITTER = False

# Registry built from whichever grammars imported: degradation is
# per-grammar, the dispatcher just asks `lang in CONFIG_PARSERS`.
CONFIG_PARSERS: dict[str, Parser] = {}
if HAS_TREE_SITTER:
    for _lang, _module in (
        ("toml", "tree_sitter_toml"),
        ("yaml", "tree_sitter_yaml"),
        ("json", "tree_sitter_json"),
        ("dockerfile", "tree_sitter_dockerfile"),
    ):
        try:
            _grammar = importlib.import_module(_module)
        except ImportError:  # pragma: no cover - partial grammar install
            continue
        CONFIG_PARSERS[_lang] = Parser(Language(_grammar.language()))


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


# --- Python ---------------------------------------------------------------


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


def _def_span(outer: Node, covered: list[bool], symbol: str) -> _Span:
    """Span of one function or method; `outer` includes the decorators."""
    start, end = _line_span(outer)
    _cover(covered, start, end)
    return _Span(start_line=start, end_line=end, symbol=symbol, node_type="function_definition")


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
    return _assemble(spans, lines, repo=repo, path=path, lang=lang, kind=kind)


# --- Configs --------------------------------------------------------------


def _config_span(
    node: Node, lines: list[str], covered: list[bool], *, symbol: str | None, node_type: str
) -> _Span:
    """Span of one config unit.

    Container nodes (a TOML table, a YAML mapping) often swallow trailing
    blank separator lines — trim them so chunks end on content.
    """
    start, end = _line_span(node)
    while end > start and not lines[end - 1].strip():
        end -= 1
    _cover(covered, start, end)
    return _Span(start_line=start, end_line=end, symbol=symbol, node_type=node_type)


def _toml_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
    """Top-level tables and arrays of tables; leading bare pairs go to gaps."""
    spans: list[_Span] = []
    for child in root.named_children:
        if child.type not in ("table", "table_array_element"):
            continue
        symbol: str | None = None
        key = next(
            (c for c in child.named_children if c.type in ("bare_key", "dotted_key", "quoted_key")),
            None,
        )
        if key is not None and key.text is not None:
            symbol = key.text.decode()
        spans.append(_config_span(child, lines, covered, symbol=symbol, node_type=child.type))
    return spans


def _yaml_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
    """Top-level mapping pairs of every document in the stream."""
    spans: list[_Span] = []
    for document in root.named_children:
        for block_node in document.named_children:
            for mapping in block_node.named_children:
                if mapping.type != "block_mapping":
                    continue
                for pair in mapping.named_children:
                    if pair.type != "block_mapping_pair":
                        continue
                    symbol: str | None = None
                    key = pair.child_by_field_name("key")
                    if key is not None and key.text is not None:
                        symbol = key.text.decode()
                    spans.append(
                        _config_span(pair, lines, covered, symbol=symbol, node_type=pair.type)
                    )
    return spans


def _json_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
    """Top-level object pairs; a top-level array yields no spans (gaps take over)."""
    spans: list[_Span] = []
    for top in root.named_children:
        if top.type != "object":
            continue
        for pair in top.named_children:
            if pair.type != "pair":
                continue
            symbol: str | None = None
            key = pair.child_by_field_name("key")
            if key is not None:
                content = next((c for c in key.named_children if c.type == "string_content"), None)
                if content is not None and content.text is not None:
                    symbol = content.text.decode()
            spans.append(_config_span(pair, lines, covered, symbol=symbol, node_type=pair.type))
    return spans


def _stage_symbol(node: Node) -> str | None:
    """Stage name of a FROM instruction: the AS alias, else the image spec."""
    for wanted in ("image_alias", "image_spec"):
        child = next((c for c in node.named_children if c.type == wanted), None)
        if child is not None and child.text is not None:
            return child.text.decode()
    return None


def _dockerfile_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
    """One span per build stage: FROM up to the last instruction before the next FROM.

    `node_type` is the synthetic "stage" — unlike other extractors the unit
    here is a group of nodes, not a single grammar node. Instructions before
    the first FROM (global ARGs) fall through to gap chunks.
    """
    spans: list[_Span] = []
    start: int | None = None
    end = 0
    symbol: str | None = None

    def close() -> None:
        if start is not None:
            _cover(covered, start, end)
            spans.append(_Span(start_line=start, end_line=end, symbol=symbol, node_type="stage"))

    for node in root.named_children:
        if not node.type.endswith("_instruction"):
            continue
        node_start, node_end = _line_span(node)
        if node.type == "from_instruction":
            close()
            start, symbol = node_start, _stage_symbol(node)
        if start is not None:
            end = node_end
    close()
    return spans


_CONFIG_EXTRACTORS: dict[str, Callable[[Node, list[str], list[bool]], list[_Span]]] = {
    "toml": _toml_spans,
    "yaml": _yaml_spans,
    "json": _json_spans,
    "dockerfile": _dockerfile_spans,
}


def chunk_config(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Chunk a config file by its top-level structure.

    One chunk per TOML table, top-level YAML/JSON key or Dockerfile stage,
    symbol = table name / key / stage alias. Everything else — comments,
    leading pairs, a top-level JSON array — becomes plain gap chunks.
    """
    parser = CONFIG_PARSERS.get(lang)
    if parser is None:
        raise RuntimeError(f"no config grammar for {lang!r} — run `uv sync --extra ast`")
    lines = text.splitlines()
    covered = [False] * (len(lines) + 1)
    root = parser.parse(text.encode()).root_node
    spans = _CONFIG_EXTRACTORS[lang](root, lines, covered)
    spans += _gap_spans(lines, covered, 1, len(lines), symbol=None, node_type=None)
    return _assemble(spans, lines, repo=repo, path=path, lang=lang, kind=kind)
