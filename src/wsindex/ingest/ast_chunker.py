"""AST chunking: Python/Rust/TypeScript/Java code plus TOML/YAML/JSON/Dockerfile configs.

The mechanism is shared and language-agnostic: definition nodes become
spans, then a line-coverage pass turns everything else into gap chunks, so
each non-blank line lands in exactly one chunk. Per-language policy (which
nodes matter and where the symbol comes from) lives in small extractors.
Grammars are optional: CODE_PARSERS and CONFIG_PARSERS hold whichever
grammars imported successfully, and the dispatcher falls back to the text
chunker for everything else. Files with syntax errors are chunked from
whatever the error-tolerant parser recognized.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

from wsindex.model import Chunk, Kind

_PY_DEFS = ("function_definition", "class_definition")
_RUST_DEFS = ("function_item", "struct_item", "enum_item", "trait_item")
_JAVA_DEFS = ("interface_declaration", "enum_declaration", "record_declaration")

_TS_DEFS = (
    "function_declaration",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "lexical_declaration",
)

try:
    from tree_sitter import Language, Node, Parser

    HAS_TREE_SITTER = True
except ImportError:  # pragma: no cover - only reachable on a base install (CI matrix)
    HAS_TREE_SITTER = False


def _load_parsers(table: tuple[tuple[str, str, str], ...]) -> dict[str, Parser]:
    parsers: dict[str, Parser] = {}
    for lang, module_name, getter in table:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover
            continue
        parsers[lang] = Parser(Language(getattr(module, getter)()))
    return parsers


CODE_PARSERS: dict[str, Parser] = {}

CONFIG_PARSERS: dict[str, Parser] = {}

if HAS_TREE_SITTER:
    CODE_PARSERS = _load_parsers(
        (
            ("python", "tree_sitter_python", "language"),
            ("rust", "tree_sitter_rust", "language"),
            ("typescript", "tree_sitter_typescript", "language_typescript"),
            ("java", "tree_sitter_java", "language"),
        )
    )

    CONFIG_PARSERS = _load_parsers(
        (
            ("toml", "tree_sitter_toml", "language"),
            ("yaml", "tree_sitter_yaml", "language"),
            ("json", "tree_sitter_json", "language"),
            ("dockerfile", "tree_sitter_dockerfile", "language"),
        )
    )


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


def _python_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
    """Functions and methods with a qualified symbol ("Cls.method"); class
    lines not taken by methods (header, docstring, attributes) carry the
    class name. Oversized functions stay whole on purpose — accepted MVP debt.
    """
    spans: list[_Span] = []
    for child in root.named_children:
        inner = _unwrap(node=child, wrapper="decorated_definition", inner=_PY_DEFS)
        if inner.type == "function_definition":
            name = _name(inner)
            if name is not None:
                spans.append(
                    _def_span(child, covered, symbol=name, node_type="function_definition")
                )
        elif inner.type == "class_definition":
            cls_name = _name(inner)
            body = inner.child_by_field_name("body")
            if cls_name is None or body is None:
                continue  # error-recovery leftovers fall through to gap chunks
            cls_start, cls_end = _line_span(child)
            for stmt in body.named_children:
                method = _unwrap(stmt, wrapper="decorated_definition", inner=_PY_DEFS)
                if method.type != "function_definition":
                    continue
                method_name = _name(method)
                if method_name is not None:
                    spans.append(
                        _def_span(
                            stmt,
                            covered,
                            symbol=f"{cls_name}.{method_name}",
                            node_type="function_definition",
                        )
                    )
            spans += _gap_spans(
                lines, covered, cls_start, cls_end, symbol=cls_name, node_type="class_definition"
            )
    return spans


def _extend_back(siblings: list[Node], index: int, start_line: int) -> int:
    """Attach contiguous preceding attribute/doc-comment siblings."""
    for prev in reversed(siblings[:index]):
        if prev.type not in ("attribute_item", "line_comment"):
            break
        prev_start, prev_end = _line_span(prev)
        if prev_end < start_line - 1:
            break  # a blank line breaks the chain
        start_line = prev_start
    return start_line


def _rust_def_span(siblings: list[Node], index: int, covered: list[bool], *, symbol: str) -> _Span:
    """Span of one rust definition, extended back over its attribute/doc prelude."""
    node = siblings[index]
    start, end = _line_span(node)
    start = _extend_back(siblings=siblings, index=index, start_line=start)
    _cover(covered, start, end)
    return _Span(start_line=start, end_line=end, symbol=symbol, node_type=node.type)


def _rust_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
    """Free functions, types and impl methods; #[attributes] and /// docs
    stick to the definition below them; method symbols use the native
    "Type::method" separator. An impl block is handled like a python class:
    methods first, leftover lines (header, closing brace) carry the type
    name — trait impls are named by the implementing type, the trait itself
    is ignored on purpose.
    """
    spans: list[_Span] = []
    children = root.named_children
    for i, child in enumerate(children):
        if child.type in _RUST_DEFS:
            name = _name(child)
            if name is not None:
                spans.append(_rust_def_span(children, i, covered, symbol=name))
        elif child.type == "impl_item":
            type_node = child.child_by_field_name("type")
            body = child.child_by_field_name("body")
            if type_node is None or type_node.text is None or body is None:
                continue
            type_name = type_node.text.decode()
            impl_start, impl_end = _line_span(child)
            members = body.named_children
            for j, member in enumerate(members):
                if member.type != "function_item":
                    continue
                method_name = _name(member)
                if method_name is not None:
                    spans.append(
                        _rust_def_span(members, j, covered, symbol=f"{type_name}::{method_name}")
                    )
            spans += _gap_spans(
                lines, covered, impl_start, impl_end, symbol=type_name, node_type="impl_item"
            )
    return spans


def _ts_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
    """Exported and plain declarations; the export keyword stays inside the
    chunk. Classes follow the python pattern: methods become "Cls.method"
    chunks, leftover class lines carry the class name. A lexical_declaration
    is a chunk only when it binds an arrow function or function expression.
    Deliberate gaps: anonymous default exports and JSDoc comments (no
    look-behind for TS yet).
    """
    spans: list[_Span] = []
    for child in root.named_children:
        node = _unwrap(node=child, wrapper="export_statement", inner=_TS_DEFS)
        if node.type in (
            "function_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        ):
            name = _name(node)
            if name is not None:
                spans.append(_def_span(child, covered, symbol=name, node_type=node.type))
        elif node.type == "class_declaration":
            cls_name = _name(node)
            body = node.child_by_field_name("body")
            if cls_name is None or body is None:
                continue  # error-recovery leftovers fall through to gap chunks
            cls_start, cls_end = _line_span(child)
            for member in body.named_children:
                if member.type != "method_definition":
                    continue
                method_name = _name(member)
                if method_name is not None:
                    spans.append(
                        _def_span(
                            member,
                            covered,
                            symbol=f"{cls_name}.{method_name}",
                            node_type="method_definition",
                        )
                    )
            spans += _gap_spans(
                lines, covered, cls_start, cls_end, symbol=cls_name, node_type="class_declaration"
            )
        elif node.type == "lexical_declaration":
            declarator = next(
                (c for c in node.named_children if c.type == "variable_declarator"), None
            )
            if declarator is None:
                continue
            value = declarator.child_by_field_name("value")
            if value is None or value.type not in ("arrow_function", "function_expression"):
                continue  # a plain constant is a legitimate gap
            name = _name(declarator)
            if name is not None:
                spans.append(
                    _def_span(child, covered, symbol=name, node_type="lexical_declaration")
                )
    return spans


def _java_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
    """Classes descend into methods and constructors ("Cls.method");
    interfaces, enums and records stay whole. Annotations live inside the
    declaration nodes (the modifiers child), so spans include them for
    free; javadoc comments are siblings and stay in gap chunks — accepted
    debt, like JSDoc for typescript.
    """
    spans: list[_Span] = []
    for child in root.named_children:
        if child.type in _JAVA_DEFS:
            name = _name(child)
            if name is not None:
                spans.append(_def_span(child, covered, symbol=name, node_type=child.type))
        elif child.type == "class_declaration":
            cls_name = _name(child)
            body = child.child_by_field_name("body")
            if cls_name is None or body is None:
                continue  # error-recovery leftovers fall through to gap chunks
            cls_start, cls_end = _line_span(child)
            for member in body.named_children:
                if member.type not in ("method_declaration", "constructor_declaration"):
                    continue
                member_name = _name(member)
                if member_name is not None:
                    spans.append(
                        _def_span(
                            member,
                            covered,
                            symbol=f"{cls_name}.{member_name}",
                            node_type=member.type,
                        )
                    )
            spans += _gap_spans(
                lines, covered, cls_start, cls_end, symbol=cls_name, node_type="class_declaration"
            )
    return spans


_CODE_EXTRACTORS: dict[str, Callable[[Node, list[str], list[bool]], list[_Span]]] = {
    "python": _python_spans,
    "rust": _rust_spans,
    "typescript": _ts_spans,
    "java": _java_spans,
}


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


def chunk_config(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Chunk a config file: one chunk per TOML table, top-level YAML/JSON
    key or Dockerfile stage; comments and leftovers become gap chunks.
    """
    return _ast_chunks(
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
    return _ast_chunks(
        parsers=CODE_PARSERS,
        extractors=_CODE_EXTRACTORS,
        text=text,
        repo=repo,
        path=path,
        lang=lang,
        kind=kind,
    )
