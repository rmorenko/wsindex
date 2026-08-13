"""Config policies: TOML tables, YAML/JSON top-level keys, Dockerfile stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import _cover, _line_span, _Span

if TYPE_CHECKING:
    from tree_sitter import Node


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


def toml_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
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


def yaml_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
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


def json_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
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


def dockerfile_spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
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
