"""Rust policy: prelude siblings stick to definitions, impls act as classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import _cover, _gap_spans, _line_span, _name, _Span

if TYPE_CHECKING:
    from tree_sitter import Node

_RUST_DEFS = ("function_item", "struct_item", "enum_item", "trait_item")


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


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
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
