"""Python policy: decorated defs unwrap, methods get qualified symbols."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import Span, def_span, gap_spans, line_span, symbol_name, unwrap

if TYPE_CHECKING:
    from tree_sitter import Node

_PY_DEFS = ("function_definition", "class_definition")


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Functions and methods with a qualified symbol ("Cls.method"); class
    lines not taken by methods (header, docstring, attributes) carry the
    class name. Oversized functions stay whole on purpose — accepted MVP debt.
    """
    spans: list[Span] = []
    for child in root.named_children:
        inner = unwrap(node=child, wrapper="decorated_definition", inner=_PY_DEFS)
        if inner.type == "function_definition":
            name = symbol_name(inner)
            if name is not None:
                spans.append(
                    def_span(child, covered=covered, symbol=name, node_type="function_definition")
                )
        elif inner.type == "class_definition":
            cls_name = symbol_name(inner)
            body = inner.child_by_field_name("body")
            if cls_name is None or body is None:
                continue  # error-recovery leftovers fall through to gap chunks
            cls_start, cls_end = line_span(child)
            for stmt in body.named_children:
                method = unwrap(stmt, wrapper="decorated_definition", inner=_PY_DEFS)
                if method.type != "function_definition":
                    continue
                method_name = symbol_name(method)
                if method_name is not None:
                    spans.append(
                        def_span(
                            stmt,
                            covered=covered,
                            symbol=f"{cls_name}.{method_name}",
                            node_type="function_definition",
                        )
                    )
            spans += gap_spans(
                lines,
                covered=covered,
                start=cls_start,
                end=cls_end,
                symbol=cls_name,
                node_type="class_definition",
            )
    return spans
