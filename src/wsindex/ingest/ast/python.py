"""Python policy: decorated defs unwrap, methods get qualified symbols."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import _def_span, _gap_spans, _line_span, _name, _Span, _unwrap

if TYPE_CHECKING:
    from tree_sitter import Node

_PY_DEFS = ("function_definition", "class_definition")


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[_Span]:
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
