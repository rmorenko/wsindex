"""TypeScript policy: export wrappers unwrap, arrow consts count as functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import Span, def_span, gap_spans, line_span, symbol_name, unwrap

if TYPE_CHECKING:
    from tree_sitter import Node

_TS_DEFS = (
    "function_declaration",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "lexical_declaration",
)


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Exported and plain declarations; the export keyword stays inside the
    chunk. Classes follow the python pattern: methods become "Cls.method"
    chunks, leftover class lines carry the class name. A lexical_declaration
    is a chunk only when it binds an arrow function or function expression.
    Deliberate gaps: anonymous default exports and JSDoc comments (no
    look-behind for TS yet).
    """
    spans: list[Span] = []
    for child in root.named_children:
        node = unwrap(node=child, wrapper="export_statement", inner=_TS_DEFS)
        if node.type in (
            "function_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        ):
            name = symbol_name(node)
            if name is not None:
                spans.append(def_span(child, covered=covered, symbol=name, node_type=node.type))
        elif node.type == "class_declaration":
            cls_name = symbol_name(node)
            body = node.child_by_field_name("body")
            if cls_name is None or body is None:
                continue  # error-recovery leftovers fall through to gap chunks
            cls_start, cls_end = line_span(child)
            for member in body.named_children:
                if member.type != "method_definition":
                    continue
                method_name = symbol_name(member)
                if method_name is not None:
                    spans.append(
                        def_span(
                            member,
                            covered=covered,
                            symbol=f"{cls_name}.{method_name}",
                            node_type="method_definition",
                        )
                    )
            spans += gap_spans(
                lines,
                covered=covered,
                start=cls_start,
                end=cls_end,
                symbol=cls_name,
                node_type="class_declaration",
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
            name = symbol_name(declarator)
            if name is not None:
                spans.append(
                    def_span(child, covered=covered, symbol=name, node_type="lexical_declaration")
                )
    return spans
