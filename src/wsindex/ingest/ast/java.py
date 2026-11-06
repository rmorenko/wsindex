"""Java policy: annotations come for free, classes descend into members."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import Span, def_span, gap_spans, line_span, symbol_name

if TYPE_CHECKING:
    from tree_sitter import Node

_JAVA_DEFS = ("interface_declaration", "enum_declaration", "record_declaration")


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Classes descend into methods and constructors ("Cls.method");
    interfaces, enums and records stay whole. Annotations live inside the
    declaration nodes (the modifiers child), so spans include them for
    free; javadoc comments are siblings and stay in gap chunks — accepted
    debt, like JSDoc for typescript.
    """
    spans: list[Span] = []
    for child in root.named_children:
        if child.type in _JAVA_DEFS:
            name = symbol_name(child)
            if name is not None:
                spans.append(def_span(child, covered=covered, symbol=name, node_type=child.type))
        elif child.type == "class_declaration":
            cls_name = symbol_name(child)
            body = child.child_by_field_name("body")
            if cls_name is None or body is None:
                continue  # error-recovery leftovers fall through to gap chunks
            cls_start, cls_end = line_span(child)
            for member in body.named_children:
                if member.type not in ("method_declaration", "constructor_declaration"):
                    continue
                member_name = symbol_name(member)
                if member_name is not None:
                    spans.append(
                        def_span(
                            member,
                            covered=covered,
                            symbol=f"{cls_name}.{member_name}",
                            node_type=member.type,
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
    return spans
