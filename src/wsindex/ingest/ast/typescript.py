"""TypeScript policy: export wrappers unwrap, arrow consts count as functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import Span, def_span, symbol_name, unwrap
from wsindex.ingest.ast.nested import NestedPolicy, extractor

if TYPE_CHECKING:
    from tree_sitter import Node

POLICY = NestedPolicy(
    types=("class_declaration",),
    members=("method_definition",),
    standalone=(
        "function_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    ),
    wrappers=("export_statement",),
)
"""The export keyword stays inside the chunk: `export function f` is one
declaration, and a chunk holding only the word `export` would be found by
nobody."""

_FUNCTION_VALUES = ("arrow_function", "function_expression")
"""What makes a `const` a definition rather than a constant. This is the
one rule the shared walk cannot express, because it is about a node's
*value* and not its type."""

_declarations = extractor(POLICY)


def _const_function(child: Node, covered: list[bool]) -> Span | None:
    """A `const f = () => ...` as a definition; None for a plain constant."""
    node = unwrap(node=child, wrapper="export_statement", inner=("lexical_declaration",))
    if node.type != "lexical_declaration":
        return None
    declarator = next((c for c in node.named_children if c.type == "variable_declarator"), None)
    if declarator is None:
        return None
    value = declarator.child_by_field_name("value")
    if value is None or value.type not in _FUNCTION_VALUES:
        return None  # a plain constant is a legitimate gap
    name = symbol_name(declarator)
    if name is None:
        return None
    return def_span(child, covered=covered, symbol=name, node_type="lexical_declaration")


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Declarations by the shared walk, plus the consts that are functions.

    Two passes rather than one branchy loop: everything typescript shares
    with python and java goes through the same code those use, and what
    is left is the single rule that is typescript's own.

    Deliberate gaps: anonymous default exports, and JSDoc comments (no
    look-behind for TS yet).
    """
    found = _declarations(root, lines, covered)
    found += [
        span
        for span in (_const_function(child, covered) for child in root.named_children)
        if span is not None
    ]
    return found
