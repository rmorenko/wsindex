"""C++ policy: namespaces recurse, classes act like python classes.

Two things make C++ unlike its ancestor, and both are structural rather
than cosmetic:

**Namespaces nest the whole file.** A file wrapped in `namespace app {}`
has exactly one top-level node, so the flat loop every other language
uses finds nothing. The walk descends into namespace bodies instead — and
into `extern "C" {}`, which nests the same way.

**A method has two places to live.** Declared inside the class body and
defined outside it, `void Server::serve(...)`, and the grammar gives the
out-of-line form a `qualified_identifier` — `Server::serve` already
spelled with C++'s own separator. Inline methods are qualified by hand to
match, so both halves of a class answer the same `--symbol Server` query.

Templates wrap the thing they parameterize, so the span covers the
`template <...>` line and the name comes from inside — the same unwrap
python does for decorators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.c import declarator_name
from wsindex.ingest.ast.core import Span, def_span, gap_spans, line_span, symbol_name

if TYPE_CHECKING:
    from tree_sitter import Node

_CONTAINERS = ("namespace_definition", "linkage_specification")
"""Nodes that hold a body of further declarations rather than declaring
anything indexable themselves."""

_TAGGED_TYPES = ("struct_specifier", "union_specifier", "enum_specifier")

_MAX_NESTING = 8
"""How deep the namespace recursion goes. Real code nests two or three
levels; a bound keeps a pathological or damaged tree from costing the
whole run."""


def _unwrap_template(node: Node) -> Node:
    """The definition a `template <...>` parameterizes, or the node itself."""
    if node.type != "template_declaration":
        return node
    for child in node.named_children:
        if child.type in ("function_definition", "class_specifier", *_TAGGED_TYPES):
            return child
    return node


def _class_spans(outer: Node, inner: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Methods as their own chunks, the rest of the class as one more.

    The python pattern: each method becomes `Class::method`, and whatever
    class lines nothing claimed — the header, access specifiers, data
    members — carry the class name so they stay findable.

    Args:
        outer: Node to span; the `template_declaration` when there is one,
            so `template <typename T>` lands inside the chunk.
        inner: The `class_specifier`/`struct_specifier` itself.
        lines: The file's lines, for the gap pass.
        covered: Shared line-coverage bookkeeping.

    Returns:
        Spans for the methods plus one for the class remainder.
    """
    name = symbol_name(inner)
    body = inner.child_by_field_name("body")
    if name is None or body is None:
        return []  # error-recovery leftovers fall through to gap chunks
    start, end = line_span(outer)
    found: list[Span] = []
    for member in body.named_children:
        # `function_definition` is an inline method; `field_declaration`
        # with a function declarator is one declared here and defined
        # elsewhere — that one is left to its out-of-line definition,
        # which carries the body worth searching.
        if member.type != "function_definition":
            continue
        method = declarator_name(member)
        if method is not None:
            found.append(
                def_span(
                    member,
                    covered=covered,
                    symbol=f"{name}::{method}",
                    node_type="function_definition",
                )
            )
    found += gap_spans(
        lines, covered=covered, start=start, end=end, symbol=name, node_type=inner.type
    )
    return found


def _declarations(
    nodes: list[Node], lines: list[str], covered: list[bool], depth: int
) -> list[Span]:
    """Claim the definitions in one scope, descending into nested ones."""
    if depth > _MAX_NESTING:
        return []
    found: list[Span] = []
    for child in nodes:
        if child.type in _CONTAINERS:
            body = child.child_by_field_name("body")
            if body is not None:
                # The namespace's own braces stay uncovered and become a
                # gap chunk, exactly like a python class header does.
                found += _declarations(body.named_children, lines, covered, depth + 1)
            continue
        inner = _unwrap_template(child)
        if inner.type in ("class_specifier", "struct_specifier"):
            found += _class_spans(child, inner, lines, covered)
        elif inner.type == "function_definition":
            name = declarator_name(inner)
            if name is not None:
                found.append(
                    def_span(child, covered=covered, symbol=name, node_type="function_definition")
                )
        elif inner.type in _TAGGED_TYPES:
            name = symbol_name(inner)
            if name is not None:
                found.append(def_span(child, covered=covered, symbol=name, node_type=inner.type))
    return found


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Functions, classes and tagged types, through any nesting of namespaces.

    Args:
        root: Root of the parsed file; may be partial.
        lines: The file's lines, for the class-remainder gap pass.
        covered: Shared line-coverage bookkeeping.

    Returns:
        Spans for every definition found at any namespace depth.
    """
    return _declarations(root.named_children, lines, covered, depth=0)
