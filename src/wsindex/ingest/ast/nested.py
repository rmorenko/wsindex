"""One extractor for the languages shaped like "a class holds methods".

C#, Kotlin, PHP and Ruby differ in almost everything except the shape
that matters here: something optional wraps the file (a namespace, a
module), types hold members, and some declarations stand alone. Writing
that walk four times would have been four chances to get the gap pass
subtly wrong — the bug that shows up as a line silently missing from the
index, or counted twice.

So the walk is written once and each language supplies a `NestedPolicy`:
which node types are containers, which are types, which are members, and
what separator the language writes between a type and its method. The
policy is data; the only code per language is the module that declares it.

The three roles, and why the distinction earns its keep:

- **Containers** (`namespace`, `module`) hold declarations but are not
  worth a chunk of their own. The walk descends and their braces fall to
  the gap pass. Without this a C#-style file — one namespace wrapping
  everything — would yield exactly one chunk.
- **Types** (`class`, `object`, `trait`) get the python treatment:
  each member becomes its own chunk named `Type<sep>member`, and whatever
  class lines nothing claimed (the header, fields, access modifiers)
  become one more chunk carrying the type name, so they stay findable.
- **Standalone** (`interface`, `enum`, a top-level function) is claimed
  whole. A node type may be standalone *and* a member: a Ruby `def` is a
  method inside a class and a function outside one.

Deliberately not handled: a type nested inside another type. Its members
fall into the outer type's remainder chunk, which is what python already
does with a class inside a class — still indexed, just not separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import (
    Span,
    SpanExtractor,
    def_span,
    gap_spans,
    line_span,
    symbol_name,
)

if TYPE_CHECKING:
    from tree_sitter import Node

_MAX_NESTING = 8
"""How deep container recursion goes. Real code nests two or three
levels; a bound keeps a pathological or damaged tree from costing a run."""


@dataclass(frozen=True, kw_only=True)
class NestedPolicy:
    """What one language calls each of the three roles.

    Attributes:
        containers: Node types that wrap declarations without deserving a
            chunk — the walk descends into them.
        types: Node types whose members become chunks of their own.
        members: Node types inside a `types` body that are worth a chunk,
            qualified with the type's name.
        standalone: Node types claimed whole, wherever they are found.
        separator: What the language writes between a type and its
            member — `.` for most, `::` where that is the native form.
        body_types: Named-child types to look in when a node has no
            `body` field. Kotlin keeps members in a `class_body` child
            rather than a field, and this is how that is spelled.
    """

    containers: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    standalone: tuple[str, ...] = ()
    separator: str = "."
    body_types: tuple[str, ...] = ()


def _body(node: Node, policy: NestedPolicy) -> Node | None:
    """The node holding a container's or type's declarations, if any."""
    found = node.child_by_field_name("body")
    if found is not None:
        return found
    for child in node.named_children:
        if child.type in policy.body_types:
            return child
    return None


def _type_spans(
    node: Node, policy: NestedPolicy, lines: list[str], covered: list[bool]
) -> list[Span]:
    """Members as their own chunks, the rest of the type as one more."""
    name = symbol_name(node)
    body = _body(node, policy)
    if name is None or body is None:
        return []  # error-recovery leftovers fall through to gap chunks
    start, end = line_span(node)
    found: list[Span] = []
    for member in body.named_children:
        if member.type not in policy.members:
            continue
        member_name = symbol_name(member)
        if member_name is not None:
            found.append(
                def_span(
                    member,
                    covered=covered,
                    symbol=f"{name}{policy.separator}{member_name}",
                    node_type=member.type,
                )
            )
    found += gap_spans(
        lines, covered=covered, start=start, end=end, symbol=name, node_type=node.type
    )
    return found


def _walk(
    nodes: list[Node], policy: NestedPolicy, lines: list[str], covered: list[bool], depth: int
) -> list[Span]:
    if depth > _MAX_NESTING:
        return []
    found: list[Span] = []
    for child in nodes:
        if child.type in policy.containers:
            body = _body(child, policy)
            if body is not None:
                found += _walk(body.named_children, policy, lines, covered, depth + 1)
        elif child.type in policy.types:
            found += _type_spans(child, policy, lines, covered)
        elif child.type in policy.standalone:
            name = symbol_name(child)
            if name is not None:
                found.append(def_span(child, covered=covered, symbol=name, node_type=child.type))
    return found


def extractor(policy: NestedPolicy) -> SpanExtractor:
    """Build the `spans` function for a language with this shape.

    Args:
        policy: What this language calls containers, types and members.

    Returns:
        A `SpanExtractor` — the callable a `LanguageSpec` takes.
    """

    def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
        return _walk(root.named_children, policy, lines, covered, depth=0)

    return spans
