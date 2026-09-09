"""One extractor for the languages shaped like "a class holds methods".

Python, Java, Rust, TypeScript, C#, Kotlin, PHP and Ruby differ in almost
everything except the shape that matters here: something optional wraps
the file, types hold members, and some declarations stand alone. Writing
that walk eight times would have been eight chances to get the gap pass
subtly wrong — the bug that shows up as a line missing from the index.

So the walk is written once and each language supplies a `NestedPolicy`:
which node types are containers, types and members, what separates a type
from its member, what wraps a definition and what precedes it. The policy
is data; the only code per language is the module that declares it.

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
    gap_spans,
    line_span,
    mark_covered,
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
        wrappers: Node types that hold a definition rather than being
            one — python's `decorated_definition`, typescript's
            `export_statement`. The chunk is the wrapper (a decorator
            belongs with what it decorates); the name and the node type
            come from what is inside.
        type_name_field: Which field holds a type's name. `name` almost
            everywhere; rust's `impl_item` calls it `type`, and that is
            the whole of the difference.
        preamble: Sibling types that belong to the definition below
            them, when nothing blank separates the two: rust's
            `attribute_item` and `///` comments. Attached by widening the
            span backwards, so `#[test] fn it_works` is one chunk.
    """

    containers: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    standalone: tuple[str, ...] = ()
    separator: str = "."
    body_types: tuple[str, ...] = ()
    wrappers: tuple[str, ...] = ()
    preamble: tuple[str, ...] = ()
    type_name_field: str = "name"


def _unwrap(node: Node, policy: NestedPolicy) -> Node:
    """What a wrapper wraps, or the node itself.

    The caller keeps both: the wrapper is the chunk's extent, the inner
    node is what it is called and what type it has.
    """
    if node.type not in policy.wrappers:
        return node
    for child in node.named_children:
        if child.type in policy.types + policy.members + policy.standalone:
            return child
    return node


def _preamble_start(siblings: list[Node], *, index: int, start: int, policy: NestedPolicy) -> int:
    """Widen a span backwards over the attributes and doc comments above it.

    Contiguity is the rule: a blank line ends the chain. Without it a
    file-level comment three lines up would be swallowed by the first
    definition below it.
    """
    for previous in reversed(siblings[:index]):
        if previous.type not in policy.preamble:
            break
        previous_start, previous_end = line_span(previous)
        if previous_end < start - 1:
            break
        start = previous_start
    return start


def _def_span_in(
    siblings: list[Node], *, index: int, policy: NestedPolicy, covered: list[bool], symbol: str
) -> Span:
    """One definition's span, wrapper and preamble included."""
    node = siblings[index]
    start, end = line_span(node)
    start = _preamble_start(siblings, index=index, start=start, policy=policy)
    mark_covered(covered, start=start, end=end)
    return Span(start_line=start, end_line=end, symbol=symbol, node_type=_unwrap(node, policy).type)


def _type_name(node: Node, policy: NestedPolicy) -> str | None:
    """A type's name, from whichever field this language keeps it in."""
    if policy.type_name_field == "name":
        return symbol_name(node)
    field = node.child_by_field_name(policy.type_name_field)
    return field.text.decode() if field is not None and field.text is not None else None


def _body(node: Node, policy: NestedPolicy) -> Node | None:
    """The node holding a container's or type's declarations, if any."""
    found = node.child_by_field_name("body")
    if found is not None:
        return found
    for child in node.named_children:
        if child.type in policy.body_types:
            return child
    return None


def type_spans(
    node: Node, policy: NestedPolicy, lines: list[str], covered: list[bool]
) -> list[Span]:
    """Members as their own chunks, the rest of the type as one more.

    Public because a language with an otherwise unusual policy still has
    ordinary classes: typescript composes this rather than writing the
    member walk and its gap pass a second time.

    Args:
        node: The type node (a class, an impl block, an object).
        policy: What this language calls members, and how it separates a
            type from one.
        lines: The file's lines, for the gap pass.
        covered: Shared line bookkeeping.

    Returns:
        One span per member, plus one for whatever type lines are left.
    """
    inner = _unwrap(node, policy)
    name = _type_name(inner, policy)
    body = _body(inner, policy)
    if name is None or body is None:
        return []  # error-recovery leftovers fall through to gap chunks
    start, end = line_span(node)
    found: list[Span] = []
    members = body.named_children
    for index, member in enumerate(members):
        if _unwrap(member, policy).type not in policy.members:
            continue
        member_name = symbol_name(_unwrap(member, policy))
        if member_name is not None:
            found.append(
                _def_span_in(
                    members,
                    index=index,
                    policy=policy,
                    covered=covered,
                    symbol=f"{name}{policy.separator}{member_name}",
                )
            )
    found += gap_spans(
        lines, covered=covered, start=start, end=end, symbol=name, node_type=inner.type
    )
    return found


def _walk(
    nodes: list[Node], policy: NestedPolicy, lines: list[str], covered: list[bool], depth: int
) -> list[Span]:
    """One level of the tree: containers descend, types split, the rest stand."""
    if depth > _MAX_NESTING:
        return []
    found: list[Span] = []
    for index, child in enumerate(nodes):
        inner = _unwrap(child, policy)
        if inner.type in policy.containers:
            body = _body(inner, policy)
            if body is not None:
                found += _walk(body.named_children, policy, lines, covered, depth + 1)
        elif inner.type in policy.types:
            found += type_spans(child, policy, lines, covered)
        elif inner.type in policy.standalone:
            name = symbol_name(inner)
            if name is not None:
                found.append(
                    _def_span_in(nodes, index=index, policy=policy, covered=covered, symbol=name)
                )
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
