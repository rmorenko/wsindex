"""Go policy: doc comments stick to declarations, receivers qualify methods.

Arrived as the example plugin and moved into the box unchanged — which
is the useful part of the story: a language written entirely against the
published plugin surface needed no rework to become a built-in. The seam
holds in both directions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import Span, line_span, mark_covered, symbol_name

if TYPE_CHECKING:
    from tree_sitter import Node


def _extend_back(siblings: list[Node], *, index: int, start_line: int) -> int:
    """Pull a definition's doc comment into its span.

    Go writes documentation as `//` lines directly above the thing they
    document, and the grammar reports each as a sibling `comment` rather
    than a child. Left alone, `// Serve writes the request path...` would
    become a chunk of its own — the single most searchable sentence about
    a method, filed apart from the method.

    A blank line ends the chain, which is exactly Go's own rule for what
    counts as a doc comment: `godoc` ignores a comment separated from the
    declaration below it.

    Args:
        siblings: The parent's named children.
        index: Position of the definition within them.
        start_line: The definition's own first line.

    Returns:
        The first line the span should start at.
    """
    for previous in reversed(siblings[:index]):
        if previous.type != "comment":
            break
        previous_start, previous_end = line_span(previous)
        if previous_end < start_line - 1:
            break  # a blank line: a detached comment, not documentation
        start_line = previous_start
    return start_line


def _documented_span(siblings: list[Node], *, index: int, covered: list[bool], symbol: str) -> Span:
    """Span of one definition, extended back over its doc comment.

    `def_span` covers the common case; this is the hand-rolled version
    for when a language needs a look-behind, built from the same
    published pieces (`line_span`, `mark_covered`, `Span`).
    """
    node = siblings[index]
    start, end = line_span(node)
    start = _extend_back(siblings, index=index, start_line=start)
    mark_covered(covered, start=start, end=end)
    return Span(start_line=start, end_line=end, symbol=symbol, node_type=node.type)


def _receiver_type(node: Node) -> str | None:
    """Name of the type a method hangs off, pointer receivers included.

    Go writes the receiver as a one-parameter list: `(s *Server)` or
    `(v Value)`. The parameter's `type` field is a `pointer_type` in the
    first case and a `type_identifier` in the second, so the pointer has
    to be peeled to get a symbol a reader would recognize — `Server`, not
    `*Server`.

    Args:
        node: A `method_declaration`.

    Returns:
        The receiver type name, or None if the tree is too damaged to
        tell (error recovery leaves partial nodes).
    """
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    for parameter in receiver.named_children:
        type_node = parameter.child_by_field_name("type")
        if type_node is None:
            continue
        if type_node.type == "pointer_type":
            # `*Server` — the identifier is the pointer's only named child.
            inner = type_node.named_children
            if not inner or inner[0].text is None:
                return None
            return inner[0].text.decode()
        if type_node.text is None:
            return None
        return type_node.text.decode()
    return None


def _type_spans(siblings: list[Node], index: int, covered: list[bool]) -> list[Span]:
    """Spans for a `type` declaration: one per named type it introduces.

    A single declaration spans the whole statement, keyword included. A
    grouped one — `type ( A struct{...}; B int )` — introduces several
    types in one node, so each inner spec gets its own chunk and the
    `type (` and `)` lines fall through to the gap pass.

    Args:
        siblings: The file's named children, for the doc-comment look-behind.
        index: Position of the `type_declaration` within them.
        covered: Shared line-coverage bookkeeping.

    Returns:
        One span per named type; empty when nothing could be named.
    """
    node = siblings[index]
    specs = [child for child in node.named_children if child.type in ("type_spec", "type_alias")]
    if not specs:
        return []  # error recovery: let the gap pass take the lines
    if len(specs) == 1:
        name = symbol_name(specs[0])
        if name is None:
            return []
        return [_documented_span(siblings, index=index, covered=covered, symbol=name)]
    # Grouped `type (...)`: the doc comment above belongs to the block, not
    # to any one type, so each spec is spanned on its own lines.
    spans: list[Span] = []
    for spec in specs:
        name = symbol_name(spec)
        if name is not None:
            start, end = line_span(spec)
            mark_covered(covered, start=start, end=end)
            spans.append(Span(start_line=start, end_line=end, symbol=name, node_type=spec.type))
    return spans


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Claim Go's top-level definitions; everything else becomes gaps.

    `lines` goes unused: the doc-comment look-behind works off sibling
    nodes and their line numbers, so the text itself is never needed. The
    parameter stays because `SpanExtractor` passes it to every language.

    Args:
        root: Root of the parsed file; may be a partial tree, since the
            parser is error tolerant.
        lines: The file's lines, unused by this language.
        covered: Shared line-coverage bookkeeping; `_documented_span`
            marks it through `mark_covered`.

    Returns:
        Spans for functions, methods and named types.
    """
    spans: list[Span] = []
    children = root.named_children
    for index, child in enumerate(children):
        if child.type == "function_declaration":
            name = symbol_name(child)
            if name is not None:
                spans.append(_documented_span(children, index=index, covered=covered, symbol=name))
        elif child.type == "method_declaration":
            name = symbol_name(child)
            receiver = _receiver_type(child)
            if name is not None:
                symbol = f"{receiver}.{name}" if receiver is not None else name
                spans.append(
                    _documented_span(children, index=index, covered=covered, symbol=symbol)
                )
        elif child.type == "type_declaration":
            spans += _type_spans(children, index, covered)
    return spans
