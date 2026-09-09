"""Go support for wsindex, written against the published plugin surface.

Everything this package imports comes from two places:

- `wsindex.ingest` — the specification: `LanguageSpec`, `GrammarSpec`.
- `wsindex.ingest.ast` — the toolkit an extractor is written with:
  `line_span` for a node's line range, `mark_covered` to claim those
  lines, `symbol_name` to read a node's name, `Span` for the result.
  (`def_span` bundles the first three; Go needs them apart because of
  the doc-comment look-behind below.)

Nothing private, and no edit to wsindex itself. That is the claim the
plugin system makes, and this package is where it gets tested.

What Go gets chunked into
-------------------------
One chunk per function, per method, and per named type, each including
the `//` doc comment written above it — that sentence is usually the most
searchable thing about a declaration, and filing it separately would be a
waste. Everything else — the package clause, imports, `var`/`const`
blocks — falls through to the gap pass and becomes chunks of its own, so
every non-blank line is indexed exactly once.

Methods carry a qualified symbol (`Server.Serve`), the same shape the
built-in Python extractor gives class methods, so `--symbol Server`
finds a type's whole method set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest import GrammarSpec, LanguageSpec
from wsindex.ingest.ast import Span, line_span, mark_covered, symbol_name
from wsindex.model import Kind

if TYPE_CHECKING:
    from tree_sitter import Node

__all__ = ["GO", "LANGUAGES", "go_spans"]


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


def go_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Claim Go's top-level definitions; everything else becomes gaps.

    Signature fixed by `wsindex.ingest.SpanExtractor`. `lines` goes
    unused here: the doc-comment look-behind works off sibling nodes and
    their line numbers, so the text itself is never needed. The parameter
    stays because the contract passes it to every extractor.

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


GO = LanguageSpec(
    name="go",
    kind=Kind.CODE,
    suffixes=(".go",),
    # Named indirectly rather than imported: if `tree-sitter-go` were
    # somehow absent, wsindex falls back to text windows for `.go` files
    # instead of dropping the language — degraded, but still indexed.
    grammar=GrammarSpec(module="tree_sitter_go", getter="language"),
    spans=go_spans,
)

LANGUAGES = (GO,)
"""What the entry point resolves to. A tuple rather than the bare spec
because a plugin may grow — Go templates would be a second entry here,
not a second entry point."""
