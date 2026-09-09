"""Lua support for wsindex, written against the published plugin surface.

Everything this package imports comes from two places:

- `wsindex.ingest` — the specification: `LanguageSpec`, `GrammarSpec`.
- `wsindex.ingest.ast` — the toolkit an extractor is written with:
  `line_span` for a node's line range, `mark_covered` to claim those
  lines, `symbol_name` to read a node's name, `Span` for the result.
  (`def_span` bundles the first three; Lua needs them apart because of
  the doc-comment look-behind below.)

Nothing private, and no edit to wsindex itself. That is the claim the
plugin system makes, and this package is where it gets tested.

Lua on purpose: it is deliberately *not* one of the languages wsindex
ships with, so this example cannot collide with a built-in. An earlier
version of it did Go — and then Go moved into the box, the two claimed
`.go`, and the loader started skipping the plugin with a conflict
warning. Exactly the behaviour it should have, and a bad look on an
example.

What Lua gets chunked into
--------------------------
One chunk per function, however it is spelled — `function f()`,
`local function f()`, `function M.f()`, `function M:f()`, and
`M.f = function()`. Each includes the `--` comment block written above
it, which is where a Lua module keeps its documentation. Everything else
falls through to the gap pass, so every non-blank line is indexed exactly
once.

Table-qualified names arrive already qualified: the grammar reports
`M.greet` and `M:method` as the function's name, so `--symbol M` finds a
module's whole surface without the extractor assembling anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest import GrammarSpec, LanguageSpec
from wsindex.ingest.ast import Span, line_span, mark_covered, symbol_name
from wsindex.model import Kind

if TYPE_CHECKING:
    from tree_sitter import Node

__all__ = ["LANGUAGES", "LUA", "lua_spans"]


def _extend_back(siblings: list[Node], *, index: int, start_line: int) -> int:
    """Pull a definition's comment block into its span.

    Lua documentation is `--` or `---` lines directly above the function,
    reported by the grammar as sibling `comment` nodes rather than as
    children. Left alone, `--- Greets someone.` would become a chunk of
    its own — the one sentence explaining a function, filed away from it.

    A blank line ends the chain: a comment separated from the code below
    is a remark about the file, not documentation of the next function.

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
            break  # a blank line: a remark, not documentation
        start_line = previous_start
    return start_line


def _documented_span(
    siblings: list[Node], *, index: int, covered: list[bool], symbol: str, node_type: str
) -> Span:
    """Span of one definition, extended back over its comment block.

    `def_span` covers the common case; this is the hand-rolled version
    for when a language needs a look-behind, built from the same
    published pieces (`line_span`, `mark_covered`, `Span`).
    """
    node = siblings[index]
    start, end = line_span(node)
    start = _extend_back(siblings, index=index, start_line=start)
    mark_covered(covered, start=start, end=end)
    return Span(start_line=start, end_line=end, symbol=symbol, node_type=node_type)


def _assigned_function_name(node: Node) -> str | None:
    """Name of a function bound by assignment, or None for a plain value.

    `M.render = function() ... end` is a function definition wearing an
    assignment's clothes; `M.timeout = 30` is not, and claiming it would
    make a constant look like code worth reading on its own.

    Args:
        node: An `assignment_statement`.

    Returns:
        The assigned name when the value is a function, else None.
    """
    values = node.child_by_field_name("value")
    if values is None or values.type != "function_definition":
        return None
    return symbol_name(node)


def lua_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Claim Lua's functions; everything else becomes gaps.

    Signature fixed by `wsindex.ingest.SpanExtractor`. `lines` goes
    unused: the comment look-behind works off sibling nodes and their
    line numbers, so the text itself is never needed. The parameter stays
    because the contract passes it to every extractor.

    Args:
        root: Root of the parsed file; may describe a partial tree, since
            the parser is error tolerant.
        lines: The file's lines, unused by this language.
        covered: Shared line-coverage bookkeeping.

    Returns:
        One span per function, comment block included.
    """
    spans: list[Span] = []
    children = root.named_children
    for index, child in enumerate(children):
        if child.type == "function_declaration":
            name = symbol_name(child)
        elif child.type == "assignment_statement":
            name = _assigned_function_name(child)
        else:
            continue
        if name is not None:
            spans.append(
                _documented_span(
                    children,
                    index=index,
                    covered=covered,
                    symbol=name,
                    node_type=child.type,
                )
            )
    return spans


LUA = LanguageSpec(
    name="lua",
    kind=Kind.CODE,
    suffixes=(".lua",),
    # Named indirectly rather than imported: if `tree-sitter-lua` were
    # somehow absent, wsindex falls back to text windows for `.lua` files
    # instead of dropping the language — degraded, but still indexed.
    grammar=GrammarSpec(module="tree_sitter_lua", getter="language"),
    spans=lua_spans,
)

LANGUAGES = (LUA,)
"""What the entry point resolves to. A tuple rather than the bare spec
because a plugin may grow — Luau or a template dialect would be a second
entry here, not a second entry point."""
