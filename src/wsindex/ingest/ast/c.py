"""C policy: functions, structs, enums and unions; typedefs stay gaps.

C does not put a name field on a function. It nests *declarators*:
`int *f(void)` parses as a `function_definition` whose declarator is a
`pointer_declarator` whose declarator is the `function_declarator` whose
declarator is finally the identifier. `declarator_name` walks that chain,
and C++ walks the same one — which is why it lives here and `cpp` imports
it rather than growing a second copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import Span, def_span, symbol_name

if TYPE_CHECKING:
    from tree_sitter import Node

# Nodes the declarator chain can bottom out in. `qualified_identifier` and
# the two special member names only occur in C++, but the walk is shared
# and a name it cannot reach is worse than a name it does not expect.
_NAME_NODES = (
    "identifier",
    "type_identifier",
    "field_identifier",
    "qualified_identifier",
    "operator_name",
    "destructor_name",
)

# Type definitions that carry a plain `name` field.
_TAGGED_TYPES = ("struct_specifier", "union_specifier", "enum_specifier")

_MAX_DECLARATOR_DEPTH = 16
"""Guard for the declarator walk. `int *(*(*f)(void))[3]` is legal and
nests deeply; a damaged tree could in principle cycle. Bounding the walk
costs nothing and cannot be the reason a file fails to index."""


def declarator_name(node: Node) -> str | None:
    """Follow a declarator chain down to the name it eventually declares.

    Args:
        node: A `function_definition`, `declaration` or any node whose
            `declarator` field leads to an identifier.

    Returns:
        The declared name — `"serve"`, or `"Server::serve"` for a C++
        out-of-line definition — or None when the chain leads nowhere,
        which is what a partial tree from error recovery looks like.
    """
    current: Node | None = node
    for _ in range(_MAX_DECLARATOR_DEPTH):
        if current is None:
            return None
        if current.type in _NAME_NODES:
            return current.text.decode() if current.text is not None else None
        current = current.child_by_field_name("declarator")
    return None


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Functions and tagged types; `#include`s and globals become gaps.

    A `typedef struct {...} Name;` is deliberately left to the gap pass:
    its span is the same lines as the struct it wraps, and claiming both
    would either double-count or make the symbol depend on which node was
    visited first.

    Args:
        root: Root of the parsed file; may be partial.
        lines: The file's lines; unused, C needs no look-behind.
        covered: Shared line-coverage bookkeeping.

    Returns:
        One span per function definition and per named struct/union/enum.
    """
    found: list[Span] = []
    for child in root.named_children:
        if child.type == "function_definition":
            name = declarator_name(child)
            if name is not None:
                found.append(def_span(child, covered=covered, symbol=name, node_type=child.type))
        elif child.type in _TAGGED_TYPES:
            name = symbol_name(child)
            if name is not None:
                found.append(def_span(child, covered=covered, symbol=name, node_type=child.type))
    return found
