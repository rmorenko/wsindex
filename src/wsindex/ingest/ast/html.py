"""HTML policy: each top-level element is a chunk, named by its tag.

Also what Angular component templates get. `foo.component.html` is a
plain HTML file, so it needs no dialect of its own — the elements a
template is built from (`<app-child>`, `<ng-template>`, a wrapper `<div>`)
are exactly what someone searching a template is looking for.

Deviation from the plan worth naming: it proposed taking the symbol from
the paired `foo.component.ts`. An extractor is handed a parse tree and
nothing else — no path, no repository — so cross-file naming cannot
happen here. The tag name is the honest local answer, and the file path
already carries the component in every hit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import Span, def_span

if TYPE_CHECKING:
    from tree_sitter import Node


def _tag_name(node: Node) -> str | None:
    """The element's tag, read from its start tag; None on a partial node."""
    tag = next((c for c in node.named_children if c.type == "start_tag"), None)
    if tag is None:
        return None
    name = next((c for c in tag.named_children if c.type == "tag_name"), None)
    return name.text.decode() if name is not None and name.text is not None else None


def spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Top-level elements, named by tag; text and comments become gaps.

    Top level only, deliberately. Descending would put every nested
    `<div>` in a chunk of its own and bury the markup that means
    something under the markup that does not.

    Args:
        root: Root of the parsed file; may be partial.
        lines: The file's lines; unused, HTML needs no look-behind.
        covered: Shared line-coverage bookkeeping.

    Returns:
        One span per named top-level element.
    """
    found: list[Span] = []
    for child in root.named_children:
        if child.type not in ("element", "script_element", "style_element"):
            continue
        name = _tag_name(child)
        if name is not None:
            found.append(def_span(child, covered=covered, symbol=name, node_type=child.type))
    return found
