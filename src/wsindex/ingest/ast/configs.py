"""Config policies: TOML tables, YAML/JSON top-level keys, Dockerfile stages, XML elements."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

from wsindex.ingest.ast.core import Span, line_span, mark_covered
from wsindex.ingest.text_chunker import WINDOW_LINES

if TYPE_CHECKING:
    from tree_sitter import Node

XML_MAX_DEPTH = 6
"""How far into the tree the walk descends. Only reached by a document
nested six levels deep whose every level is over a window long; past that
the structure has stopped telling a reader anything a line range does not."""


def _config_span(
    node: Node, lines: list[str], covered: list[bool], *, symbol: str | None, node_type: str
) -> Span:
    """Span of one config unit.

    Container nodes (a TOML table, a YAML mapping) often swallow trailing
    blank separator lines — trim them so chunks end on content.
    """
    start, end = line_span(node)
    while end > start and not lines[end - 1].strip():
        end -= 1
    mark_covered(covered, start=start, end=end)
    return Span(start_line=start, end_line=end, symbol=symbol, node_type=node_type)


def toml_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Top-level tables and arrays of tables; leading bare pairs go to gaps."""
    spans: list[Span] = []
    for child in root.named_children:
        if child.type not in ("table", "table_array_element"):
            continue
        symbol: str | None = None
        key = next(
            (c for c in child.named_children if c.type in ("bare_key", "dotted_key", "quoted_key")),
            None,
        )
        if key is not None and key.text is not None:
            symbol = key.text.decode()
        spans.append(_config_span(child, lines, covered, symbol=symbol, node_type=child.type))
    return spans


def _descend(nodes: Iterable[Node], *path: str) -> Iterator[Node]:
    """Children reached by following a chain of node types.

    `_descend(root.named_children, "block_node", "block_mapping", "pair")`
    reads the way the tree is shaped. Written out as four nested loops it
    was the deepest code in the package, and none of that depth was about
    yaml — only about walking.
    """
    if not path:
        yield from nodes
        return
    wanted, rest = path[0], path[1:]
    for node in nodes:
        for child in node.named_children:
            if child.type == wanted:
                yield from _descend([child], *rest) if rest else iter([child])


def yaml_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Top-level mapping pairs of every document in the stream."""
    spans: list[Span] = []
    for pair in _descend(root.named_children, "block_node", "block_mapping", "block_mapping_pair"):
        key = pair.child_by_field_name("key")
        symbol = key.text.decode() if key is not None and key.text is not None else None
        spans.append(_config_span(pair, lines, covered, symbol=symbol, node_type=pair.type))
    return spans


def _json_key(pair: Node) -> str | None:
    """The key of a json pair, without its quotes."""
    key = pair.child_by_field_name("key")
    if key is None:
        return None
    content = next((c for c in key.named_children if c.type == "string_content"), None)
    return content.text.decode() if content is not None and content.text is not None else None


def json_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Top-level object pairs; a top-level array yields no spans (gaps take over)."""
    return [
        _config_span(pair, lines, covered, symbol=_json_key(pair), node_type=pair.type)
        for pair in _descend([root], "object", "pair")
    ]


def _stage_symbol(node: Node) -> str | None:
    """Stage name of a FROM instruction: the AS alias, else the image spec."""
    for wanted in ("image_alias", "image_spec"):
        child = next((c for c in node.named_children if c.type == wanted), None)
        if child is not None and child.text is not None:
            return child.text.decode()
    return None


def dockerfile_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """One span per build stage: FROM up to the last instruction before the next FROM.

    `node_type` is the synthetic "stage" — unlike other extractors the unit
    here is a group of nodes, not a single grammar node. Instructions before
    the first FROM (global ARGs) fall through to gap chunks.
    """
    spans: list[Span] = []
    start: int | None = None
    end = 0
    symbol: str | None = None

    def close() -> None:
        if start is not None:
            mark_covered(covered, start=start, end=end)
            spans.append(Span(start_line=start, end_line=end, symbol=symbol, node_type="stage"))

    for node in root.named_children:
        if not node.type.endswith("_instruction"):
            continue
        node_start, node_end = line_span(node)
        if node.type == "from_instruction":
            close()
            start, symbol = node_start, _stage_symbol(node)
        if start is not None:
            end = node_end
    close()
    return spans


def _xml_elements(node: Node) -> list[Node]:
    """Direct element children, reached through the `content` node.

    tree-sitter-xml puts an element's children inside a `content` node
    rather than under the element itself, so "the children of `<project>`"
    is one hop longer than it looks. A self-closing element has no
    `content` at all.
    """
    found: list[Node] = []
    for child in node.named_children:
        if child.type == "element":
            found.append(child)
        elif child.type == "content":
            found.extend(sub for sub in child.named_children if sub.type == "element")
    return found


def _xml_name(element: Node) -> str | None:
    """The tag name, from the opening tag or the self-closing one.

    Every element this grammar produces opens with `STag` or
    `EmptyElemTag`, error trees included — six malformed samples in
    six malformed samples all kept it, including an orphan closing tag
    and a `< >`. So the None below guards against a future grammar rather than
    a tree anyone has seen.
    """
    tag = element.named_children[0] if element.named_children else None
    if tag is None or tag.type not in ("STag", "EmptyElemTag"):
        return None  # pragma: no cover - no such tree from tree-sitter-xml
    name = next((sub for sub in tag.named_children if sub.type == "Name"), None)
    return name.text.decode() if name is not None and name.text is not None else None


def _xml_group(group: list[Node], path: list[str], covered: list[bool]) -> Span | None:
    """One span covering a run of sibling elements."""
    if not group:
        return None
    start, _ = line_span(group[0])
    # No trailing-blank trim, unlike `_config_span`: a run ends at an
    # element's closing tag, which is content by definition.
    _, end = line_span(group[-1])
    mark_covered(covered, start=start, end=end)
    # A run keeps its parent's path; a single element adds its own name,
    # which is what makes `--symbol project/dependencies` worth typing.
    symbol = "/".join([*path, _xml_name(group[0]) or "?"]) if len(group) == 1 else "/".join(path)
    return Span(
        start_line=start,
        end_line=end,
        symbol=symbol or None,
        node_type="element" if len(group) == 1 else "elements",
    )


def _xml_walk(element: Node, path: list[str], covered: list[bool], *, depth: int) -> list[Span]:
    """Chunk one element's children, descending only into the big ones."""
    spans: list[Span] = []
    group: list[Node] = []

    def flush() -> None:
        span = _xml_group(group, path, covered)
        if span is not None:
            spans.append(span)
        group.clear()

    for child in _xml_elements(element):
        start, end = line_span(child)
        size = end - start + 1
        if size > WINDOW_LINES and _xml_elements(child) and depth < XML_MAX_DEPTH:
            flush()
            inner = _xml_walk(child, [*path, _xml_name(child) or "?"], covered, depth=depth + 1)
            if inner:
                # The container's own tags go to the runs at either end
                # rather than becoming chunks. Left alone they are a
                # two-line `<dependencies>` and a two-line
                # `</dependencies>` — and searching a real pom for "where
                # is the dependency version configured" returned exactly
                # those two, above every dependency in the file. Short
                # chunks made of nothing but the container's name are the
                # strongest lexical match and the weakest answer.
                first, last = inner[0].start_line, inner[-1].end_line
                inner[0] = replace(inner[0], start_line=start)
                inner[-1] = replace(inner[-1], end_line=end)
                # Only the tag lines just absorbed. Marking the whole
                # container instead would claim the comments *between*
                # its groups, which no span covers — the gap pass would
                # then skip them and they would belong to nobody.
                mark_covered(covered, start=start, end=first - 1)
                mark_covered(covered, start=last + 1, end=end)
            spans.extend(inner)
            continue
        if group and end - line_span(group[0])[0] + 1 > WINDOW_LINES:
            flush()
        group.append(child)
    flush()
    return spans


def xml_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    """Runs of sibling elements, windowed, with the element path as symbol.

    Not "one chunk per top-level element" like the extractors above: a
    TOML file has many top-level tables, an XML file has exactly one root.
    Measured on Maven's own `pom.xml`, `<project>` is 1287 of 1306 lines,
    so that rule gives one chunk per file.

    One chunk per *child* of the root is wrong at both ends, and both ends
    were measured: Tomcat's `web.xml` has 1029 children of four lines
    each, and Maven's pom has a `<dependencyManagement>` of 514 — most of
    it past what the embedding model reads.

    So the unit is a run of siblings up to `WINDOW_LINES`, the same window
    the text chunker uses, and any element too big for that is descended
    into. Result: 47 chunks for Maven's pom, 109 for Tomcat's web.xml, 1
    for an Android manifest.

    Whatever is left falls to the gap pass, as everywhere else.
    """
    spans: list[Span] = []
    for top in _xml_elements(root):
        spans.extend(_xml_walk(top, [_xml_name(top) or "?"], covered, depth=0))
    return spans
