"""Single-file components: one file, several languages inside it.

A `.vue` or `.svelte` file is not written in one language — it is a
`<template>`, a `<script>` and a `<style>`, each its own. Chunking it as
one thing would give hits that are three languages at once.

So a container language declares a `sections` splitter instead of a
`spans` extractor: it says where each block starts and what language it
is, and `chunk_file` chunks each with that language's own machinery. A
`<script lang="ts">` gets the real TypeScript extractor for free.

The tags themselves belong to no section, so the gap pass sweeps them up
— without it `<script>` and `</script>` would fall out of the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tree_sitter import Node

# `lang="..."` on a `<script>` tag, mapped onto the language names the
# registry knows. Absent means JavaScript, which is what both Vue and
# Svelte default to.
_SCRIPT_LANGS = {
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "tsx",
    "js": "javascript",
    "javascript": "javascript",
    "jsx": "javascript",
}

_STYLE_LANG = "css"
"""What a `<style>` block is chunked as. No grammar ships for it, so the
text chunker windows it — still searchable, just not by rule."""


@dataclass(frozen=True, kw_only=True)
class Section:
    """One part of a container file, to be chunked as its own language.

    Attributes:
        lang: Registry name of the language this part is written in. A
            language with no grammar installed simply falls back to text
            windows, exactly as it would in a file of its own.
        start_line: 1-based line in the *container* where `text` begins.
            The chunker adds this back after chunking, so a hit points at
            the real line of the real file.
        text: The part's source, with no surrounding tags.
    """

    lang: str
    start_line: int
    text: str


class SectionSplitter(Protocol):
    """How a container language reports its parts.

    The counterpart to `SpanExtractor`: that one says which lines of a
    file deserve a chunk, this one says which languages a file is written
    in and where each starts.
    """

    def __call__(self, root: Node, lines: list[str]) -> list[Section]:
        """Split one parsed container file.

        Args:
            root: The container parsed with its own grammar.
            lines: The container's lines, for splitters that need the
                text rather than the tree.

        Returns:
            The sections, in file order.
        """
        ...


def _attribute(tag: Node, name: str) -> str | None:
    """Value of one attribute on a start tag, unquoted; None if absent."""
    for attribute in tag.named_children:
        if attribute.type != "attribute":
            continue
        key = next((c for c in attribute.named_children if c.type == "attribute_name"), None)
        if key is None or key.text is None or key.text.decode() != name:
            continue
        for child in attribute.named_children:
            if child.type in ("quoted_attribute_value", "attribute_value"):
                # `quoted_attribute_value` wraps the value in a child;
                # `attribute_value` is the bare form.
                inner = child.named_children[0] if child.named_children else child
                return inner.text.decode() if inner.text is not None else None
    return None


def _start_tag(node: Node) -> Node | None:
    return next((c for c in node.named_children if c.type == "start_tag"), None)


def _raw_section(node: Node, lang: str) -> Section | None:
    """A `<script>`/`<style>` block's contents as a Section.

    The grammar hands the body back as a single `raw_text` node, which is
    what is wanted — unparsed, with its own position — but that node
    begins immediately after the `>` of the opening tag, *mid-line*. Its
    first "line" is therefore the tail of `<style scoped>`, and a section
    handed over that way would claim the tag line as its own: the chunk's
    text would not match the file's lines at that range.

    So the leading newline is dropped and the start line moved past the
    tag. A block written entirely on the tag line (`<style>x</style>`)
    has no line of its own to give and is left to the gap pass.
    """
    raw = next((c for c in node.named_children if c.type == "raw_text"), None)
    if raw is None or raw.text is None:
        return None  # an empty block: nothing to chunk
    text = raw.text.decode()
    start_line = raw.start_point[0] + 1
    if raw.start_point[1] > 0:
        newline = text.find("\n")
        if newline == -1:
            return None
        text = text[newline + 1 :]
        start_line += 1
    return Section(lang=lang, start_line=start_line, text=text)


def _script_section(node: Node) -> Section | None:
    """A `<script>` block, in whatever language its `lang=` attribute names."""
    tag = _start_tag(node)
    declared = _attribute(tag, "lang") if tag is not None else None
    # Unknown values fall back to JavaScript rather than being dropped:
    # a dialect we do not know is still closer to JS than to nothing.
    return _raw_section(node, _SCRIPT_LANGS.get((declared or "js").lower(), "javascript"))


def sections(root: Node, lines: list[str]) -> list[Section]:
    """Split a single-file component into its parts.

    Vue and Svelte share one splitter because the difference between them
    is a convention, not a structure: Vue wraps its markup in an explicit
    `<template>`, Svelte leaves it bare at the top level. Both are the
    same parse — elements, a script element, a style element — so the
    walk keeps whichever it finds.

    Args:
        root: The container parsed as HTML.
        lines: The container's lines; unused, every offset comes from the
            tree.

    Returns:
        The sections, in file order. Markup is reported as `html` so it
        goes through the HTML extractor; script and style go to whatever
        language they declare.
    """
    found: list[Section] = []
    for child in root.named_children:
        if child.type == "script_element":
            section = _script_section(child)
        elif child.type == "style_element":
            section = _raw_section(child, _STYLE_LANG)
        elif child.type == "element":
            # Vue's `<template>` and Svelte's bare markup alike. Kept as
            # HTML text rather than descended into: the container's own
            # parse already covers it, and re-chunking it here would
            # claim the same lines twice.
            if child.text is None:
                continue
            section = Section(
                lang="html", start_line=child.start_point[0] + 1, text=child.text.decode()
            )
        else:
            continue
        if section is not None:
            found.append(section)
    return found
