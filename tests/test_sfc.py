"""HTML, and the single-file-component mechanism.

The new machinery of the stage is not a grammar — it is that one file may
be several languages. A `.vue` or `.svelte` component holds a template, a
script and a style block, and each is chunked by the language it is
actually written in.

What matters most here is what the mechanism could quietly get wrong:
line numbers that stay relative to a section, and the tag lines between
sections falling out of the index entirely. Both have tests of their own.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from conftest import needs_grammar
from wsindex.ingest import REGISTRY, chunk_file
from wsindex.model import Chunk, Kind, SourceFile

if TYPE_CHECKING:
    from tree_sitter import Node

VUE = dedent("""\
    <template>
      <div class="card">
        <h1>{{ title }}</h1>
      </div>
    </template>

    <script setup lang="ts">
    import { ref } from "vue"

    export function useTitle(): string {
      return "hi"
    }
    </script>

    <style scoped>
    .card { color: red; }
    </style>
""")

SVELTE = dedent("""\
    <script>
      export let name = "world";

      function greet() {
        return `hi ${name}`;
      }
    </script>

    <h1>Hello {name}</h1>

    <style>
      h1 { color: blue; }
    </style>
""")

ANGULAR = dedent("""\
    <div class="wrap">
      <h1>{{ title }}</h1>
      <app-child [value]="v" (out)="onOut($event)"></app-child>
    </div>

    <ng-template #tpl>
      <span>x</span>
    </ng-template>
""")


def has(lang: str) -> bool:
    return REGISTRY.parser(lang) is not None


def chunks(text: str, *, lang: str, path: str) -> list[Chunk]:
    return chunk_file(text, SourceFile(repo="r", path=path, lang=lang, kind=Kind.CODE))


# --- HTML, which is also what Angular templates are ----------------------


@needs_grammar("html")
def test_top_level_elements_are_named_by_their_tag() -> None:
    got = [(c.symbol, c.node_type) for c in chunks(ANGULAR, lang="html", path="a.component.html")]
    assert got == [
        ("div", "element"),
        ("ng-template", "element"),
    ]


@needs_grammar("html")
def test_nested_elements_stay_inside_their_parent() -> None:
    # Descending would put every `<div>` in a chunk of its own and bury
    # the markup that means something under the markup that does not.
    wrapper = chunks(ANGULAR, lang="html", path="a.component.html")[0]
    assert "app-child" in wrapper.text


@needs_grammar("html")
def test_angular_component_templates_need_no_dialect() -> None:
    # `foo.component.html` is a plain HTML file; the walker matches it on
    # `.html` and nothing else is required.
    from pathlib import Path

    matched = REGISTRY.match(Path("src/app/foo.component.html"))
    assert matched is not None
    assert matched.name == "html"


# --- the container mechanism ---------------------------------------------


@needs_grammar("vue")
def test_a_vue_script_block_is_chunked_as_typescript() -> None:
    # The whole point: the section is handed to the real TypeScript
    # extractor, so a function inside a `.vue` file is found the same way
    # a function in a `.ts` file is.
    by_symbol = {c.symbol: c for c in chunks(VUE, lang="vue", path="Card.vue")}
    assert "useTitle" in by_symbol
    assert by_symbol["useTitle"].node_type == "function_declaration"


@needs_grammar("vue")
def test_section_chunks_carry_container_line_numbers() -> None:
    # A section is chunked as if it were a file, so its chunks start at
    # line 1. Without adding the offset back, every hit in a component
    # would point at the top of the file.
    by_symbol = {c.symbol: c for c in chunks(VUE, lang="vue", path="Card.vue")}
    found = by_symbol["useTitle"]
    assert found.start_line == 10
    assert VUE.splitlines()[found.start_line - 1].startswith("export function useTitle")


@needs_grammar("vue")
def test_chunks_keep_the_containers_language() -> None:
    # `lang` answers "what kind of file is this" everywhere else in the
    # index, so a `.vue` file whose chunks claimed to be TypeScript would
    # be the one thing `--lang vue` could not find.
    assert {c.lang for c in chunks(VUE, lang="vue", path="Card.vue")} == {"vue"}


@needs_grammar("vue")
def test_the_template_is_its_own_chunk() -> None:
    by_symbol = {c.symbol: c for c in chunks(VUE, lang="vue", path="Card.vue")}
    assert by_symbol["template"].text.startswith("<template>")


@needs_grammar("vue")
def test_tag_lines_between_sections_are_still_indexed() -> None:
    # Regression: sections cover what is *between* the tags, so
    # `<script setup lang="ts">` and `</script>` belong to no section at
    # all. Before the gap pass they fell out of the index silently.
    produced = chunks(VUE, lang="vue", path="Card.vue")
    assert any("<script setup" in c.text for c in produced)
    assert any("</style>" in c.text for c in produced)


@needs_grammar("svelte")
def test_svelte_bare_markup_is_chunked_too() -> None:
    # Svelte leaves its markup at the top level instead of wrapping it in
    # a `<template>`; the same splitter handles both.
    by_symbol = {c.symbol for c in chunks(SVELTE, lang="svelte", path="App.svelte")}
    assert "greet" in by_symbol  # the script block, chunked as JavaScript
    assert "h1" in by_symbol  # the bare markup, chunked as HTML


@needs_grammar("svelte")
def test_a_script_without_a_lang_attribute_is_javascript() -> None:
    by_symbol = {c.symbol: c for c in chunks(SVELTE, lang="svelte", path="App.svelte")}
    assert by_symbol["greet"].node_type == "function_declaration"


@pytest.mark.parametrize(
    ("lang", "source", "path"),
    [("vue", VUE, "Card.vue"), ("svelte", SVELTE, "App.svelte"), ("html", ANGULAR, "a.html")],
)
def test_containers_cover_every_line(lang: str, source: str, path: str) -> None:
    # The invariant, on the machinery most likely to break it.
    if not has(lang):
        pytest.skip(f"no {lang} grammar installed")
    produced = chunks(source, lang=lang, path=path)
    covered = [line for c in produced for line in range(c.start_line, c.end_line + 1)]
    non_blank = {i for i, line in enumerate(source.splitlines(), 1) if line.strip()}
    assert non_blank <= set(covered)
    assert len(covered) == len(set(covered))


@pytest.mark.parametrize(
    ("lang", "source", "path"),
    [("vue", VUE, "Card.vue"), ("svelte", SVELTE, "App.svelte"), ("html", ANGULAR, "a.html")],
)
def test_container_chunk_text_is_a_verbatim_slice(lang: str, source: str, path: str) -> None:
    # The offset arithmetic is the easiest thing here to get wrong by
    # one; this catches it directly.
    if not has(lang):
        pytest.skip(f"no {lang} grammar installed")
    lines = source.splitlines()
    for chunk in chunks(source, lang=lang, path=path):
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


@needs_grammar("vue")
def test_an_empty_component_yields_nothing() -> None:
    assert chunks("", lang="vue", path="Empty.vue") == []


@needs_grammar("vue")
def test_a_component_with_only_a_template_still_indexes() -> None:
    produced = chunks("<template>\n  <div/>\n</template>\n", lang="vue", path="T.vue")
    assert produced
    assert produced[0].symbol == "template"


@needs_grammar("vue")
def test_a_broken_component_does_not_crash() -> None:
    assert isinstance(chunks("<script>\nfunction (\n", lang="vue", path="B.vue"), list)


# --- what the specification now allows -----------------------------------


def test_a_container_spec_may_not_also_have_spans() -> None:
    # A container's parts are chunked by their own languages; an
    # extractor on top would claim the same lines twice.
    from wsindex.ingest.ast.core import Span
    from wsindex.ingest.languages import (
        GrammarSpec,
        LanguageRegistry,
        LanguageSpec,
        Section,
    )

    def noop_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
        return []

    def noop_sections(root: Node, lines: list[str]) -> list[Section]:
        return []

    registry = LanguageRegistry()
    with pytest.raises(ValueError, match="not both"):
        registry.register(
            LanguageSpec(
                name="both",
                kind=Kind.CODE,
                suffixes=(".both",),
                grammar=GrammarSpec(module="tree_sitter_nonexistent", getter="language"),
                spans=noop_spans,
                sections=noop_sections,
            )
        )


def test_a_container_spec_needs_a_grammar() -> None:
    # The splitter is handed a parse tree; without a grammar there is
    # nothing to hand it.
    from wsindex.ingest.languages import LanguageRegistry, LanguageSpec, Section

    def noop_sections(root: Node, lines: list[str]) -> list[Section]:
        return []

    registry = LanguageRegistry()
    with pytest.raises(ValueError, match="needs a `grammar`"):
        registry.register(
            LanguageSpec(
                name="orphan", kind=Kind.CODE, suffixes=(".orphan",), sections=noop_sections
            )
        )


@needs_grammar("vue")
def test_container_specs_report_themselves_as_containers() -> None:
    from wsindex.ingest.languages import REGISTRY as R

    vue = R.get("vue")
    assert vue is not None
    assert vue.is_container
    assert not vue.is_ast
