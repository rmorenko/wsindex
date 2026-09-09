"""Tests for the Lua plugin — and a template for testing your own.

A plugin is testable the same way the built-in languages are: feed source
through `chunk_file` and assert the shape that comes out. Nothing here
needs a repository, a store or an index; the plugin's whole contribution
is "which spans does this tree yield", and that is what gets pinned.

Run with:

    uv pip install -e examples/wsindex-lang-lua
    uv run pytest examples/wsindex-lang-lua/tests
"""

from textwrap import dedent

import pytest
from wsindex.ingest import REGISTRY, chunk_file
from wsindex.model import Chunk, Kind

# `importorskip`, not a plain import: without the package installed a
# plain import fails during collection, which is an error rather than a
# skip — a confusing way to meet a test suite whose README tells you to
# run it. This turns "you forgot to install it" into a readable message.
plugin = pytest.importorskip(
    "wsindex_lang_lua",
    reason="run `uv pip install -e examples/wsindex-lang-lua` first",
)
LUA = plugin.LUA
LANGUAGES = plugin.LANGUAGES

pytestmark = pytest.mark.skipif(
    REGISTRY.parser("lua") is None,
    reason="the tree-sitter-lua grammar is not installed",
)

SOURCE = dedent("""\
    -- A tiny module.
    local M = {}

    --- Greets someone.
    function M.greet(name)
    \treturn "hi " .. name
    end

    function M:method(a)
    \treturn a
    end

    local function helper(x)
    \treturn x * 2
    end

    M.render = function()
    \treturn nil
    end

    M.timeout = 30

    return M
""")


def chunks(text: str) -> list[Chunk]:
    return chunk_file(text, repo="r", path="mod.lua", lang="lua", kind=Kind.CODE)


# --- the plugin is discovered at all -------------------------------------


def test_the_entry_point_registers_lua() -> None:
    # If this fails, the package is not installed — the plugin machinery
    # is fine, `pip install -e` just has not run.
    spec = REGISTRY.get("lua")
    assert spec is not None
    assert spec.suffixes == (".lua",)
    assert spec.kind is Kind.CODE


def test_the_exported_tuple_holds_the_spec() -> None:
    # What the entry point resolves to; a tuple so the plugin can grow.
    assert LANGUAGES == (LUA,)


def test_lua_files_are_selected_by_the_walker() -> None:
    from pathlib import Path

    matched = REGISTRY.match(Path("src/init.lua"))
    assert matched is not None
    assert matched.name == "lua"


def test_lua_is_not_a_built_in() -> None:
    # The reason this example is Lua: a language wsindex ships with would
    # claim the same suffix, and the loader would skip the plugin.
    from wsindex.ingest.languages import BUILTIN_LANGUAGES

    assert "lua" not in {spec.name for spec in BUILTIN_LANGUAGES}


# --- what Lua gets chunked into ------------------------------------------


def test_every_spelling_of_a_function_gets_a_chunk() -> None:
    symbols = [c.symbol for c in chunks(SOURCE) if c.symbol]
    assert symbols == ["M.greet", "M:method", "helper", "M.render"]


def test_table_qualified_names_arrive_qualified() -> None:
    # The grammar reports `M.greet` as the name, so `--symbol M` finds a
    # module's whole surface without the extractor assembling anything.
    symbols = {c.symbol for c in chunks(SOURCE)}
    assert {"M.greet", "M:method"} <= symbols


def test_a_plain_assignment_is_not_a_function() -> None:
    # `M.timeout = 30` is a constant; claiming it would make it look like
    # code worth reading on its own.
    assert all(c.symbol != "M.timeout" for c in chunks(SOURCE))


def test_an_assigned_function_is_one() -> None:
    by_symbol = {c.symbol: c for c in chunks(SOURCE)}
    assert by_symbol["M.render"].text.startswith("M.render = function()")


def test_module_prologue_becomes_a_gap_chunk() -> None:
    # Not claimed by the extractor, so the gap pass takes it: still
    # indexed, just without a symbol.
    first = chunks(SOURCE)[0]
    assert first.symbol is None
    assert first.text.startswith("-- A tiny module.")


def test_every_non_blank_line_lands_in_exactly_one_chunk() -> None:
    # The invariant the gap pass exists to guarantee. Worth asserting in
    # any plugin: an extractor that claims overlapping spans breaks it.
    produced = chunks(SOURCE)
    covered = [line for c in produced for line in range(c.start_line, c.end_line + 1)]
    non_blank = {i for i, line in enumerate(SOURCE.splitlines(), 1) if line.strip()}
    assert non_blank <= set(covered)
    assert len(covered) == len(set(covered))  # no line claimed twice


def test_chunk_text_is_a_verbatim_slice() -> None:
    lines = SOURCE.splitlines()
    for chunk in chunks(SOURCE):
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


def test_metadata_flows_through() -> None:
    chunk = chunks(SOURCE)[0]
    assert (chunk.repo, chunk.path, chunk.lang, chunk.kind) == ("r", "mod.lua", "lua", Kind.CODE)


# --- comment blocks belong to what they document -------------------------


def test_a_comment_block_joins_the_function_below_it() -> None:
    # Lua's convention: `---` or `--` directly above the function. That
    # sentence is usually the most searchable thing about it, so filing
    # it as a chunk of its own would waste it.
    documented = next(c for c in chunks(SOURCE) if c.symbol == "M.greet")
    assert documented.text.startswith("--- Greets someone.")
    assert "function M.greet" in documented.text


def test_several_comment_lines_all_join() -> None:
    source = "-- One.\n-- Two.\n-- Three.\nfunction f() end\n"
    documented = next(c for c in chunks(source) if c.symbol == "f")
    assert documented.text.count("--") == 3


def test_a_comment_separated_by_a_blank_line_stays_apart() -> None:
    # A comment cut off from the code below is a remark about the file,
    # not documentation of the next function.
    source = "-- Not documentation.\n\nfunction f() end\n"
    produced = chunks(source)
    assert next(c for c in produced if c.symbol == "f").text == "function f() end"
    assert any("-- Not documentation." in c.text for c in produced if c.symbol is None)


# --- damaged input must not raise ----------------------------------------


def test_a_syntax_error_is_chunked_from_what_parsed() -> None:
    # The parser is error tolerant and so must the extractor be: a file
    # mid-edit is the normal state of a file being searched for.
    assert isinstance(chunks("function broken(\n"), list)


def test_an_empty_file_yields_nothing() -> None:
    assert chunks("") == []


def test_a_file_with_only_a_statement_still_indexes() -> None:
    produced = chunks("local M = {}\n")
    assert len(produced) == 1
    assert produced[0].symbol is None
