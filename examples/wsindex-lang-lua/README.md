# wsindex-lang-lua

Lua support for [wsindex](../../README.md), and the worked example every
language plugin can be copied from.

Install it and `.lua` files start being indexed — every function as its
own chunk, however it is spelled. No change to wsindex itself:

```bash
uv pip install -e examples/wsindex-lang-lua
uv run wsindex index
uv run wsindex search "how does it greet someone" --lang lua
```

```
demo/mod.lua:4-7   0.508  --- Greets someone.
demo/mod.lua:9-11  0.312  function M:method(a)
```

The hit starts at the comment because the comment and the function it
documents are one chunk — see [below](#writing-the-extractor).

**Why Lua?** Because it is deliberately *not* a language wsindex ships
with, so this example cannot collide with a built-in. An earlier version
did Go — and then Go moved into the box, the two claimed `.go`, and the
loader started skipping the plugin with a conflict warning. Correct
behaviour, bad look on an example.

## What a plugin is

Three things, and the third is the only one that takes thought.

**1. A dependency on the grammar.** wsindex ships grammars for the
languages it supports itself; a plugin ships its own.

```toml
dependencies = ["tree-sitter-lua>=0.5,<1"]
```

**2. An entry point.** This is the entire integration surface — the line
that makes an installed package visible to wsindex:

```toml
[project.entry-points."wsindex.languages"]
lua = "wsindex_lang_lua:LANGUAGES"
```

Nothing scans the filesystem for plugins. The build backend copies that
into `wsindex_lang_lua-0.1.0.dist-info/entry_points.txt` when the package
is installed, and wsindex reads it back with `importlib.metadata`. A
plugin becomes visible by being **installed**, which is why the core
needs no list of plugins.

The value is `module:attribute`, and the attribute may be one
`LanguageSpec` or an iterable of them — so a plugin covering a language
and its template dialect declares one entry point, not two.

**3. A `LanguageSpec`.** Everything wsindex needs to know:

```python
LUA = LanguageSpec(
    name="lua",  # the `lang` on every chunk
    kind=Kind.CODE,  # CODE, CONFIG or DOC
    suffixes=(".lua",),  # lowercase, dot-prefixed
    grammar=GrammarSpec(module="tree_sitter_lua", getter="language"),
    spans=lua_spans,  # the interesting part
)
```

Register that and every stage picks it up: the walker starts selecting
`.lua` files, the chunker starts routing them through the grammar.

A `DOC` language needs neither `grammar` nor `spans` — docs are chunked
by headers, not by syntax. For `CODE` and `CONFIG` the two go together;
wsindex refuses a spec with one and not the other, because an extractor
with no tree to read and a tree no extractor reads are both dead weight.

## Writing the extractor

`spans` is called once per file and answers one question: *which parts of
this tree deserve a chunk of their own?* Everything you do not claim is
swept into "gap" chunks, so every non-blank line is indexed exactly once
whether or not you thought about it.

```python
def lua_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
    spans = []
    for child in root.named_children:
        if child.type == "function_declaration":
            name = symbol_name(child)
            if name is not None:
                spans.append(def_span(child, covered=covered, symbol=name, node_type=child.type))
    return spans
```

`def_span` builds the span *and* marks its lines in `covered`; that
bookkeeping is what keeps the gap pass from emitting the same lines
twice. The helpers come from `wsindex.ingest.ast`:

| Helper         | Use                                                               |
| -------------- | ----------------------------------------------------------------- |
| `def_span`     | Claim a definition's lines and return its span                    |
| `symbol_name`  | Read a node's `name` field; None on a damaged tree                |
| `gap_spans`    | Chunk the lines *inside* a definition that nothing nested claimed |
| `line_span`    | A node's 1-based inclusive line range                             |
| `unwrap`       | Reach the definition inside a decorator or `export` wrapper       |
| `mark_covered` | Mark lines by hand, when `def_span` does not fit                  |

Three habits worth copying from this package:

- **Return, do not raise, on a damaged tree.** The parser is error
  tolerant, so `symbol_name` can be None on a file mid-edit. Skip that
  node and let the gap pass cover its lines — a raise would cost the
  whole file.
- **Qualify symbols the way a reader would search.** Lua hands you
  `M.greet` and `M:method` already qualified, so `--symbol M` finds a
  module's whole surface. Where a grammar does not, assemble it — the
  built-in Python extractor writes `Cls.method`, C++ writes
  `Class::method`, each using its own language's separator.
- **Name the grammar, do not import it.** `GrammarSpec` takes a module
  name so the spec stays declarable when the grammar is missing; wsindex
  then falls back to plain text windows for `.lua` files rather than
  dropping the language.

## When it goes wrong

A broken plugin is a warning and a skip, never a crash — someone else's
package must not cost the user their workspace. You will see one of
these on stderr:

```
language plugin 'lua' skipped: import failed — No module named 'tree_sitter_lua'
language plugin 'lua' skipped 'lua': suffix 'lua' must be lowercase and start with '.'
language plugin 'lua' skipped 'lua': suffix '.lua' is already claimed by 'luau'
```

Failures are per language, not per package: a plugin offering three
languages, one of which collides with something already installed,
contributes the other two.

## Running its tests

```bash
uv pip install -e examples/wsindex-lang-lua
uv run pytest examples/wsindex-lang-lua/tests
```
