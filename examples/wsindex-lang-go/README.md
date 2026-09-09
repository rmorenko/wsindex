# wsindex-lang-go

Go support for [wsindex](../../README.md), and the worked example every
language plugin can be copied from.

Install it and `.go` files start being indexed — functions, methods and
named types each as their own chunk. No change to wsindex itself:

```bash
uv pip install -e examples/wsindex-lang-go
uv run wsindex index
uv run wsindex search "how is a request served" --lang go
```

```
demo/server.go:23-27  0.412  // Serve writes the request path to the log and returns.
demo/server.go:8-12   0.361  // Server owns the listening socket and routes requests to handlers.
```

The hit starts at the doc comment because the comment and the method it
documents are one chunk — see [below](#writing-the-extractor).

## What a plugin is

Three things, and the third is the only one that takes thought.

**1. A dependency on the grammar.** wsindex ships grammars for the
languages it supports itself; a plugin ships its own.

```toml
dependencies = ["tree-sitter-go>=0.23,<1"]
```

**2. An entry point.** This is the entire integration surface — the line
that makes an installed package visible to wsindex:

```toml
[project.entry-points."wsindex.languages"]
go = "wsindex_lang_go:LANGUAGES"
```

Nothing scans the filesystem for plugins. The build backend copies that
into `wsindex_lang_go-0.1.0.dist-info/entry_points.txt` when the package
is installed, and wsindex reads it back with `importlib.metadata`. A
plugin becomes visible by being **installed**, which is why the core
needs no list of plugins.

The value is `module:attribute`, and the attribute may be one
`LanguageSpec` or an iterable of them — so a plugin covering a language
and its template dialect declares one entry point, not two.

**3. A `LanguageSpec`.** Everything wsindex needs to know:

```python
GO = LanguageSpec(
    name="go",  # the `lang` on every chunk
    kind=Kind.CODE,  # CODE, CONFIG or DOC
    suffixes=(".go",),  # lowercase, dot-prefixed
    grammar=GrammarSpec(module="tree_sitter_go", getter="language"),
    spans=go_spans,  # the interesting part
)
```

Register that and every stage picks it up: the walker starts selecting
`.go` files, the chunker starts routing them through the grammar.

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
def go_spans(root: Node, lines: list[str], covered: list[bool]) -> list[Span]:
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
- **Qualify symbols the way a reader would search.** Go methods become
  `Server.Serve`, matching what the built-in Python extractor does for
  class methods, so `--symbol Server` finds a type's whole method set.
- **Name the grammar, do not import it.** `GrammarSpec` takes a module
  name so the spec stays declarable when the grammar is missing; wsindex
  then falls back to plain text windows for `.go` files rather than
  dropping the language.

## When it goes wrong

A broken plugin is a warning and a skip, never a crash — someone else's
package must not cost the user their workspace. You will see one of
these on stderr:

```
language plugin 'go' skipped: import failed — No module named 'tree_sitter_go'
language plugin 'go' skipped 'go': suffix 'go' must be lowercase and start with '.'
language plugin 'go' skipped 'go': suffix '.go' is already claimed by 'golang'
```

Failures are per language, not per package: a plugin offering three
languages, one of which collides with something already installed,
contributes the other two.

## Running its tests

```bash
uv pip install -e examples/wsindex-lang-go
uv run pytest examples/wsindex-lang-go/tests
```
