# WSIndex

[![CI](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml/badge.svg)](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Frmorenko%2Fwsindex%2Fbadges%2Fcoverage.json)](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml)

**WSIndex** is a CLI that semantically indexes a developer workspace — multiple
repositories at once — and answers natural-language questions with exact
`file:line` locations. Code (Python, JavaScript/JSX, TypeScript/TSX, Java,
C, C++, C#, Go, Rust, Kotlin, PHP, Ruby), front-end components (Vue,
Svelte, Angular templates) and configs (TOML, YAML, JSON, Dockerfile) are
chunked by their syntax trees, docs by headers; every chunk is embedded and
searched by meaning, not by keywords.

## Quickstart

```bash
uv sync --extra ml --extra ast   # engine + real model + tree-sitter grammars
uv run wsindex init myws         # writes wsindex.toml in the current directory
uv run wsindex add-repo wsindex ~/wsindex
uv run wsindex index             # first run downloads the embedding model (~90 MB)
uv run wsindex search "how are markdown files split into chunks"
```

Real output on this very repository:

```
$ uv run wsindex index
files: 48  chunks: 467  written: 466

$ uv run wsindex search "how are markdown files split into chunks" -k 3
wsindex/tests/test_chunker.py:25-29  0.632  def test_doc_markdown_produces_sections() -> None:
wsindex/src/wsindex/ingest/text_chunker.py:107-111  0.598  def chunk_text(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
wsindex/src/wsindex/ingest/text_chunker.py:59-104  0.567  def chunk_markdown(
```

_(After this README itself gets indexed, it will match its own example
query too — semantic search is honest like that.)_

## Incremental indexing

`index` asks git what changed since the commit it last indexed, so a
re-run reads only those files — and deletes the chunks they no longer
produce, which is what keeps an edited file from answering with its old
contents forever.

Measured on the acceptance corpus (149 files, 3458 chunks, real model):

| Scenario           | Files read | Seconds  |
| ------------------ | ---------- | -------- |
| cold (first index) | 149        | 7.6      |
| no changes         | 0          | **0.18** |
| one changed file   | 1          | **0.24** |

Two conditions put a repo on that fast path: it must be a git repository
(a plain directory is a configuration error, not a silent fallback), and
its working tree must be clean. A dirty tree costs a full pass, because a
commit-to-commit diff cannot see uncommitted edits or untracked files —
`index` says so on stderr rather than being quietly slow. The full pass
is a reconcile, not just an append: chunks the current tree no longer
produces are removed either way.

The last indexed commit per repo lives in `state.json` inside the index
directory. It is a cache, so a corrupt or outdated one costs a full
re-index and nothing more; and it is per-machine even when the vectors
sit in shared S3, since two hosts on different branches must not share
one "last indexed commit".

## Repos the workspace fetches for itself

Give a repo a `remote` and `wsindex sync` keeps the working copy current
— clone it if `path` does not exist yet, fast-forward it afterwards, then
re-index whatever moved:

```bash
uv run wsindex add-repo app ~/checkouts/app --remote https://github.com/you/app.git
uv run wsindex sync
```

```
app: updated
files: 2  chunks: 2  written: 2  deleted: 1
```

Repos without a `remote` are checkouts you maintain yourself, and sync
leaves them alone.

**Sync never touches work the remote does not have.** Uncommitted
changes, or local commits that were never pushed, make it decline and say
so on stderr (exit code 1) instead of choosing for you. The only write it
ever performs is a fast-forward, which by definition destroys nothing.
Declining costs speed and not correctness: `index` still runs, and a
dirty tree simply gets the full pass. Use `--no-index` to update working
copies without indexing.

## Reclaiming space

Deleting a chunk hides it immediately but does not free its bytes, and
every write adds a version — so the index grows even when the workspace
does not. `wsindex compact` is the pass that shrinks it:

```
$ uv run wsindex compact
reclaimed 2.8 MB (9.3 MB -> 6.5 MB); 160 -> 2 versions
```

That is the acceptance corpus after a cold index and five incremental
runs: 30% reclaimed in 0.1s, all 3458 chunks still searchable. Most of
those 160 versions come from the cold index alone — chunks are written
per file, so a 149-file repo leaves ~149 versions behind. Compaction is
worth running even on an index that was never edited.

It is deliberately not part of `index`: it is the one command that
discards history, since a store that still holds old versions can be
rolled back and a compacted one cannot. Pass `--keep-days N` when
something else may be reading the same store — a search that began
before the pass would otherwise be reading a version it removes.

## Front-end components

A `.vue` or `.svelte` file is not one language, it is three: a template,
a script and a style block. wsindex splits the file and hands each part
to the language it is actually written in — so a function inside
`<script lang="ts">` is chunked by the real TypeScript extractor, and is
found exactly the way a function in a `.ts` file is.

```
$ uv run wsindex search "how is the title computed" --lang vue
src/Card.vue:10-12  0.584  export function useTitle(): string {
```

Line numbers point at the real line of the real file, and chunks keep the
container's `lang`, so `--lang vue` finds a component's whole contents —
markup, script and styles alike. Angular needs nothing special:
`foo.component.html` is a plain HTML file, chunked one top-level element
at a time.

## Teaching it a new language

The languages above are what wsindex ships with, not what it can index. A
language is one `LanguageSpec` — how to recognize its files, and how to
split them — and an installed package can supply one without any change
to wsindex:

```toml
# in your plugin's pyproject.toml
[project.entry-points."wsindex.languages"]
lua = "wsindex_lang_lua:LANGUAGES"
```

```python
from wsindex.ingest import GrammarSpec, LanguageSpec
from wsindex.ingest.ast import Span, def_span, symbol_name


def lua_spans(root, lines, covered) -> list[Span]:
    """Which parts of the tree deserve a chunk of their own."""
    spans = []
    for child in root.named_children:
        if child.type == "function_declaration":
            name = symbol_name(child)
            if name is not None:
                spans.append(def_span(child, covered=covered, symbol=name, node_type=child.type))
    return spans


LANGUAGES = (
    LanguageSpec(
        name="lua",
        kind=Kind.CODE,
        suffixes=(".lua",),
        grammar=GrammarSpec(module="tree_sitter_lua", getter="language"),
        spans=lua_spans,
    ),
)
```

Install it and the walker starts selecting `.lua` files while the chunker
routes them through the grammar. Whatever your extractor does not claim
becomes a "gap" chunk, so every non-blank line is indexed exactly once
either way. A broken plugin is a warning and a skip, never a crash.

[`examples/wsindex-lang-lua/`](examples/wsindex-lang-lua/) is a complete,
installable plugin — every spelling of a Lua function, with its comment
block attached — and a
[guide for plugin authors](examples/wsindex-lang-lua/README.md):

```bash
uv pip install -e examples/wsindex-lang-lua
uv run wsindex index          # .lua files are now indexed
```

## Storage

The vector index is an embedded [LanceDB](https://github.com/lancedb/lancedb)
database at the `[store] uri` from `wsindex.toml` (default: `.wsindex/`
next to the config). Fully offline once the model is downloaded;
embedding runs in-process. The uri may also point at S3-compatible
storage (`s3://bucket/prefix`) — endpoint and credentials come from the
standard `AWS_*` environment variables, never from the config file.

The storage cost is modest because embedding dominates: on the acceptance
corpus (3458 chunks, real model) indexing takes 7.8s on a local path vs
10.1s over MinIO, and the 7 acceptance searches take 0.1s vs 0.3s. Both
storages return bit-identical results (cross-check delta 0.0000). A MinIO
for local experiments ships in the compose file
(`docker compose up -d minio minio-init` — the init service creates the
`wsindex` bucket).

The design decision (LanceDB replacing the earlier Tensorus + LocalStore
pair) is recorded in [ADR-7](docs/adr/adr-007-post-mvp-storage.md).

## Extras

| Extra | Enables                                                                   | Without it                                                               |
| ----- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `ml`  | `sentence-transformers` embeddings (real semantic search)                 | `--provider fake`: deterministic pseudo-vectors, exact-text matches only |
| `ast` | tree-sitter chunking for py/rs/ts/java code and toml/yaml/json/Dockerfile | sliding-window text chunks for everything                                |

Every grammar degrades independently: a language without its grammar falls
back to plain text chunks, nothing crashes.

## Development

```bash
uv sync --extra ml --extra ast
uv run poe check           # ruff + mypy --strict + pytest (fast suite)
uv run pytest -m slow      # real-model smoke test (network, model download)
```

Tasks are defined in `pyproject.toml` under `[tool.poe.tasks]`; `uv run poe --help`
lists them.

| Task                   | Description                       |
| ---------------------- | --------------------------------- |
| `uv run poe install`   | Sync dependencies (`uv sync`)     |
| `uv run poe lint`      | Lint with ruff                    |
| `uv run poe fmt`       | Format with ruff                  |
| `uv run poe typecheck` | Type-check with mypy              |
| `uv run poe test`      | Run pytest                        |
| `uv run poe check`     | Lint + type-check + test          |
| `uv run poe hooks`     | Run all pre-commit hooks          |
| `uv run poe clean`     | Remove caches and build artifacts |

## Known limitations

- Javadoc and JSDoc comments land in plain gap chunks instead of sticking
  to the definition below them (Rust `///` docs do attach).
- `.tsx` files are not indexed; anonymous TypeScript default exports fall
  into gap chunks.
- Oversized functions stay whole — no window splitting inside a definition.

## Design docs

- Concept — [CONCEPT_en.md](CONCEPT_en.md)
- Business requirements — [BRD_en.md](BRD_en.md)
- Architecture — [ARCH_en.md](ARCH_en.md)

_(Russian originals: `CONCEPT_ru.md`, `BRD_ru.md`, `ARCH_ru.md`.)_

## Toolchain

Package management **uv** · lint & format **ruff** · types **mypy (strict)** ·
tests **pytest** · hooks **pre-commit** · CI **GitHub Actions**.

## License

MIT — see [LICENSE](LICENSE).
