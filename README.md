# WSIndex

[![CI](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml/badge.svg)](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Frmorenko%2Fwsindex%2Fbadges%2Fcoverage.json)](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml)

**WSIndex** is a CLI that semantically indexes a developer workspace — multiple
repositories at once — and answers natural-language questions with exact
`file:line` locations. Code (Python, JavaScript/JSX, TypeScript/TSX, Java,
C, C++, C#, Go, Rust, Kotlin, PHP, Ruby), front-end components (Vue,
Svelte, Angular templates) and configs (TOML, YAML, JSON, XML, Dockerfile)
are chunked by their syntax trees, docs by headers; every chunk is embedded
and searched by meaning, not by keywords.

## Quickstart

```bash
uv sync --extra ml --extra ast   # engine + real model + tree-sitter grammars
uv run wsindex init myws         # writes wsindex.toml in the current directory
uv run wsindex add-repo wsindex ~/wsindex
uv run wsindex index             # first run downloads the embedding model (~90 MB)
uv run wsindex search "how are markdown files split into chunks"
uv run wsindex why chunk_markdown        # the commits that wrote it, and why
uv run wsindex refs 8080                 # everything that names this port
```

Real output on this very repository:

```
$ uv run wsindex index
files: 87  chunks: 1228  written: 1227  deleted: 0  commits: 95

$ uv run wsindex search "how are markdown files split into chunks" -k 3
wsindex/README.md:14-38  0.707  ## Quickstart
wsindex/tests/test_chunker.py:41-45  0.632  def test_doc_markdown_produces_sections() -> None:
wsindex/src/wsindex/ingest/text_chunker.py:107-111  0.598  def chunk_text(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
```

_(After this README itself gets indexed, it will match its own example
query too — semantic search is honest like that.)_

## Ranking the results

Search is one funnel. The store returns k×4 candidates by vector
similarity, and — when it is enabled — a cross-encoder reads each
`(query, chunk)` pair properly and re-sorts them down to k:

```toml
[rank]
enabled = true
model = "cross-encoder/ms-marco-MiniLM-L6-v2"
```

Off by default, because it loads a second model. Measured on the
acceptance corpus it moved three of ten queries up and none down —
"expose dataset operations over http" from rank 4 to 1, "generate
embeddings for text" from 4 to 2 — at about 4 ms per pair, so a search
with `-k 5` pays roughly 80 ms.

Two better-known ideas were measured and **not** built, both on this
corpus rather than in the abstract:

- **BM25 hybrid.** Semantics is supposed to be blind to exact rare
  tokens, so identifier queries should fail. They did not: `list_to_tensor`,
  `tensorus-models>=0.0.3` and a verbatim error string all land in the
  top 3, two of them first. A probe also showed Tantivy's default
  tokenizer splits on `_`, `-` and `.`, so BM25 would not have given an
  exact match anyway — only a bag of sub-words, which is where MiniLM is
  already strong. The three identifier queries stay in the acceptance
  criteria as a regression test.
- **MaxSim / late interaction.** One vector per token instead of one per
  chunk costs **×123** on this corpus (651 MB against 5.3 MB), and 19% of
  chunks are longer than the model reads anyway, so the per-token
  representation is truncated exactly like the pooled one. Re-scoring the
  same candidate set with MaxSim made ranking *worse* — mean rank 2.10 →
  2.30, and the two queries it pushed down are the two the cross-encoder
  pulls up. This tested MaxSim over a bi-encoder's token embeddings, not
  ColBERT, whose token vectors are trained for it; that would be a second
  model and its own index format.

The threshold for revisiting either is a real recall failure: a query
whose answer never reaches the candidate set at all. Re-ranking cannot
help there, and neither can any amount of it.

## Marking up a repository

What is worth indexing is a property of a repository, not of a
workspace: in an Angular repo a `.component.html` is source, in a Python
repo an `.html` is generated noise. Two optional keys per repo say so —
and deliberately only two, because the third would be the start of a
rules engine:

```toml
[[repos]]
id = "app"
path = "~/checkouts/app"
ignore = ["vendor/*", "*.min.js"]

[repos.formats.".sql"]
lang = "sql"
kind = "code"
```

`ignore` takes path globs, matched against the whole repo-relative path;
`*` crosses directory separators, so `vendor/*` means the whole subtree.

`formats` maps a suffix to a language and a kind, overriding the built-in
table for this repo alone. **A language with no grammar is the point, not
a limitation**: the chunker falls back to sliding windows, so marking
`.sql`, `.proto` or `.tf` makes them searchable immediately, and an AST
extractor can be bought later for the ones that earn it. Point a suffix
at a language that *does* have a grammar — `".pom" = { lang = "xml" }` —
and it gets the syntax tree for free.

Anything wsindex does not recognize in a repo entry is an error rather
than a shrug: a misspelled `ignores` that silently indexed everything it
was meant to exclude is the mistake this format invites most.

Changing the markup re-reads the repository on the next `index`, even
though git reports the tree as unchanged — the index remembers which
markup produced it, because a commit alone does not say which files were
selected from it.

Build output is skipped everywhere, no configuration needed:
`node_modules`, `target`, `dist`, `build`, `out`, `coverage`, `htmlcov`,
`__pycache__` and every dot-directory. That list is not overridable, which
is a real limitation: a repository whose `build/` holds source has no way
to say so. `ignore` narrows, nothing widens.

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
files: 2  chunks: 2  written: 2  deleted: 1  commits: 1
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

## Asking why, and who

Two commands read what indexing recorded.

`why` walks from a definition to the commits that wrote its lines, and
prints their reasoning — which is usually the only place it exists:

```
$ uv run wsindex why add_chunks
LanceDBStore.add_chunks  self/src/wsindex/store/lancedb.py:171-209
  written by:
    3964bb7  feat: LanceDBStore — single-table vector store per ADR-7
        dedup before embedding (batch-internal included), one Lance commit
        per add_chunks call.
        see PROJ-412 -> https://jira.example.com/browse/PROJ-412
```

`refs` is the inverted index over links: ask about a port, a ticket, a
commit or a url, and it answers who names it.

```
$ uv run wsindex refs 8080
8080
  read by:
    svc/client.py:1
  (nothing declares it — code and configuration have drifted)
```

Not "who calls this function": code-to-code edges are deferred until they
can be shown to pay for their noise — 11% of resolvable call names in
this repository are ambiguous — so a function name has no callers to
list yet. See [ADR-9](docs/adr/adr-009-links-as-entities.md).

## Commits are part of the corpus

A repository's reasoning is not in its code. "Why is dedup before
embedding" is answered in a commit message and nowhere else. So `index`
reads the history too: each message becomes a searchable chunk, and
`git blame` becomes edges from a chunk of code to the commits that wrote
its lines.

```
$ uv run wsindex index
files: 87  chunks: 1228  written: 1227  deleted: 0  commits: 95

$ uv run wsindex search "why is dedup done before embedding" --kind commit
```

Commits are their own `--kind`, not documents: folding them into `doc`
would make that filter mean two things and leave no way to search the
code without the history.

Cheap, and proportional to the work. Reading a whole history takes
milliseconds; blame costs ~28 ms per *indexed* file, which the
incremental pass already keeps down to what changed. Untracked files have
no history and simply get no edges.

## References out of the repository

A commit message names a ticket; a document links a page. `index` records
those as links, so "why is this here" can reach the tracker. Nothing is
downloaded — it is a bridge, not a connector.

Bare urls are recognised out of the box. Anything else has to be
declared, because a pattern cannot guess:

```toml
[references]
"PROJ-" = "https://jira.example.com/browse/PROJ-{key}"
"#" = "https://github.com/org/repo/issues/{key}"
"!" = "https://gitlab.example.com/org/repo/-/merge_requests/{key}"
```

`{key}` is the digits. Declaring the prefix is what makes this useful
rather than noisy: a built-in `[A-Z]+-\d+` rule looks like a Jira key and
also matches `ADR-7`, `UTF-8` and `ISO-8601` — measured on this
repository it produced 198 matches and not one was a ticket.

## Fetching what a reference points at

A link says where a document is; a connector goes and gets it. One
document at a time, by url — nothing walks a wiki space, so the cost of
an external source stays proportional to what the repository actually
mentions.

```console
$ wsindex fetch https://github.com/astral-sh/uv/issues/1
https://github.com/astral-sh/uv/issues/1
title: Add basic GitHub Actions CI
author: charliermarsh
state: closed
...
```

Which connector answers which url is configuration:

```toml
[[connectors]]
type = "github"
url_pattern = "https://github.com/myorg/*"
token_env = "GITHUB_TOKEN"

[[connectors]]
type = "generic-http"
url_pattern = "https://*"
```

First match wins, in file order — so the specific entry goes above the
catch-all. `github` fetches an issue or a pull request (one API endpoint
answers for both) and hands back the body as its author wrote it;
`generic-http` takes anything already textual, and strips HTML to what a
reader would see — a documentation page measured here went from 83 KB of
markup to 6.9 KB of text.

`token_env` names an **environment variable**, never a token. The config
file gets committed; the secret is read at fetch time and stored nowhere
— the same rule the S3 backend and `wsindex sync` keep. A connector
configured with a token env var that is not set refuses to run rather
than falling back to an anonymous request: GitHub answers 404 for a
private repository, which would otherwise read as "no such issue" and
send you looking in the wrong place.

## Teaching it a new source

The two connectors above are what wsindex ships with, not what it can
fetch. A source is one `Connector` subclass, and an installed package can
supply one without any change to wsindex — the same seam the language
plugins use. The name on the left is the `type` a config entry asks for:

```toml
# in your plugin's pyproject.toml
[project.entry-points."wsindex.connectors"]
notion = "wsindex_connector_notion:NotionConnector"
```

```python
from wsindex.connectors import Connector, Document


class NotionConnector(Connector):
    def matches(self, url: str) -> bool:
        """Narrower than the config's pattern: what you can really fetch."""
        return page_id(url) is not None

    def fetch(self, url: str) -> Document:
        """Whatever the source speaks, turned into text."""
```

`matches` and `fetch` are separate so a url can be routed without
anything being requested over the network, and a connector that says no
lets the router fall through to a generic entry. A plugin may not take a
name that already exists — `github` means the built-in, and a config that
says so must keep meaning it. A broken plugin is a warning and a skip,
never a crash.

[`examples/wsindex-connector-notion`](examples/wsindex-connector-notion/)
is the worked one, and it is Notion because Notion's API returns neither
markdown nor text: a page is a tree of blocks, so the connector has to
rebuild the document rather than pass it along. That is the case most
real sources are, and the one the built-ins do not exercise.

## Keeping a snapshot of what you fetch

A fetched document does not go into the index directly. It is written to
disk as markdown, in a directory that is itself a git repository, and
`index` reads that repository like any checkout:

```toml
[[repos]]
id = "docs"
path = "snapshots/docs"
source = "connector"
urls = [
  "https://github.com/myorg/handbook/issues/42",
  "https://wiki.example.com/retention-policy",
]
```

```console
$ wsindex sync
docs: 4 added
files: 4  chunks: 25  written: 25  deleted: 0  commits: 1

$ wsindex sync
docs: up to date (4 documents)
files: 0  chunks: 0  written: 0  deleted: 0  commits: 0
```

The second run costs nothing, and that is the design rather than an
optimization. Every sync is a commit, so the snapshot's `git log` is the
history of the source — a wiki that keeps none, or a tracker whose
history is a list of field changes, becomes `git log -p`. Nothing about
the fetch is recorded in the file, only what the source said about the
document: a timestamp would make every sync a diff and turn that log
into a heartbeat.

Because the snapshot is a git repository, incremental indexing applies to
it unchanged, and a search hit points at a line of a file that exists.
The url lives in the frontmatter, so the live page is one hop away, but
the citation is not a claim about a page that may have changed since.

Each pass reconciles rather than accumulates: drop a url from the config
and the file is deleted, with the deletion in the log. A document that
*fails* to fetch keeps its file — a timeout is not a deletion, and a
history whose job is to say when things changed must not claim one.

## Drift between code and configuration

While indexing, wsindex notes two things: ports that code expects to
reach, and ports that configuration publishes. A reference nothing
answers is reported — that is the only automatic evidence that the two
have grown apart:

```
$ uv run wsindex index
files: 2  chunks: 3  written: 3  deleted: 0  commits: 1
drift: 1 unresolved config reference(s), first at svc/client.py:1 -> 8080
```

This repository's own history is the test case: `config.py` once
defaulted to `http://localhost:8080` while `docker-compose.yml` published
the service on 8000. Nothing failed; it was found by hand, later.

The links live in `links.db` beside the index and are keyed by chunk, so
they are deleted exactly when their chunk is. That is what keeps the
report readable — and it is also why deleting the config that published a
port makes the code reading it drift again, with nothing to update by
hand. See [ADR-9](docs/adr/adr-009-links-as-entities.md).

## Asking many questions at once

Most of a `wsindex search` is spent before it searches anything: loading
the embedding model, opening the store. `wsindex shell` pays that once.

```bash
uv sync --extra shell
uv run wsindex shell
```

```
wsindex> how are chunks deduplicated --lang python
  1  0.71  wsindex/src/wsindex/store/lancedb.py  171-209  add_chunks
  2  0.63  wsindex/tests/test_lancedb_store.py    90-101  test_duplicates…
wsindex> 1            # show that hit in full, highlighted
wsindex> :open 1      # and in $EDITOR, at the right line
```

Arrow keys walk the history, Tab completes flags and repo ids, and the
query flags are the ones `wsindex search` takes — the shell is another
adapter over the same library, so `--lang python` means the same thing in
both.

## Search from an agent

An agent client speaks MCP, so the answer to "is there an IDE plugin" is
that no plugin has to exist:

```bash
uv sync --extra mcp
uv run wsindex mcp        # stdio; point a client's command at this
```

Three tools — `search`, `refs` and `why` — the same three commands worth
calling from outside. `index` is deliberately not one of them: a tool an
agent may call again without thinking should not be minutes of CPU and
somebody's git remotes.

A workspace already running `wsindex serve` offers the same tools over
HTTP at `/mcp`, from the same tool code. Two transports, one
implementation.

## Running it as a server

The same engine behind HTTP, for a workspace more than one person
searches:

```bash
uv sync --extra server
export WSINDEX_TOKEN=...
uv run wsindex serve            # http://127.0.0.1:8000, admin at /admin
```

```toml
[server]
token_env = "WSINDEX_TOKEN"   # the variable's name, never the token
interval = 900                # seconds between automatic syncs; 0 = off
```

`GET /search?q=...&k=&repo=&lang=&kind=&path=&symbol=` is this CLI's
`search` with its flags as query parameters, `POST /index` is `index`,
`GET /status` reports the workspace and the last runs, and `/healthz`
answers without a token because a load balancer is not a reader.
`POST /hooks/sync` syncs and re-indexes now — point a git host's webhook
at it; the body is ignored, since "something changed" is all an
incremental run needs to hear. OpenAPI comes free at `/openapi.json`.

`/admin` is a page with the repo list, a sync button and the recent runs.

**Nothing server-shaped leaks into the engine.** Every endpoint is a call
into the same library the CLI uses — an endpoint that could not be
written that way would mean the library was missing something, not that
the server should grow it. See
[ADR-10](docs/adr/adr-010-library-server-boundary.md).

**Two processes, one index.** Measured rather than assumed: concurrent
writers lose nothing, a stale writer is still a correct writer, and
deduplication holds across processes. The one real hazard is reading —
a handle is pinned to the version it opened at, so a long-lived process
would answer from the corpus it started with and never fail doing it.
Searches refresh first, at about 4 ms against a 111 ms search. Indexing
runs one at a time and a second caller is told so (409) rather than
queued: two runs of the same repo do the same work twice.

Binding is `127.0.0.1` unless you say otherwise, and a `token_env`
naming an unset variable stops the server from starting rather than
opening the index to whoever can reach the port.

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

| Extra    | Enables                                                                       | Without it                                                               |
| -------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `ml`     | `sentence-transformers` embeddings (real semantic search)                     | `--provider fake`: deterministic pseudo-vectors, exact-text matches only |
| `shell`  | `wsindex shell`: one loaded model, many questions (prompt-toolkit)            | a fresh process and a fresh model per search                             |
| `mcp`    | `wsindex mcp`: the index as tools for an agent client                         | search from the CLI, HTTP or shell                                       |
| `server` | `wsindex serve`: HTTP API, scheduler and admin page (FastAPI)                 | search from the CLI only                                                 |
| `ast`    | tree-sitter chunking for py/rs/ts/java code and toml/yaml/json/xml/Dockerfile | sliding-window text chunks for everything                                |

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
