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
uv run wsindex status                    # what is configured, and what the index holds
uv run wsindex explain src/thing.tf      # why a file is (or is not) searchable
uv run wsindex domains                   # what it is made of, and what is tangled
```

Real output on this very repository:

```
$ uv run wsindex index
files: 87  chunks: 1228  written: 1227  deleted: 0  commits: 95  in 6.42s

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

**It does not scale, and the number is worth knowing before turning it
on.** Under load (`poe load --rerank`) the 500 ms budget holds to **six**
concurrent searches and breaks at eight — p95 438 ms against 607 ms — and
throughput is flat at 16-17 searches per second however many clients ask.
Latency simply grows about 73 ms per extra client.

That ceiling is hardware, not software, which is the part that decides
what to do about it. Both models run on the GPU (`mps` on this machine),
and one re-ranking of twenty realistic candidates costs 54 ms of it.
Threads buy nothing (18.3/s at one, 18.4/s at eight) and separate
processes barely more (18.5/s at one, 28.0/s at four) — they contend for
the same device. `torch.set_num_threads(1)` changes the figure not at
all, which is how the CPU was ruled out: it is a CPU knob, and this is
not CPU work. So a busier server needs another machine, a smaller
cross-encoder or fewer candidates; running more copies of this one will
not do it.

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

That check runs when a command runs. To get it while you are still
typing, point your editor at `wsindex.schema.json` — one line at the top
of the file, understood by taplo, VS Code's Even Better TOML and the
JetBrains TOML plugin:

```toml
#:schema https://raw.githubusercontent.com/rmorenko/wsindex/main/wsindex.schema.json
```

The schema is *generated* from the same constants the validator uses
(`poe schema`), not written a second time — a hand-kept copy would drift,
and a schema that lies is worse than none, because it underlines a config
that works. A test asserts the checked-in file is what the generator
produces.

Changing the markup re-reads the repository on the next `index`, even
though git reports the tree as unchanged — the index remembers which
markup produced it, because a commit alone does not say which files were
selected from it.

Build output is skipped everywhere, no configuration needed:
`node_modules`, `target`, `dist`, `build`, `out`, `coverage`, `htmlcov`,
`__pycache__` and every dot-directory. That list is not overridable, which
is a real limitation: a repository whose `build/` holds source has no way
to say so. `ignore` narrows, nothing widens.

**A file is indexed only if a suffix claims it, and only if it really is
a file in that repository.** Both halves are load-bearing. The suffix
table is why `.env`, `id_rsa`, `key.pem` and `.netrc` are not indexed —
treat that as policy, not luck, before adding a suffix. And a symlink is
skipped rather than followed: `is_file()` follows one, so a repository
holding `notes.md -> ~/.ssh/id_rsa` had the key's contents indexed, which
let the author of a cloned repository pick which of *your* files went
into your index.

What the suffix table does not save you from is a `secrets.yaml` or a
`credentials.json` sitting in a repo — those are config files and get
indexed like any other. Untracked files count too, as long as
`.gitignore` does not exclude them.

Six rules can leave a file out, and from outside they all used to look
the same: the file was simply absent. `wsindex explain` names the one
that caught yours, because each points at a different fix.

```console
$ wsindex explain src/schema.tf
r/src/schema.tf: not indexed — no language claims this suffix; add a `formats` entry for it

$ wsindex explain src/main.py
r/src/main.py: indexed as python (code)
  7 chunk(s), 6 with a symbol

$ wsindex explain src/half-written.py
r/src/half-written.py: indexed as python (code)
  2 chunk(s), but the python grammar reported errors — the parts it could not read are indexed as text, not definitions
```

`wsindex status` answers the other half — whether the index has run at
all, and at which commit:

```console
$ wsindex status
repos:
  self -> /Users/me/wsindex  (indexed 2026-09-10 08:47 at e15022e)
  new  -> /Users/me/other    (not indexed)
```

A repo that has never been indexed takes no part in any search, so
`search` says so before showing results rather than letting a partial
answer look like a whole one:

```console
$ wsindex search "how are chunks deduplicated"
warning: not searched (never indexed): new — run `wsindex index`
```

## Incremental indexing

`index` asks git what changed since the commit it last indexed, so a
re-run reads only those files — and deletes the chunks they no longer
produce, which is what keeps an edited file from answering with its old
contents forever.

Measured on the acceptance corpus (149 files, 3458 chunks, real model):

| Scenario           | Files read | Seconds  |
| ------------------ | ---------- | -------- |
| cold (first index) | 149        | 4.5      |
| no changes         | 0          | **0.15** |
| one changed file   | 1          | **0.24** |

Two conditions put a repo on that fast path: it must be a git repository
(a plain directory is a configuration error, not a silent fallback), and
its working tree must be clean. A dirty tree costs a full pass, because a
commit-to-commit diff cannot see uncommitted edits or untracked files.
The full pass is a reconcile, not just an append: chunks the current tree
no longer produces are removed either way.

When a repo does go the long way, `index` says which one and why —
naming the reason rather than listing the possibilities, because a note
that offers three causes names none of them when the real one is a
fourth:

```console
$ wsindex index
files: 122  chunks: 1795  written: 1795  deleted: 0  commits: 115  in 8.71s
note: full pass for self — uncommitted work, which a commit-to-commit diff cannot see; commit or stash it
warning: could not read 1 tracked file(s) — self/src/locked.py
```

That last line is not a policy skip. A file git tracks and the
filesystem refuses to open was *meant* to be indexed; saying `files: 121` and nothing else would have left you to discover it by noticing a
search that finds nothing. A file the grammar cannot read gets its own
warning for the same reason: it *is* indexed, as text windows rather than
definitions, which is worse to search and — until it says so —
impossible to notice. Syntax newer than the installed grammar, and a
`formats` entry aimed at the wrong language, both land here.

**When the message is not enough, `WSINDEX_DEBUG=1` opens the door.**
The traceback comes through instead of one tidy line, the library's own
log records reach stderr, and blame runs in a single thread so a
breakpoint lands where you put it. A variable rather than a flag, because
by the time you want it the command has already failed — set it and
repeat the same line.

```console
$ wsindex index
error: self/src/wsindex/pipeline.py: RuntimeError: the chunker fell over

$ WSINDEX_DEBUG=1 wsindex index
...the whole traceback, with source context
```

Note the file in that message. An index run touches a hundred-odd files,
and a failure in one of them used to arrive as `error: the chunker fell over` with no way to tell which. The default is otherwise unchanged, and
so is the rule behind it: a traceback in ordinary output is a bug. This
is a door, not a reversal.

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
milliseconds. Blame is per file, so it runs `git blame` once per
*indexed* file — which the incremental pass already keeps down to what
changed — in a pool, and in a child process (see
[ADR-12](docs/adr/adr-012-spawning-processes.md)). Untracked files have
no history and simply get no edges.

**How far back it reaches is the one setting that depends on the
repository rather than the machine**, so it is the one that is
configurable:

```toml
[index]
max_commits = 10000   # a full pass; incremental runs read the diff instead
```

The default used to be a thousand, and measuring six real repositories
showed that to be an order of magnitude too tight. On five of them the
cap never fires at all — they have fewer than a thousand commits. On the
sixth it cut 10 351 commits to the newest 1 000, leaving history visible
back to 2022 in a project that starts in 2005: **17.7 years and 90% of
the commits, invisible**, to save 23 MB and 15 seconds on an index that
already took 130 seconds and 181 MB. For a feature whose premise is that
the reasoning lives in the messages, that is the trade the wrong way
round.

A cap still has to exist — a repository the size of the Linux kernel
would add over a million chunks — and history costs about 1.4 s and
2.5 MB per thousand commits, so ten thousand bounds the worst case at
roughly what openemr's *entire* history costs. How much this governs
varies more than any other constant here: measured, the share of an index
that is commit messages runs from 0% (a repository with no history)
through 6.1% (this project) to 31.3%, where the cap decides a third of
everything searchable.

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

The other half of that rule is where the token *goes*. Redirects are
followed, but not with the token attached: `urllib` copies headers onto
a redirect across hosts, so one 302 from a source that changed domains
used to hand `$GITHUB_TOKEN` to whoever answered. Same-host redirects
keep it, since that is the server it was sent to.

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

**`search` takes a token budget, which is the one thing `k` cannot say.**
Twenty hits on this corpus cost 4 201 tokens, and until now the caller
found that out by spending them. `budget` trims the answer to fit and
names what it dropped:

```
budget  200 ->  2 hits,  156 tokens, 18 dropped
budget 1000 ->  4 hits,  649 tokens, 16 dropped
budget 2000 ->  8 hits, 1348 tokens, 12 dropped
```

A prefix, not a knapsack: hits arrive best-first, and skipping a large
one to fit two small ones would quietly trade relevance for bytes. The
count is exact — the store asks its own tokeniser — and falls back to a
measured three-characters-per-token estimate for a backend that has none,
erring high so a budget is never overrun.

This is all that survived spiking the "context pack" idea. Assembling
context *better* turned out to have nothing to improve: filling a
1 000-token budget with plain search already holds the expected answer
for all ten acceptance queries, in fifteen chunks. Adding provenance
changed nothing. What was actually missing was the budget itself.

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

`/admin` is a page with the repo list, a sync button, the recent runs and
what gets asked — the same aggregate `wsindex stats` prints: how many
searches, how long they took, the questions this corpus answered *worst*
and the ones asked most.

Aggregate, and that is a decision rather than a shortcut. The step this
came from asked for analytics **per user**; there is no user to split by
— the token belongs to the server, not to a person, and the search log
deliberately records no identity — and inventing one would mean per-user
authentication plus a log of other people's questions attributed to them.
That is a privacy decision, not a feature of a page. So the page shows
everyone's questions together, says so in those words, and names the two
switches: `[stats] enabled = false` to stop recording,
`wsindex stats --forget` to empty it.

`GET /metrics` is Prometheus exposition — request counts and durations by
route, indexing runs by outcome, chunks written, when the last run
succeeded. Behind the token, unlike `/healthz`: a probe is not a reader,
and these describe the workspace. **No query text is ever a label**, which
is both a cardinality rule and the same privacy rule `wsindex stats`
follows. The 500 ms bucket is the SLO below, so compliance is a division
of two scraped series rather than a recording rule:

```
wsindex_http_request_seconds_bucket{route="/search",method="GET",le="0.5"} 238
wsindex_index_runs_total{outcome="busy"} 7
wsindex_last_index_success_timestamp_seconds 1.7889e+09
```

**How much load it takes.** `poe load` starts a real server on a real
corpus and judges it against an SLO fixed *before* the first run — p95
under 500 ms, p99 under 1 s, no failed requests — the same discipline the
acceptance criteria use, and for the same reason.

Both harnesses record the machine's load and refuse to judge on a busy
one, which is not caution but experience: a run here reported a 30%
regression that was another program using ten of fourteen cores.
Repeating a measurement cannot fix that — every repeat meets the same
disturbance, and the minimum of three spoiled runs is a spoiled run.
Measured on an idle M-series laptop with eight concurrent clients:

| Scenario                          |   p50 |    p95 | verdict                |
| --------------------------------- | ----: | -----: | ---------------------- |
| search, idle server               | 56 ms |  76 ms | PASS                   |
| search during an ordinary sync    | 79 ms | 129 ms | PASS                   |
| search during a **full** re-index | 91 ms | 133 ms | PASS                   |
| 8 simultaneous `POST /index`      |     — |      — | PASS: 1 run, 7 refused |

Eleven rules of eleven — but only after the third row failed at
**1459 ms** and was chased down rather than negotiated. What caused it:
the blame pass runs eight `git blame` processes at once, and **starting a
process from the process that holds the embedding model is where the time
goes.** 117 spawns of `git --version` — a command that does nothing —
block a search for 1.4 s just the same, and eight threads doing it get no
parallelism at all (9.7 ms per spawn either way). Three tidier
explanations were measured and ruled out first: the embedding batch size
(chopping `encode` to 32 changes nothing), torch's thread count (one
thread stalls the same 1.8 s) and the GIL (a continuous monitor loses
53 ms at worst).

The mechanism, chased down afterwards because it decides whether the cure
generalises. The cost is in the **exec**, not the fork: a bare fork storm
leaves a working thread at 1.1x its idle speed, fork+exec puts it at 27x,
and pipes make no difference. After a fork the child holds a
copy-on-write copy of the parent's address space and exec must tear it
down first — so the price is the parent's *map*, not its bytes. A
gigabyte in one mapping costs what nothing costs (1.01 ms an exec against
0.90); the same gigabyte in sixteen thousand mappings costs 2.82 ms; the
model costs 2.43 ms with only three thousand regions, because its regions
are file-backed mappings of large dylibs. Concurrent execs serialise on
that teardown, which is the missing parallelism.

**This is a macOS problem.** On Linux CPython uses `vfork`, no copy is
made, nothing is torn down: sixteen thousand mappings cost 0.37 ms an
exec against a bare process's 0.49, and the worker stays at 2.6x rather
than 34.7x. The cure is kept unconditional anyway — on Linux it is one
process start per batch, 21 ms against a pass that takes seconds, which
is cheaper than a platform branch and a second path that only half the
machines would test.

The rule this leaves behind — a program run per file belongs in a child,
not in the engine — is [ADR-12](docs/adr/adr-012-spawning-processes.md),
with a test that fails if a future ingest step forgets it.

`poe load --rerank` measures the other configuration, the one the SLO was
deliberately not written for: with a cross-encoder in the funnel every
search scenario misses 500 ms at eight clients (607, 801 and 928 ms).
That is a capacity limit rather than a defect — see "Ranking the results"
for what it is and why more processes are not the answer — and
re-ranking is off by default.

The cure itself:
[`wsindex.ingest.blame`](src/wsindex/ingest/blame.py) hands the batch to a
small child that imports nothing from this package, and the child does
the spawning. Search during a full re-index went **1459 ms → 133 ms** p95,
and a cold index got **11% faster** (8.18 s → 7.29 s) — those forks were
never necessary work. Under four files the batch still runs in-process,
where a child would cost more than the forks it saves; that threshold is
derived rather than chosen, and `index-one-file` is unmoved at 0.32 s.

Two cures that *were* trades got measured and dropped on the way. `nice`
does nothing (2079 ms → 2143 ms). Fewer blame workers do work (2 workers:
p95 284 ms) at 48% of indexing throughput. `posix_spawn` is faster again,
but CPython only takes it with `close_fds=False`, and a probe showed the
child then inherits seven of this process's descriptors.

**It writes down what it did.** `serve` turns on uvicorn's access log
and the library's own records; every CLI command stays silent, because a
log line is not an interface for somebody watching a terminal. The
library follows the rule libraries are supposed to follow — modules log,
the package handles nothing, whoever embeds it decides where that goes:

```
INFO:     127.0.0.1:53298 - "GET /search?q=chunks HTTP/1.1" 200 OK
INFO     full pass for self: uncommitted work, which a commit-to-commit diff cannot see; commit or stash it
INFO     indexed self: 122 files, 1827 chunks, 75 written, 43 deleted
INFO     index finished in 1.27s: 122 files, 1827 chunks
```

Before this, a successful request left no trace at all and a failed one
left ~69 lines of traceback that named neither the time nor the query.
The scheduler's failures are logged with their traceback as well as
recorded: the run log holds twenty entries in memory and loses them on
restart, which is the wrong place for the one failure nobody watched
happen.

**Nothing server-shaped leaks into the engine.** Every endpoint is a call
into the same library the CLI uses — an endpoint that could not be
written that way would mean the library was missing something, not that
the server should grow it. See
[ADR-10](docs/adr/adr-010-library-server-boundary.md).

**One search embeds its query once.** The store is asked one dataset at
a time, so a workspace of eight repositories used to run the same
sentence through the model eight times — measured at 52.4 ms against
25.6 ms now, exactly twice the work. The vector is remembered for one
query string, which is what a single search actually repeats; caching
across searches was measured separately and does not pay.

**Two processes, one index.** Measured rather than assumed: concurrent
writers lose nothing, a stale writer is still a correct writer, and
deduplication holds across processes. The one real hazard is reading —
a handle is pinned to the version it opened at, so a long-lived process
would answer from the corpus it started with and never fail doing it.
Searches refresh first, at about 4 ms against a 111 ms search. Indexing
runs one at a time and a second caller is told so (409) rather than
queued: two runs of the same repo do the same work twice.

Binding is `127.0.0.1` unless you say otherwise, and the server refuses
to start rather than open the index to whoever can reach the port: a
`token_env` naming an unset variable stops it, and so does a public
address with no token at all (`--insecure` if you meant it).

Two rules guard every request, and both apply to `/mcp` as well —
mounting an application, unlike routing one, brings its own empty stack,
and the tools behind it were once reachable with no token while
`/search` answered 401. The token must match, compared whole; and a
request that changes something must not come from another origin. A
caller with no `Origin` at all — curl, a webhook, this CLI — is let
through, because a browser always sends one on POST, so its absence is
the one case CSRF cannot come from.

The index directory is created `0700`. Everything a workspace knows is
in it, and on a shared machine it used to be readable by every other
account.

## What you asked, and what it could not answer

`wsindex stats` reads a log this machine keeps of its own searches — the
quality loop the roadmap asks for, because ten invented acceptance
queries decide less than however many the tool was actually asked.

```console
$ wsindex stats
14 search(es) since 2026-09-10, picked 3
waited: p50 2.3s  p95 2.4s (per command, model load included)
answered worst:
  0.190  'how do i bake sourdough bread'
  0.233  'kubernetes ingress controller tls'
asked most:
     4x  'how are chunks deduplicated'
```

The metric the plan wanted was a zero-result rate, and it reads 0%
forever: semantic search answers *something* whatever you ask it.
Measured on the real model, an answered query tops out at 0.53–0.61 and
one with no answer at 0.19–0.34 — so what is worth reading is the
**weakest** answers, ranked. No threshold, deliberately: that gap
belongs to this model and this corpus, and a ranking does not go stale.

The latency is labelled because it is honest and reads wrong without the
label: every `wsindex search` is a fresh process, so it is mostly the
model load. `poe bench` measures the search itself, at 8 ms. The gap
between those two numbers is the argument for `wsindex shell`.

A pick — opening a hit from the shell — is the only signal in this
project that somebody *found* what they wanted, which is why the shell
was worth building before this was.

**Strictly local, and tested rather than promised.** The log is a
SQLite file in the same `0700` directory as the index, and a test
asserts a search opens no sockets. Deliberately not in the link store:
links may be a shared Postgres since ADR-11, and one person's questions
do not belong in a team's database.

```toml
[stats]
enabled = false     # default: true
```

`wsindex stats --forget` empties it. A log you cannot switch off or
empty is a log you did not agree to.

## Reclaiming space

Deleting a chunk hides it immediately but does not free its bytes, so an
index that is edited often grows. `wsindex compact` is the pass that
shrinks it:

```console
$ wsindex compact
reclaimed 0.1 MB (1.7 MB -> 1.6 MB); 21 -> 2 versions
```

Manual on purpose, because it is the one command that throws history
away: until it runs the store can be rolled back to an earlier version,
and afterwards it cannot. Use `--keep-days` when something else may be
reading the same store.

How much it buys depends on how much was written. Chunks reach the store
in batches of a couple of thousand rather than one write per file, so a
cold index of 3458 chunks leaves 4 versions rather than 149 — and a fresh
index is already as compact and as fast to search as a compacted one
(2.4 ms per search against 2.3 ms). Compaction earns its keep on an index
that has been re-indexed many times, not on a new one.

## A chunk the model cannot read is a chunk nobody can find

The embedding model reads 256 tokens and stops. Anything past that is not
in the vector, so no query reaches it — and nothing said so, because the
index still reported the chunk as written.

Measured within one run: a line the model read is found by its own words
50% of the time at median depth 2; a line past the cut, 18% of the time
at median depth 33. Then counted exactly across two real repositories:
**74.1% and 32.5% of the indexed text sat past that cut.**

Chunks are therefore bounded in characters as well as in lines — about
900, calibrated against the 3.5-4.0 characters per token that real code
runs at. The parts that could be fixed were: the sliding window and the
gap pass between AST definitions held 38.0% and 5.2% of the text, and now
hold 13.5% and 0.1%. It costs 5% more chunks on ordinary code and 41% on
a repository full of minified vendored assets, where a single 8 000-character
line still cannot be split — a chunk's text has to stay a verbatim slice
of its line range, which is what makes a hit point at a real location.

What is left is inside definitions: a function longer than the model
reads stays one chunk, because halving it changes what a hit means.
That is 31.8% and 27.2% respectively, and it is the open question here
rather than an oversight. The acceptance criteria score the same 9/10
either way, which is worth knowing about both the fix and the measure:
ten hand-written queries do not see most of what these numbers describe.

## What a repository is made of

A different question from search, for a different reader: not "where is
X" but "what are the parts, and what is tangled".

```
$ uv run wsindex domains
67 files in 9 packages
  ingest 28  (root) 12  cli 7  config 5  server 5  connectors 4  store 3  embed 2  rank 1
agreement 57% (meaning recovers the layout; 11% would be chance)

filed away from their subject:
  src/wsindex/paths.py  [(root)]  0.696
      near src/wsindex/cli/composition.py, src/wsindex/config/__init__.py

coupled across packages (35), most-changed first:
   11 commits  similarity +0.748   src/wsindex/pipeline.py + src/wsindex/store/base.py
   10 commits  similarity +0.643   src/wsindex/pipeline.py + src/wsindex/server/api.py
```

Two signals, and the value is where they disagree. **Meaning** comes from
the vectors already in the index — a file's subject is the average of its
chunks. **Change** comes from the history already indexed: files that keep
moving in the same commit are coupled whether or not anything imports
anything.

Read `agreement` first: it says how much of the layout the meaning
recovers, against the baseline of saying nothing. Well above it and the
exceptions are worth reading; near it and nothing was found. On this
project the two signals agree — files that change together score 0.623 to
each other against 0.422 for all pairs — which is what makes the three
*strangers* interesting. Two of them are fair: `paths.py` is about
resolving config locations and sits at the root; `ingest/link_extract.py`
is filed by when it runs rather than by what it is about.

A **coupled pair** with high similarity is honest coupling — the two
files change together because they are about the same thing. **Low
similarity is the one to read**: something binds two files that are not
about the same subject.

The third signal the design called for — structural, from `READS_KEY`
links — is not used, and the reason is measured: the link extractor's
entire vocabulary is port numbers, which came to ten names across five
thousand files. It would contribute nothing until that grows. There is
also **no graph viewer**, deliberately: the useful output is three short
answers, and an interactive graph is what these tools become instead of
answering them.

## The same code in two places

```
$ uv run wsindex dupes
29543 of 86837 code chunks were long enough to fingerprint; 26288 duplicate pair(s) in 2230 place(s)

wholesale  4911 pair(s)  within Documentation/EHI_Export/docs/tables
wholesale  3539 pair(s)  within src/FHIR/R4/FHIRDomainResource
wholesale   547 pair(s)  Documentation/EHI_Export/docs/bower/admin-lte/plugins/jQueryUI
                         Documentation/EHI_Export/schemaspy/layout/bower/admin-lte/plugins/jQueryUI

  copied      9 pair(s)  contrib/forms/ped_fever
                         contrib/forms/ped_pain
    1.00  contrib/forms/ped_fever/view.php:1  <->  contrib/forms/ped_pain/view.php:1
```

Found by what the code is made of — runs of five identifiers, hashed and
compared — not by what it means. Embeddings were measured against this
and lost: real copies score 0.994, 0.840 and 0.717 as they are edited
more heavily, unrelated pairs sit at a median of 0.135, but the tails
overlap (unrelated p99 of 0.860 against adapted p90 of 0.864), so a
threshold that catches an adapted copy flags two or three percent of
everything else. The near-identical copies embeddings *do* separate are
separated exactly and far more cheaply by a fingerprint.

**The grouping is the feature.** Run on a twenty-year codebase, a plain
list of pairs is twelve thousand lines of generated FHIR classes and
jQuery checked in twice. Collapsed by the directories they connect, that
becomes a handful of `wholesale` lines — one decision somebody made once
— and underneath them the copy-paste a person can act on: a form's
`view.php` copied verbatim into the next form.

The threshold was placed by reading what sits on either side of it rather
than by picking a round number. Below 0.45 the report fills with
generated classes that are duplicated by construction; from 0.45 up it is
copies. `--min` moves it.

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

**Search is an exact scan, and that has a ceiling.** No approximate
index is built: every vector in the dataset is compared, which is the
right trade for a workspace — an ANN index costs build time, recall and
a tuning surface, and buys nothing until the corpus is large. Where
"large" starts, measured:

| Chunks  | Search  | Filtered |
| ------- | ------- | -------- |
| 2 000   | 2.2 ms  | 3.3 ms   |
| 20 000  | 4.3 ms  | 3.7 ms   |
| 100 000 | 12.5 ms | 7.5 ms   |
| 400 000 | 36.0 ms | 20.7 ms  |

Linear at the margin, so **a million chunks is about 90 ms** — roughly
60 000 files, and the point where a search stops feeling instant. Until
then the scan is cheaper than any index would be. Note the second
column: past about 20 000 chunks a *filtered* search is faster than an
unfiltered one, because the filter runs before the scan rather than
after it.

**Links live in SQL, not beside the vectors**, and that is measured
rather than assumed. Two things decide it, and neither is read speed —
the columnar store is in fact 6.6x faster to write and 3.5x smaller.
First, the drift report is an anti-join (`which reads_key has no declares`) and a vector store's filter language cannot express it at
all. Second, links are deleted per changed file on every run, which is
13x slower there and leaves 131 table versions behind for 120 files. A
vector is written once and read by similarity; a link is rewritten
constantly and read by exact key.

```toml
[links]
backend = "postgres"          # default: sqlite, which needs no service
dsn_env = "WSINDEX_LINKS_DSN" # the variable's name, never the string
```

SQLite is the default and the only backend that needs nothing — offline
machines, single-user workspaces and every test get it. Postgres is for
a **shared** index: `[store] uri = "s3://..."` makes the vectors common
to a team, and links are workspace data by the same argument (every
field of one is derived from content, so two machines indexing the same
commit produce identical links). Leaving them on one machine was an
asymmetry, and `wsindex status` says so when it sees one.

One database per shared index, exactly as there is one `[store] uri`:
the table is keyed by repo id and nothing else, so two workspaces
pointing at one DSN merge their links the same way two workspaces
pointing at one store uri merge their datasets.

There is one set of queries, not two implementations — a four-field
dialect carries everything SQLite and Postgres spell differently, so
parity is a property of the code rather than a discipline. The contract
suite runs against both; the Postgres half skips itself when no database
answers (`docker compose up -d postgres` makes it run, and the compose
service reads `WSINDEX_PG_*` rather than the generic `POSTGRES_*`, which
a shared `.env` had already claimed).

The design decisions are recorded in
[ADR-7](docs/adr/adr-007-post-mvp-storage.md) (LanceDB replacing the
earlier Tensorus + LocalStore pair) and
[ADR-11](docs/adr/adr-011-links-backend.md) (why links are not in it,
with the numbers).

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
- Quality attributes and the early decisions (ADR-1 to ADR-6) —
  [ARCH_en.md](ARCH_en.md). A map, deliberately short: what the tool does
  today is this file, which is measured and tested against the CLI.

_(Russian originals: `CONCEPT_ru.md`, `BRD_ru.md`, `ARCH_ru.md`.)_

## Toolchain

Package management **uv** · lint & format **ruff** · types **mypy (strict)** ·
tests **pytest** · hooks **pre-commit** · CI **GitHub Actions**.

## License

MIT — see [LICENSE](LICENSE).
