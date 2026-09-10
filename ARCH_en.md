# WSIndex — Architecture

> **A map, not a specification.** This file holds the two things that do
> not belong anywhere else: the quality attributes that shape every
> decision, and the early ADRs (1–6) that no separate file records.
> Everything a reader needs about *what the system does today* is in the
> README, which is kept current because it is measured and, since
> review 8, tested against the CLI.
>
> It used to be 604 lines describing Tensorus v1 as the index database.
> Tensorus was replaced by ADR-7 in August 2026 and the code removed on
> 2026-08-21, and by the tenth review the document was also missing
> connectors, snapshots, per-repo markup, links, the server, MCP, the
> shell, plugins and the config schema — roughly half of what the
> project does. Rewriting it in full would have created a second source
> of truth to fall behind again. This is the other choice: keep what is
> durable, point at what is current.

## Where to look for what

| Question                                      | Where                                               |
| --------------------------------------------- | --------------------------------------------------- |
| What does it do, and how well?                | [README.md](README.md) — with measurements          |
| Why is the storage like that?                 | `docs/adr/adr-007-post-mvp-storage.md`              |
| Where do the config and index live?           | `docs/adr/adr-008-path-resolution.md`               |
| What are `refs` and `why` built on?           | `docs/adr/adr-009-links-as-entities.md`             |
| Why are links not in the vector store?        | `docs/adr/adr-011-links-backend.md`                 |
| Where is the line between library and server? | `docs/adr/adr-010-library-server-boundary.md`       |
| What can `wsindex.toml` say?                  | `wsindex.schema.json` (generated) and the README    |
| Why is this line of code like that?           | the docstring above it — 45% of `src/` is reasoning |

## Goal

**WSIndex** indexes a developer's workspace — several git repositories at
once — and answers natural-language questions with exact `file:line`
locations. Code and configs are chunked by their syntax trees,
documentation by headers; every chunk is embedded and searched by
meaning.

Everything runs locally: source never leaves the machine, embeddings are
computed in-process by a local model, and the index is a file next to the
config. Measured, not asserted — a warm search opens zero sockets
(review 6).

## Quality attributes

The project is educational first, so the priorities are ordered
differently than in typical production. This is deliberate and it runs
through every decision below.

**Understandability > Modifiability > Portability/Offline > Performance.**

| Priority | Attribute             | What it means in practice                                         | How the architecture holds it                                                                         |
| -------- | --------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 1        | Understandability     | A reader sees the data flow without a debugger                    | One linear pipeline `repos -> walk -> chunk -> embed -> store`; small modules; no hidden asynchrony   |
| 2        | Modifiability         | Swapping the model, backend or chunker without rewriting the core | `Embedder`, `Chunker`, `VectorStore` interfaces; entry-point plugins for languages and connectors     |
| 3        | Portability / offline | Runs and is tested without external services                      | Local models tried from cache before the network; a fake embedder for tests; TOML config; no cloud    |
| 4        | Performance           | Adequate on the corpora people actually have                      | Exact scan to ~1M chunks (threshold measured in the README); optimizations only where measured to pay |

**Drivers:**

- **Multi-repo is the primary scenario.** One question, answers from
  several repositories — hence a dataset per repository and merging in
  the pipeline.
- **Polyglot corpus.** Hence tree-sitter with many grammars, and the
  split between `ast_chunker` and `text_chunker`.
- **Locality and privacy.** Code never leaves the machine.
- **Deterministic re-indexing.** A chunk id is a hash of content and
  path, so re-indexing is reproducible and deduplication is free.
  Verified in review 9: two fresh indexes give identical ids, and
  identical search order and scores.
- **Extensibility by phase.** Structural filters, re-rank and a graph
  had to fit without breaking the core — and did.

## Components, in one line each

- **`cli/`** — the commands. Adapters only: each is a call into the
  library and a rendering of the answer (ADR-10).
- **`pipeline.py`** — the engine: plan a run against git state, read and
  chunk, merge a search, and answer questions about a finished index.
- **`ingest/`** — what to index and how to cut it: the walker's policy,
  the chunkers, git state, commits and blame, link extraction.
- **`store/`** — the `VectorStore` contract and its LanceDB
  implementation. The pipeline sees only the contract.
- **`embed/`** — text in, vectors out. Loads its model lazily, from the
  local cache before the network.
- **`links.py`** — SQLite, the inverted index behind `refs` and `why`
  (ADR-9).
- **`connectors/`**, **`snapshot.py`** — documents from outside the
  repositories, materialized into a git snapshot repo.
- **`server/`**, **`mcp_server.py`**, **`shell.py`**, **`ui.py`** — three
  more adapters over the same library, and where output is decided.
- **`config/`**, **`paths.py`** — what a workspace says about itself, and
  where that is found (ADR-8).

## Architectural decisions (ADR-1 to ADR-6)

### ADR-1. tree-sitter for AST chunking

- **Context.** The corpus is polyglot (Python, Rust, TypeScript, as well as TOML/YAML/JSON/Dockerfile configs). Line/regex splitting loses function boundaries and config structure, degrading relevance and snippets.
- **Decision.** Use tree-sitter with a set of grammars (python, rust, typescript, toml, yaml, json, dockerfile, markdown). For code, split by functions/classes/methods/blocks; for configs, by structural nodes (tables/keys/sections/stages). Fill `symbol`, `node_type`, `start_line`, `end_line` from the AST.
- **Consequences.** (+) Meaningful chunk boundaries, ready ground for structural filters in Phase E2. (+) A single mechanism for many languages. (−) Dependence on the grammars and their build; (−) an unsupported language needs a fallback to text splitting. Documentation (Markdown/txt/rst) is not split via AST — it goes to `text_chunker`.

### ADR-2. Tensorus v1 as the index DB + LocalStore fallback

- **Context.** A self-hosted vector store with k-NN is needed. Tensorus v1 is the project's target DB (including as an educational demonstration of Tensorus itself). But requiring a running Rust server for every run is a barrier for learning and tests.
- **Decision.** The primary backend is `TensorusStore` (REST to v1, HNSW, cosine). Plus `LocalStore` (numpy brute-force cosine, local files) as an offline fallback. The backend is chosen in the config.
- **Consequences.** (+) The project runs and is tested without external services; (+) a real production path through Tensorus. (−) Two implementations must be kept at semantic parity (both cosine, the same contract, the unified `Hit` type). (−) LocalStore does not scale, but that is not its job.
- **Superseded by ADR-7** (`docs/adr/adr-007-post-mvp-storage.md`, 2026-08-19): the premises above died — see ADR-7's Context.

### ADR-3. Single-vector in the MVP, tensor re-rank deferred

- **Context.** "Tensor" retrieval (late interaction / MaxSim) is the conceptual core of the project's growth, but it is harder to implement and explain.
- **Decision.** In the MVP — **single-vector**: one vector per chunk, search via `search/similar`. Tensor re-rank / late interaction (MaxSim) is not in the MVP but in the roadmap (Phase E3).
- **Consequences.** (+) A simple, understandable, and fast baseline; (+) directly compatible with Tensorus's HNSW. (−) A lower relevance ceiling than late interaction. (+) The data model and store are already ready to accept re-rank on top of top-k without breaking the core.

### ADR-4. The VectorStore abstraction

- **Context.** At least two stores are needed (Tensorus and local), and in the future — a graph and possibly other DBs. Modifiability is the priority.
- **Decision.** Introduce the `VectorStore` interface (`create`, `upsert`, `search(dataset, vector, k) -> List[Hit]`) in `store/base`. Implementations: `TensorusStore`, `LocalStore`. The `pipeline` depends only on the interface and the `Hit` type, not on the backends' native responses.
- **Consequences.** (+) Switching the backend is a config matter; (+) tests run on LocalStore; (+) an extension point for new stores. (−) A small abstraction overhead — justified by the modifiability priority. `Embedder` and `Chunker` are introduced symmetrically.

### ADR-5. Dataset-per-repository for isolation

- **Context.** Tensorus's property-search does not filter by arbitrary `metadata`, yet isolation and deletion by repository are mandatory (multi-repo is the primary scenario).
- **Decision.** One Tensorus dataset per repository, `repo_id` = the dataset name. Multi-repo search = selecting a set of datasets and merging on the client; fine-grained filtering (`lang`, `kind`) — with a client-side post-filter.
- **Consequences.** (+) Natural isolation, simple deletion/re-indexing of a repo; (+) works around the property-search limitation. (−) Merging and final ranking of results fall on the client (`pipeline`), but with a single cosine metric the scores are comparable.
- **Canonicity.** This is the only source of the rationale for the "dataset-per-repo" scheme; §6.2 and §7.4 refer here.

### ADR-6. TOML config and Typer CLI

- **Context.** A readable workspace config and a convenient, self-documenting CLI are needed. The priorities are understandability and a low barrier to entry.
- **Decision.** The workspace config in **TOML** (repositories, backend, model, dim, Tensorus URL, chunking parameters). A CLI on **Typer** (Click under the hood): `init`, `add-repo`, `index`, `search`, `status`.
- **Consequences.** (+) TOML is familiar to the Python ecosystem and to the corpus itself (`pyproject.toml`, Rust configs); (+) Typer provides typed arguments and auto-help almost for free. (−) TOML is less flexible for deeply nested structures — sufficient for the MVP.

### ADR-7 onwards

From ADR-7 each decision is its own file under `docs/adr/`, which is
where they are read:

- **ADR-7** — post-MVP storage: LanceDB, one table with a `dataset`
  column. Supersedes ADR-2.
- **ADR-8** — where the config and the index live, and how they are found.
- **ADR-9** — links as entities: what `refs` and `why` are built on.
- **ADR-10** — the library/server boundary, and the one-writer rule.
- **ADR-11** — why links keep their own SQL store, and when it may be
  shared. Refines ADR-9 with the measurements that decide it.
