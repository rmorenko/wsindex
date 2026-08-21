# WSIndex

[![CI](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml/badge.svg)](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Frmorenko%2Fwsindex%2Fbadges%2Fcoverage.json)](https://github.com/rmorenko/wsindex/actions/workflows/ci.yml)

**WSIndex** is a CLI that semantically indexes a developer workspace — multiple
repositories at once — and answers natural-language questions with exact
`file:line` locations. Code (Python, Rust, TypeScript, Java) and configs
(TOML, YAML, JSON, Dockerfile) are chunked by their syntax trees, docs by
headers; every chunk is embedded and searched by meaning, not by keywords.

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

Re-running `index` embeds nothing: chunks are deduplicated by a
deterministic id before the (expensive) embedding step.

## Backends

**local** (default) — an embedded [LanceDB](https://github.com/lancedb/lancedb)
index at the `[store] uri` from `wsindex.toml` (default: `.wsindex/` next
to the config). Fully offline once the model is downloaded; embedding runs
in-process. The uri may also point at S3-compatible storage
(`s3://bucket/prefix`) — endpoint and credentials come from the standard
`AWS_*` environment variables, never from the config file.

The storage cost is modest because embedding dominates: on the acceptance
corpus (3458 chunks, real model) indexing takes 7.8s on a local path vs
10.1s over MinIO, and the 7 acceptance searches take 0.1s vs 0.3s. Both
storages return bit-identical results (cross-check delta 0.0000). A MinIO
for local experiments ships in the compose file
(`docker compose up -d minio minio-init` — the init service creates the
`wsindex` bucket).

**tensorus** — embedding and search happen server-side on a
[Tensorus](https://github.com/tensorus/tensorus) instance. The same model
name travels in the config, so both backends produce identical scores.

```bash
# .env must define POSTGRES_PASSWORD and TENSORUS_API_KEYS
docker compose up -d app db
export TENSORUS_API_KEY=<one key from TENSORUS_API_KEYS>
uv run wsindex init myws --backend tensorus   # server expected at localhost:8000
```

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
make check                 # ruff + mypy --strict + pytest (fast suite)
uv run pytest -m slow      # real-model smoke test (network, model download)
uv run pytest -m live      # integration against a running tensorus server
```

| Target           | Description                       |
| ---------------- | --------------------------------- |
| `make install`   | Sync dependencies (`uv sync`)     |
| `make lint`      | Lint with ruff                    |
| `make fmt`       | Format with ruff                  |
| `make typecheck` | Type-check with mypy              |
| `make test`      | Run pytest                        |
| `make check`     | Lint + type-check + test          |
| `make hooks`     | Run all pre-commit hooks          |
| `make clean`     | Remove caches and build artifacts |

## Known limitations

- Javadoc and JSDoc comments land in plain gap chunks instead of sticking
  to the definition below them (Rust `///` docs do attach).
- The tensorus backend embeds one chunk per HTTP request: indexing is
  ~30x slower than local (measured: 3458 chunks in 252s vs 7.5s), and
  each search takes seconds (the server embeds the query per request).
- The upstream tensorus `/index/build` endpoint is broken, so server-side
  search runs brute-force.
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
