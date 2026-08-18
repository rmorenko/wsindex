# ADR-7. Post-MVP storage: LanceDB for vectors

- Status: Accepted — supersedes ADR-2 (Tensorus v1 as the index DB +
  LocalStore fallback, ARCH §8)
- Date: 2026-08-19
- Author: Roman Morenko (drafted with Claude)

## Context

ADR-2 bet on Tensorus as the primary index DB with LocalStore as an
offline fallback. Its technical premises are dead: the promised HNSW
index and `search-similar` never materialized (the server's
`/index/build` is broken upstream and search runs brute-force), the
server needs our own patches to run at all (the compose file carries
them), and it is ~33x slower than LocalStore on the acceptance
corpus. The MVP acceptance cross-check made the verdict measurable:
a 0.0000 score delta on 7/7 criteria proves Tensorus today behaves
as a remote, slower LocalStore — no unique capability is left. The
key question of this ADR — "is Tensorus the goal of the project or
an accident of its birth?" — was answered: an accident.

sqlite-vec was briefly chosen (2026-08-16) and rejected the same
day: LanceDB covers the same need (embedded vector search, zero
infrastructure) and additionally the scale thresholds for which it
was already candidate #1 — an optional ANN index, out-of-RAM
indexes, and S3-compatible object storage.

A probe (step 17b, lancedb 0.37.1, MinIO on localhost) established
the facts this decision rests on:

- insert of 1000 x dim-384 vectors: one batch — 3.6 ms local /
  12.9 ms S3; row-by-row — 1175 ms (328x) / 12626 ms (981x);
- search over those 1000 vectors: compact table — 1.15 ms local /
  3.3 ms S3; after 1000 row-by-row inserts — 24.3 ms (21x) /
  364.9 ms (110x): fragmentation punishes reads, not just writes;
- search without `create_index()` is exact brute force — results
  are deterministic and cross-checkable against another backend;
- there is no primary key: duplicate ids insert silently; an
  explicit Arrow schema enforces types and dimension but coerces
  values silently (int id became "1");
- `connect()` is eager (it lists table manifests immediately), so
  misconfiguration fails fast, before any indexing work;
- credentials work from standard `AWS_*` environment variables
  without any code-level configuration.

## Decision

1. **LanceDB (embedded) is the only vector store.** The storage
   location is a uri in `wsindex.toml` (`[store] uri` — a local path
   or `s3://bucket/prefix`). Credentials, endpoint and `allow_http`
   come exclusively from the environment (standard `AWS_*`
   variables); a local-path uri requires no environment at all.
1. **One table per workspace with a `repo` column**, not
   table-per-dataset. ADR-5's driving premise — Tensorus
   property-search could not filter by metadata — died with
   Tensorus; LanceDB prefilters (`where("repo = ...")`) before an
   exact KNN. This buys: a global top-k in a single query, uniform
   repo/lang/kind filtering, repo removal as
   `delete("repo = '...'")`, and cross-repo analytics over one
   table. The `VectorStore` contract keeps `dataset_name`; the
   store maps it onto the `repo` column, so the pipeline and the
   migration cross-check stay untouched. The 17v test suite must
   confirm prefilter parity (KNN over a repo subset equals KNN over
   an equivalent standalone table) and deterministic tie-breaking.
1. **Chunk metadata are real columns** (id, repo, path, lang, kind,
   symbol, node_type, start_line, end_line, text), never a JSON
   blob: prefiltering (step 19g) is only possible over columns.
1. **SQLite stays relational-only** (stage-10 links, stage-8 git
   state); vectors never go there.
1. **Tensorus is removed** (step 17e) only after the migration is
   green: `make acceptance` 7/7 on LanceDB with a 0.0000 cross-check
   delta is its exit exam. Until then it stays as the independent
   referee of the migration.
1. **Any future server is our own thin layer** over this same
   library (stage 11); the shared index lives on the S3 uri.

## Consequences

What we now owe:

- `add_chunks` writes one batch per call — row-by-row insertion is
  328–981x slower and fragments the table;
- incremental indexing (stage 8) will accumulate fragments over
  many small commits; compaction (`optimize`) on a threshold or
  schedule is required — fragmented search is 21–110x slower;
- deduplication is ours: known ids are queried before embedding
  (no primary key exists, and embedding is the expensive stage);
- `score = 1 - _distance` is cosine-specific (l2 returns a squared
  distance); the score formula is bound to the metric and the
  metric guard stays in our code — LanceDB binds metric at query
  time, not at table creation;
- the schema is always declared explicitly, and input typing is
  disciplined at the boundary — the schema will not reject a wrong
  type, it may silently coerce it.

What we gain:

- exact KNN by default: deterministic results, comparable across
  backends — the property the migration cross-check stands on;
- fail-fast configuration: a wrong endpoint or a missing
  `allow_http` fails at store construction, not mid-indexing;
- S3-compatible storage at ~3x local search cost (localhost MinIO)
  — the foundation for the stage-11 shared index, with optimistic
  concurrency (`_transactions/`) supporting multiple writers.

Revisit thresholds:

- an ANN index (with its recall knob) is bought when the corpus
  approaches ~500k chunks or brute-force p95 misses targets — the
  step-37 bench suite is the watchdog;
- the compaction schedule is decided in stage 8, when increments
  are real;
- the LanceDB API is young (`table_names()` deprecated in favour of
  `list_tables()` within one minor line) — the version is pinned
  and renames are expected.
