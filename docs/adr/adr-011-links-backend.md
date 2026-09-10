# ADR-11. Links keep their own SQL store, and it may be shared

- **Status.** Accepted, 2026-09-10.
- **Supersedes** nothing; **refines** ADR-9, which chose SQLite and gave
  a reason weaker than the real one.

## Context

ADR-9 put links in SQLite and justified it in one sentence: "every query
against them is equality, not similarity". Challenged directly — *why
not LanceDB, it is the same kind of information* — that sentence does
not survive contact with measurements. It is also, read as a performance
claim, partly wrong.

Measured, 200 000 links, same data both ways:

|                              | SQLite      | LanceDB               |
| ---------------------------- | ----------- | --------------------- |
| write the lot                | 506 ms      | **76 ms**             |
| on disk                      | 17 MB       | **4.8 MB**            |
| `refs`: by_name              | **0.02 ms** | 4.08 ms               |
| `why`: out_of, 50 ids        | **0.04 ms** | 5.18 ms               |
| drift: `dangling`            | **69 ms**   | `ValueError`          |
| forget one run's files (120) | **49 ms**   | 632 ms, +131 versions |

So the columnar store wins the two things it is built for — bulk append
and size — and the read gap (100-200x) is invisible at the scale this
project actually reaches: the workspace it was measured on holds ~3 000
links, where 4 ms is nothing.

The two that decide it are elsewhere.

**`dangling` is an anti-join and LanceDB cannot say it.** `.where()`
takes a filter expression, not a query; `NOT EXISTS (SELECT ...)` is
rejected outright. The workaround is to pull both sides into Python and
subtract sets — comparable in speed (64 ms against 68 at 200k), but it
moves the logic out of the store, and a store that cannot answer its own
question is a file format.

**Links are deleted constantly.** Every changed file forgets its links
on every incremental run. That is 13x slower in the columnar store, and
it leaves 131 versions behind for 120 files — links would need their own
`compact`, which is the exact cost ADR-7 and review 4 already paid once
for the vectors.

The asymmetry is not in the data. It is in the access pattern: a vector
is written once per chunk and read by similarity; a link is written and
deleted per file on every run and read by exact key.

A second question came with the first: **`[store] uri = "s3://..."`
makes the vectors common to a team, and the links stay on one machine.**
The code called them "local, per-machine notes about this host, like
`state.json`" — and that is a category error. `state.json` genuinely is
per-machine: two hosts sit on different branches. Links are not. Every
field of one is derived from content, so two machines indexing the same
commit produce identical links. A shared index with unshared links is
half a feature: each machine repeats the blame pass, and two servers
over one store answer `refs` differently.

## Decision

1. **Links stay in SQL.** Not in the vector store, for the two reasons
   above, recorded with their numbers so the next reader does not have
   to re-measure to disagree.

1. **SQLite remains the default and the only backend that needs
   nothing.** Offline machines, single-user workspaces and the whole
   test suite get it without a line of configuration.

1. **Postgres is available for a shared index**, chosen by
   `[links] backend = "postgres"` with `dsn_env` naming the variable
   that holds the connection string — the name, never the string, the
   rule `token_env` and the S3 store already keep.

1. **One set of queries, not two implementations.** ADR-2 paired
   Tensorus with LocalStore and spent its life keeping them at parity,
   because their semantics differed. SQLite and Postgres are both SQL
   with the same semantics, so the queries are written once and a
   four-field `_Dialect` carries the differences: placeholder style, how
   each spells "ignore a duplicate", and how to ask which columns exist.
   Parity is then a property of the code rather than a discipline.

1. **The contract suite runs against both**, parametrized, with the
   Postgres half skipping itself when no database answers — the pattern
   the tree-sitter tests already use for a missing grammar. `docker compose up -d postgres` is what makes them run.

1. **A shared index with local links says so.** `wsindex status` prints
   a note when `[store] uri` is `s3://` and links are SQLite. Review 6's
   rule: a partial answer must not look whole.

1. **One database per shared index.** The table is keyed by repo id and
   nothing else, so the DSN is the isolation boundary exactly as
   `[store] uri` is for vectors. Found by pointing a probe at the test
   database and watching `refs` answer with rows the suite had left
   there.

## Consequences

- (+) The shared-index scenario that `s3://` and `wsindex serve` exist
  for is whole for the first time.
- (+) The reason links are not in LanceDB is now written down with
  numbers, and has a threshold: if `dangling` stops being needed and
  deletes become rare, the decision is worth recomputing.
- (−) A second backend to keep working, and psycopg as an optional
  extra. Mitigated by (4) and (5) rather than by promises.
- (−) Postgres tests need a service, so CI runs them only where one is
  up. The SQLite half — the default everybody gets — always runs.
- (−) `state.json` stays per-machine and always will; only links moved.
  Anyone reading the composition root will find the two treated
  differently, which is now the point rather than an oversight.
