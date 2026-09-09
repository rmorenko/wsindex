# ADR-10. The library/server boundary, and who owns the index

- Status: Accepted
- Date: 2026-09-09
- Author: Roman Morenko (drafted with Claude)

Numbered 10, not 9, although the plan called this step "ADR-9": that
number went to [ADR-9](adr-009-links-as-entities.md) when the links spike
was written first.

## Context

wsindex is a local CLI. The plan (Этап 11) adds a second way to run it —
a shared server with an admin page that syncs and re-indexes on a
schedule — and the whole value of that addition depends on one property
being kept: **the engine stays a library, and the server is a thin layer
over the same `Pipeline`.** A server that grows its own indexing path
would be a second implementation of the thing hardest to keep correct.

That much was decided in advance. What was not known is the question the
step names explicitly: *who owns the index when a CLI run and a server
touch it at the same time?* A shared index on an `s3://` uri is exactly
what steps 17g/17d built, so "two processes, one store" is the intended
deployment, not an edge case.

Lance uses optimistic concurrency, which is a promise about conflicts,
not about everything else. So it was measured — `probes/step30`, real
processes, real stores, no mocking of what the question is about.

## What the probes measured

**Concurrent writers do not lose writes.** Two and then four processes
appending 20 chunks each at the same instant: every one reported success,
and a fresh handle afterwards held every row. No conflict was raised and
none had to be retried by us.

**A stale writer is still a correct writer.** A process whose handle
predates ten outside writes appends without conflict, and its own commit
brings it up to date. Deduplication survives the situation that looks
worst — re-adding chunks another process wrote returned "0 written",
because a commit refreshes the handle before the dedup lookup runs.

**Deletes are not limited to what a handle can see.** A stale process
deleted five rows written by another process after its snapshot: deletes
go out as a predicate and run against the current version. Two processes
deleting different rows at once both succeeded (1.07 s, exit code 0 for
both).

**The one real hazard is read staleness.** A `Table` is pinned to the
version it was opened at, and *reads never refresh it*. A reader polling
40 times over 0.8 s while another process committed saw its own opening
snapshot every time. For a CLI this is invisible — the process is younger
than the question. For a server it is a correctness bug that never
crashes: after any outside `index`, the server serves the old corpus
forever.

`checkout_latest()` cures it and costs **4 ms** against a 111 ms search
on a local path — 3.6%, and the alternative is answering with a corpus
that no longer exists.

## Decisions

**1. The boundary.** The library is everything that already exists:
`Pipeline`, `VectorStore`, `Config`, ingest, connectors. The server owns
HTTP, scheduling, authentication and the admin page, and it may not add
indexing behaviour of its own — an endpoint that cannot be expressed as a
call into the library is a signal that the library is missing something,
not that the server should grow it.

**2. Reads refresh; the contract says so.** `VectorStore.refresh()` joins
the contract rather than living in the server, because staleness is a
property of a long-lived process, not of HTTP: the MCP server of Этап 12
and any library caller have exactly the same problem. `Pipeline.search`
calls it, so no caller has to remember. 4 ms per search is the price of
never answering from a corpus that has been re-indexed underneath.

**3. Writers are safe, but one writer is the recommendation.** The
measurements above say concurrent indexing is *correct*; they do not say
it is *sensible*. Two processes indexing the same repo do the same work
twice and each keeps its own `state.json`, so neither can shorten the
other's next run. The deployment we document is therefore: the server
indexes, everyone else searches. This is a recommendation enforced by
documentation, not a lock — a lock would have to live in the store, would
have to be released after a crash, and would buy nothing that the
measured behaviour does not already give.

**4. Per-machine state stays per-machine.** `state.json` and `links.db`
live in `index_dir`, never on the shared uri (this was already true and
is now load-bearing). "The commit I last indexed" is a fact about one
host: two servers on different branches sharing one store must not
inherit each other's answer. The cost of getting it wrong is a silently
skipped re-index; the cost of keeping it per-machine is one redundant
full pass when a second host joins.

**5. FastAPI, as the plan hypothesised.** Confirmed rather than
re-litigated: it is typed, it produces the OpenAPI document for free
(which is what makes step 36's MCP adapter cheap), and Pydantic is
already the shape of everything the CLI passes around. It arrives in a
`server` extra, so a local install stays what it is today.

**6. Authentication is a token from the environment.** The same rule as
connectors (step 29a) and the S3 store (ADR-7): the config names a
variable, never a value. Absent variable means the server refuses to
start rather than starting open — a search index over private
repositories is not a thing to serve accidentally.

## Consequences

(+) The server is small enough to read in one sitting, and every
behaviour it exposes is one the CLI already had.
(+) `refresh()` closes a bug class that would have been found in
production, at a measured 3.6% of a search.
(−) The contract grows a method that a store with no snapshot semantics
would implement as a no-op. Acceptable: the ABC has one implementation,
and the alternative is a server reaching into `store.tbl`.
(−) "One writer" is a convention, and conventions get broken. The
measurements say the failure mode is wasted work rather than corruption,
which is why this is documented instead of enforced.
