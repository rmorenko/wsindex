# ADR-9. Links as entities: which edges, from where, and what "dangling" means

- Status: Accepted
- Date: 2026-09-09
- Author: Roman Morenko (drafted with Claude)

Numbered 9, not 8, although the plan called this step "spike + ADR-8":
[ADR-8](adr-008-path-resolution.md) was written in the meantime and holds
that number.

## Context

Этап 10 already discarded "the graph" as an abstraction: centrality and
PageRank give search nothing, and a general graph invites building
machinery before knowing which edges pay. What survived is narrower and
testable — a few concrete edge kinds, and the position that **a link is
an entity, not chunk metadata**, because the query inverts ("who reads
X"), because a link outlives the chunk it was found in, and because
many-to-many relations carry attributes of their own.

What was *not* known is whether those edges can be had cheaply. The
chunkers already parse every file; if edges fall out of the trees they
already build, links cost an extra pass over data in hand. If they need
their own analysis, the price changes completely.

So this step was a spike, and it was given two exams — both against this
repository's own history, so neither could be answered by wishful
thinking:

1. **Would code → config links have caught the 8080/8000 bug?** At
   `918ecf8`, `config.py` defaulted to `http://localhost:8080` while
   `docker-compose.yml` published the service on `8000`. Nothing failed;
   the mismatch was found by hand, later.
1. **Would BLAMED_BY answer "why is dedup before embedding"?** A design
   decision whose reasoning lives in a commit message and nowhere else.

The spike is `probes/step26/`.

## What the spike measured

**Exam 1 — passed, precisely.** At the buggy revision the detector
reports one dangling link and it is the right one:

```
compose publishes ports: ['5050', '5432', '8000', '8501']
DANGLING  PORT=8080  config.py:38: base_url="http://localhost:8080",
>>> 1 dangling of 1
```

At `7235427`, the commit that fixed it, the same rule reports zero. The
bug was findable, mechanically, from trees the chunker was already
building.

**Exam 2 — passed.** Blame on the dedup line resolves to `3964bb7`,
whose message reads *"dedup before embedding (batch-internal included),
one Lance commit"*. The reasoning is recoverable; it just is not
anywhere a reader would look.

**The cost, per edge kind, on the repository as it stands:**

| Edge                       | Refs | Dangling | False alarms  |
| -------------------------- | ---- | -------- | ------------- |
| code → config, by value    | 1    | 0        | **0%**        |
| code → config, by env name | 13   | 11       | **85%**       |
| code → code, by name       | 130  | —        | 11% ambiguous |

Code → code resolves half the call names it sees (130 of 258) and 11% of
those are ambiguous — a name defined more than once (`spans`, `search`,
`compact`). Noisy, as expected, but not hopeless.

Two things the spike taught that the plan had not anticipated:

**"Dangling" is meaningless without a declared scope.** All eleven
false alarms are real environment variables that are simply not in
`docker-compose.yml`: `AWS_*` come from the ambient environment,
`WSINDEX_*` are the tool's own knobs. Reading a variable with a default
is not a broken link, it is an optional input. The bug shape that
matters is narrower: *a value that claims to point at something this
workspace itself publishes.* An 85% false-alarm rate is not a tuning
problem, it is the wrong question.

**A spike measures the extractor as much as the idea.** The first run
reported 21 dangling links of 24. Half of them were a bug in the probe:
it took every string argument of `os.environ.get`, so the *default value*
became a second "env reference" — `os.environ.get("X", "http://...")`
produced a url masquerading as a variable name. Fixing that halved the
noise before any judgement about the idea was possible.

## Decision

1. **Links are entities in their own store, keyed by chunk id.**
   `Link(src_chunk_id, kind, name, line, dst_chunk_id | None)`, in
   SQLite as ADR-7 anticipated. They do not belong in the vector store:
   every query against them is equality, not similarity, and the two
   have different lifetimes — re-chunking a file replaces its chunks and
   must not silently orphan what pointed at them.

   *(Refined by **ADR-11**, 2026-09-10. That sentence was challenged and
   is weaker than the real reason: measured, the vector store is faster
   to write and smaller on disk. What decides it is that the drift
   report is an anti-join it cannot express, and that links are deleted
   on every run at 13x the cost. ADR-11 carries the numbers, and adds
   Postgres for a shared index.)*

1. **Edges come from the trees the chunkers already build.** Extraction
   is a second visitor over the same parse, not a second parse. This is
   what the spike established and it is what keeps links affordable.

1. **Ship code → config first, by value, and only by value.** It scored
   zero false alarms and it caught the real bug. Matching a literal in
   code against a value a config file *publishes* is the whole rule.

1. **"Dangling" requires a declaration source, named explicitly.** A
   link is dangling when it names something a config in this workspace
   was supposed to declare and does not — not when a value is absent
   from some file we happened to look at. Without that scope the
   detector cries 85% of the time and gets turned off.

1. **Code → config by env name is deferred, not adopted.** Same
   mechanism, but it needs the scope from (4) to be worth anything; it
   ships when there is somewhere to declare that a variable is the
   workspace's own.

1. **Code → code by name ships second, and is presented as a guess.**
   11% ambiguity is usable for "who calls this" if the answer is a
   ranked list rather than a claim. It must never be the input to
   anything that looks like a fact.

   *(Shipped 2026-09-12, and the ambiguity problem is sidestepped
   rather than solved. `DEFINES`/`MENTIONS` claims only that a name
   occurs in a place — which is never a guess — instead of claiming
   which definition a use refers to, which is what the 11% was about.
   A call graph is still deferred.*

   *Two things were measured before writing it. The pair's value: 1 090
   names in caddyserver have a definition in one file and a mention in
   another, against **three** names across five workspaces for the
   config pair, because `links_for` had never read `chunk.symbol` and so
   a vocabulary of nine was the ceiling of what four regexes caught. The
   pair's cost: the extractor sees one file and cannot know the symbol
   table, so the filter has to be decidable from a token alone. Storing
   every token of four characters or more kept all of the value and cost
   186 100 edges — nineteen per chunk, 1.9M rows on a 100k-chunk
   workspace. Requiring an internal word boundary keeps 82% for 31 839,
   and needs no per-language stop-list because every keyword in all
   sixteen grammars is a single lowercase word. The second was chosen.*

   *The bill, same repo indexed twice with everything else held equal:
   **0.3s of 24.4** — embedding dominates and this is lost in it — and
   **21.8 MB of `links.db` against 19.1 MB of vectors**. That second
   number is the one to watch. The link store is now the larger half of
   the index, at 440 bytes a row for a row that holds a name, a kind and
   a 64-character chunk id repeated across three indexes. Nothing here
   introduced that width; this change multiplied what it costs by three,
   which is what makes it worth writing down.)*

1. **BLAMED_BY is worth building** — exam 2 answered a real design
   question from data already on disk. It depends on commits being a
   corpus (step 27), so it follows that step rather than leading it.

## Consequences

What we owe:

- A declaration source for (4). Until then the dangling detector has
  exactly one usable rule — the value match — and that is the one that
  ships.

- ~~Link lifetimes have to be handled at the same place chunk lifetimes
  are.~~ **Paid.** `LinkStore.delete_by_source` is called from
  `Pipeline._apply` with the same `stale` set that feeds
  `delete_chunks`, and an end-to-end test asserts every link's source is
  still a live chunk after a rewrite.

  Building it changed one thing in this ADR. Resolution was going to
  happen at write time, but that is impossible in an incremental
  indexer: re-indexing one changed source file gives no access to config
  files nobody touched. So **both sides are links** — code emits
  `READS_KEY`, configuration emits `DECLARES` — and "dangling" is a
  query over whatever is stored right now. That turned out to be the
  better design for its own sake: delete the config publishing a port
  and its `DECLARES` links die with its chunks, so the code reading it
  becomes dangling on the next query, with nothing to update by hand.
  The lifetime rule and the drift feature are one mechanism.

  Also revised: the extractor reads chunk *text*, not the parse tree.
  The value rule turned out not to need a parse — a `host:port` in a
  string is recognisable in text, and the chunks are already in hand, so
  this is cheaper than "a second visitor over the same parse", not more
  expensive. It matches ports in comments too, which for a drift
  detector is a feature. A rule that genuinely needs the tree (calls,
  imports) is when the tree gets threaded through.

- Ambiguous code → code edges must be labelled ambiguous wherever they
  surface. `wsindex refs` (step 28) showing a 50/50 guess as a fact
  would be worse than showing nothing.

What we gain:

- The 8080/8000 class of bug becomes a report instead of an afternoon.
  Dangling links are the feature, not an error state: they are the only
  automatic evidence that code and configuration have drifted apart.
- "Why is this like this" gets an answer path — definition, blame,
  commit message — which is what `wsindex why` (step 28) is for.
- Links stay cheap enough to keep: one visitor over parses that already
  happened, in a store that costs a file.

Not addressed here (deliberately deferred):

- **Cross-repository links.** A workspace holds several repos and a
  service in one may read a config in another. The model allows it —
  chunk ids are workspace-unique — but nothing resolves it yet.
- **Ranking by links.** Explicitly out: this is the "graph gives search
  nothing" conclusion Этап 10 started from, and finding edges cheap
  does not reopen it.
