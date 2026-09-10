# ADR-12. External processes are started from a small child, not from the engine

- **Status.** Accepted, 2026-09-10.
- **Supersedes** nothing. Constrains any future code that wants to run a
  program, which is why it is written down rather than left in one
  afternoon's commit messages.

## Context

Step 39 fixed a service level objective before measuring anything —
p95 search under 500 ms — and then failed it in one scenario: search
while a **full** re-index runs, p95 **1459 ms**, three times the budget.
The everyday path passed comfortably (an ordinary sync: 168 ms), so this
was the maintenance case, not the common one. It was still a real state:
a first index, or an index directory that lost its state file.

Three explanations were measured and ruled out before the real one:

- **Not the embedding batch size.** Splitting the indexer's `encode` call
  down to 32 texts changed nothing; the stall stayed at ~1.5 s.
- **Not torch's thread count.** With `torch.set_num_threads(1)` the stall
  was 1.88 s — if anything worse.
- **Not the GIL.** A monitor thread sampling continuously lost 53 ms at
  its worst while a search was blocked for 1.4 s.

What it is: the blame pass starts one `git blame` per indexed file, and
**starting a process from the process that holds the embedding model is
what costs.** The clearest single measurement is that the command does
not matter — 117 spawns of `git --version`, which does nothing at all,
block a search in the same process for the same 1.4 s. Eight threads
doing it buy no parallelism: 9.7 ms per spawn either way.

### Why

Chased down afterwards, because it decides whether the cure generalises.

The cost is in the **exec**, not the fork. A bare fork storm leaves a
working thread at 1.1x its idle speed; `fork` + `exec` puts it at 27x,
with or without pipes. After a fork the child holds a copy-on-write copy
of the parent's address space, and `exec` has to tear that down before it
can load the new binary — so what is paid for is the parent's *map*, not
its bytes:

| parent holds       | regions | per exec | worker |
| ------------------ | ------: | -------: | -----: |
| bare               |    1991 |  0.90 ms |  11.2x |
| 1 GB, one mapping  |    2002 |  1.01 ms |   8.1x |
| 1 GB, 16k mappings |   17996 |  2.82 ms |  34.7x |
| the model          |    2997 |  2.43 ms |  32.3x |

A gigabyte in a single mapping costs what nothing costs. The same
gigabyte in sixteen thousand mappings does not. The model is dearer than
its region count alone predicts, because its regions are file-backed
mappings of large dylibs rather than anonymous memory. Concurrent execs
serialise on the teardown, which is where the missing parallelism went.

Two other candidates died here too: **resident memory** (see the second
row) and **thread count** — sixteen idle threads cost 0.36 ms a fork
against a bare process's 0.30 — and so did **the allocator's lock**,
since during a fork storm a numpy thread holds its idle median exactly.

**This is a macOS problem.** On Linux CPython uses `vfork`, no copy of the
address space is made and there is nothing to tear down: sixteen thousand
mappings cost 0.37 ms an exec against a bare process's 0.49, and the
working thread stays at 2.6x rather than 34.7x. Measured in a container.

### Cures that were rejected, with their prices

- **`posix_spawn`.** Fastest of all — the storm drops from 1.46 s to
  0.28 s and the stall from 1477 ms to 112 ms, from the *same* process.
  CPython only takes that path with `close_fds=False`, and a probe showed
  the child then inherits seven of this process's descriptors (LanceDB's
  and SQLite's among them). Refused: this project does not trade a file
  descriptor boundary for latency.
- **`nice` on the subprocesses.** Does nothing at all: 2079 ms → 2143 ms.
  The scheduler is not what is being contended for.
- **Fewer blame workers.** Works — two workers bring p95 to 284 ms — and
  costs 48% of indexing throughput, penalising `wsindex index` for
  everybody to protect a case that only a server meets.

## Decision

1. **A batch of external processes is started from a small child, not
   from the process that holds the model.** `wsindex.ingest.blame` is
   that child for the blame pass: it runs the same eight-way pool, and
   the engine starts one process instead of one per file.

1. **The child imports nothing from this package**, and a test asserts
   it. `import wsindex.ingest` costs 30 ms of unrelated imports against a
   21 ms bare interpreter — enough to move the break-even from four files
   to nearer ten. A stray `from wsindex...` would fail no other test; it
   would quietly make the threshold wrong.

1. **One implementation, two callers.** How to invoke git arrives as an
   argument, so the parent passes `run_git` and keeps every guarantee it
   makes — the timeout, the read-only lock hint, the typed errors — while
   the child passes its own plain one. Neither the porcelain parser nor
   "how to call git safely" exists twice. This is ADR-11's rule about
   `_Dialect` applied to a process boundary instead of a SQL one.

1. **Small batches stay in-process**, and the threshold is derived rather
   than chosen: the child wins once `N * 9.7 > 21 + N * 9.7 / 8`, so
   `HELPER_FROM = 4`. The everyday sync changes one file and pays
   nothing.

1. **Every way the child can fail falls back to doing the work here.** No
   interpreter, a non-zero exit, nonsense on stdout, a hang, a frozen
   build with no source file: all of them return `None` and the caller
   blames in-process, where the real error arrives from `run_git` with
   its own type. An optimisation that can stop an index run is worse than
   no optimisation. There is a test for each.

1. **Kept unconditional, including on Linux**, where it buys nothing and
   costs one process start (21 ms) per batch. A platform branch would be
   a second path that only half the machines exercise, which is a worse
   trade than 21 ms against a pass that takes seconds.

1. **The rule is guarded, not remembered.** A test asserts that indexing
   twenty more files does not start twenty more processes *in the engine
   process* — a shape rather than a census, so it cannot go stale — and a
   second test switches the child off and asserts the count does grow, so
   the first is known not to be vacuous. It was: its first version built
   a pipeline with no `LinkStore`, which skips the blame pass entirely,
   and passed while measuring a run that never blamed anything.

## Consequences

- (+) The SLO is met in every scenario, without the budget moving: search
  during a full re-index went **1459 ms → 123 ms** p95.
- (+) A cold index got **11% faster** (8.18 s → 7.29 s). Those forks were
  never necessary work, so this is not a trade — `index-one-file` is
  unmoved at 0.32 s and no other benchmark scenario moved more than 5%.
- (+) The engine now starts **six** processes per index run — `log`,
  `ls-files`, `rev-parse` x3, `status`, about 15 ms of exec — instead of
  one per file. Measured, not assumed.
- (−) One more moving part, and a JSON protocol across a pipe. Paths from
  git may hold lone surrogates; they survive because the protocol is
  ASCII-escaped JSON, and there is a test saying so.
- (−) The mechanism is established on macOS only. The Linux figures come
  from a container and say the problem does not arise there; they do not
  come from the full product on Linux.
- **When to revisit.** If a future ingest step wants to run a program per
  file — a linter, a formatter, `ripgrep`, a language server — it belongs
  behind this same child, or behind one of its own. The guard test is
  what will say so.
