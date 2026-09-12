# Field trial — twenty real workspaces

wsindex had never been run on a codebase that was not its own. This is
what happened when it was: twenty organizations from GitHub, 105
repositories, 4.4 GB of working trees and full history, and 204 questions
written down **before** wsindex was allowed to run.

It is not a demonstration. Three workspaces are in because wsindex has no
grammar for their language, and every search result has a `ripgrep`
control, because a question `grep` already answers is not a question that
needs an index.

The corpus was selected from the GitHub API rather than from memory:
506 organizations collected by searching twelve languages across two
star bands, 394 with three or more live repositories, twenty chosen
against size, age, repo count and language coverage. The harness, the
frozen questions and every raw measurement live under `probes/`, which
this repository does not track — review artifacts stay out of git. What
is reproducible from here is the method below.

## How the questions were made honest

A tester opened each workspace with `Read`, `Glob`, `Grep` and `git log`
only — **never wsindex** — and wrote twelve questions a developer joining
that codebase would ask, together with the true answer (`file:line`) found
by reading. Those files were written to disk and not edited afterwards.

Twelve per workspace, in three classes:

- **Literal** (4) — the words of the question are in the code.
- **Descriptive** (6) — the behaviour described in words that do *not*
  appear in the answer file. Mechanically checked: every word of four or
  more letters, minus function words, must be absent from the whole file
  **as a substring**. Roughly forty candidate questions were discarded
  for failing this.
- **Cross-repo** (2) — the answer is in a different repository from the
  one a person would open first.

The ordering is the whole point. It is very easy to look at what an index
returned and decide that was the question; every wrong conclusion in this
project's history came from grading an answer against a question chosen
after seeing it.

## Does it index?

|                                            | Workspaces |
| ------------------------------------------ | ---------: |
| Indexed properly (94–107% of source files) |     **14** |
| Indexed almost nothing, silently           |      **3** |
| Crashed outright                           |      **3** |

Where it works, it works well and it is fast. Throughput held at roughly
700–900 chunks per second across two orders of magnitude of corpus size.
The largest workspace that finished — dbeaver, 11 786 files — took 258
seconds and 349 MB for 125 353 chunks.

Re-indexing is the quiet triumph. **Re-running an unchanged workspace
costs 0.35–0.89 seconds and does not grow with corpus size**: dbeaver,
with 125 353 chunks, comes back in 0.68 s. That is the claim the README
makes and it survives contact with strangers' repositories.

One incremental number does not survive. The README's **0.24 s for a
single changed file** was not reached on any of the seventeen workspaces;
the range measured was **2.53–4.16 s**, including on an idle machine. The
difference from the unchanged case is the embedding model, which has to
be loaded before one chunk can be embedded.

### Three crashes, both mechanisms exact

**`OverflowError: timeout is too large`** — `ingest/commits.py` passes
`timeout=GIT_TIMEOUT * len(rel_paths)` to a subprocess. With
`GIT_TIMEOUT = 120.0` and Python's `poll()` capped at 2³¹−1 milliseconds,
**any repository with more than 17 896 files to blame ends the whole
run**. Hit by syncthing (`docs-pre-rendered`, 33 048 files) and
LadybirdBrowser (`ladybird`, 19 253 source files). The second was
predicted from the first before it was run, and came true.

**`UnicodeEncodeError: surrogates not allowed`** — `read_commits`
decodes git's output with surrogate escapes, so a commit message
containing a non-UTF-8 byte yields a lone surrogate; `Chunk.chunk_id`
then calls `text.encode("utf-8")` and raises. **Six of FreeType's 8 545
commit messages** carry one — Latin-1 names written between 2000 and
2005: Céline, Würkner, Syrjälä, Domröse. Six messages out of eight and a
half thousand stop the indexing of a whole workspace.

Neither is a degraded index. Both are a traceback and an empty store.

### Three silences, which are worse

`pow-auth` indexed **30 of 417 files**. `circe` indexed 67 of 455.
`phoenixframework`, 286 of 892. Their languages — Elixir and Scala — have
no entry in the language table, so their files are not chunked as text,
they are **skipped entirely**.

The cure is three lines of `[repos.formats]` config, and `wsindex explain <path>` says exactly that when asked. But `wsindex index` printed
`files: 30` and nothing else, and `wsindex status` showed three healthy
repositories. wsindex warns loudly when a grammar *partly* fails to read
one file. It says nothing at all when it skips 93% of a workspace.

For those three, the resulting index is 88%, 96% and 80% commit messages,
and `search --kind code` answers `no results`.

## Does search work?

204 questions, 17 workspaces. Each cell counts answers found at all —
**default** is `wsindex search -k 10`, **best** is `-k 50 --kind code --kind doc`, **ripgrep** is `rg -l` returning twenty files or fewer.

| Class           | Questions | Default |   Best | ripgrep                            |
| --------------- | --------: | ------: | -----: | ---------------------------------- |
| Literal         |        68 |      34 |     46 | **56** found, 10 drowned, 2 missed |
| **Descriptive** |       102 |   **5** | **28** | **0** found, 1 drowned, 101 missed |
| Cross-repo      |        34 |       9 |     14 | 15 found, 7 drowned, 12 missed     |

Read the descriptive row twice. **ripgrep found none of them** — so the
questions are real, and the need for something other than grep is real.
And wsindex, by default, found five.

### The index is not the problem

Handed a line taken verbatim out of the answer file, the index returns
that file **first** in 44 of 48 cases across four workspaces. Storage,
chunking, embedding and search are all sound. What fails is the crossing
from a developer's plain English to the code's vocabulary, and that is a
property of `all-MiniLM-L6-v2`, a general-purpose sentence model with a
256-token window.

### The defaults hide what there is

Descriptive answers found: **5 at `-k 10`, 28 at `-k 50 --kind code --kind doc`**. A fifth of the class is reachable and is not being
reached, because it sits at ranks 10–50 behind commit messages.

Commit chunks are between 8% and 96% of an index. On `DatabaseCleaner`,
27 of 36 top-three slots went to commit messages; on `circe`, 31 of 36.
Asked about a named constant in a Rust repository, the top ten hits were
all commits, with scores within 0.005 of each other.

The re-ranker is built for exactly this and cannot reach it: measured on
one workspace, it lifted an answer from rank 6 to rank 1 and another from
14 to 8, but the answers sitting at 17, 22 and 46 were never in its
candidate pool.

### Where it already beats grep

On the two largest workspaces that indexed, ripgrep starts drowning and
ranking starts paying:

- **dbeaver** (125 353 chunks): wsindex put **4 of 4** literal answers in
  the top three. ripgrep found one, drowned on two.
- **icsharpcode** (78 440 chunks): wsindex 3 of 4 in the top ten by
  default, 4 of 4 configured. ripgrep found one, drowned on three.

This is the shape of the real case: the bigger the codebase, the less a
flat list of matches is worth.

## The analysis commands

`domains` ran 193 times across the corpus. **151 of those runs reported
"too few to have domains. Index first," while a complete index sat
beside them** — the default `--prefix src/` is this project's own layout,
and a Ruby gem keeps code in `lib/`, a Go module at the root, a Java
project under `src/main/java`. Given the right prefix it works: 42 runs
reported real package counts.

`dupes` found pairs in 47 repositories and nothing above the threshold in
42\. It is scoped to one repository at a time, so on the workspace chosen
*because* its three adapter gems are near-copies of each other, it cannot
see the duplication at all.

`refs` found nothing for either port probe in any of 29 attempts, which
matches what the project already measured: the link vocabulary is thin.

`explain` was the best-behaved command in the trial. Asked about 240
sampled files it gave a correct and specific answer every time, including
the three workspaces where it was the only thing that could have told a
user what had gone wrong.

## The verdict, against the rule declared in advance

The rule was written before the run: **works** if the cold index
completes and its warnings are true; **needed** if descriptive `hit@3 ≥ 0.6` while the control finds the answer in no more than 30% of the same
questions.

**Works: 14 of 20.** Three crashed, three said nothing while indexing
almost nothing.

**Needed: the need is proven, the answer is not delivered.** The control
found **0 of 102** descriptive answers, so the questions developers
cannot grep are real and common. wsindex placed **1 of 102** in the top
three. That is the cell of the table marked *fails at its own job* — and
it fails at a job that genuinely needs doing.

The machinery under it is not what fails. An index that returns the right
file first for 92% of verbatim queries, re-indexes 125 000 chunks in
0.68 s, and beats grep outright on the largest codebase in the corpus is
a working engine with the wrong model in it, a candidate pool too
shallow, and history drowning the code.

## What happened next, and it changes the verdict above

That paragraph was written as a consolation and turned out to be the
finding. Everything it names was fixed or tested afterwards, and the last
of them settled the question this document could not.

History was capped at a fifth of a result list; chunks were given their
own name and place to be embedded with; `domains` and `dupes` were made
to read a workspace rather than one project's layout and one repository;
the two crashes and the silent skipping were fixed with tests that pin
the mechanism. The questions here became a permanent instrument, `poe relevance`, with the ripgrep control and a pinned corpus, running in CI.

The thin link vocabulary was diagnosed rather than accepted. `refs` found
nothing in 29 attempts because the extractor had never read the name the
syntax tree already put on every chunk — the vocabulary was whatever four
regular expressions caught, and nothing else. Reading it gives caddyserver
1 090 names with a definition in one file and a mention in another, where
five workspaces had previously produced three.

Then the model was replaced. **Same chunks, same questions, same
pipeline**: the plain-English class goes from 2 to **12 of 23 in the top
three** and 5 to **18 of 23 in the top ten**, and identifier questions to
16 of 16 — every one that is reachable. The gate this trial declared in
advance, descriptive hit@10 of 0.50, is cleared at 0.78.

So the verdict stands as a description of what shipped on the day, and
must not be read as a description of the design. The promise was not
unachievable. It was locked behind a 23M-parameter model chosen to keep
everything on one laptop, and every part of the system underneath it was
sound. What is still true: that trade is a real one, and nothing in this
repository sends your code anywhere by default.
