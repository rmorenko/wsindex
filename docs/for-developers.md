# wsindex, for the developer deciding whether to install it

## What the thing is

You point it at the several git repositories you actually work in. It
reads them, cuts every file along its syntax tree — functions, classes,
config sections, document headings — turns each piece into a vector, and
keeps the lot in a file next to the config. Then you ask questions and it
gives you `file:line`.

Everything happens on your machine. The model runs in your process, the
index is a file you can delete, and a warm search opens zero sockets.
There is no account, no server to run, and nothing is sent anywhere.

That is the whole idea. What follows is what it is measured to do, on
twenty codebases that are not its own — see [field-trial.md](field-trial.md)
for how that was measured and what else it found.

## What it gives you today

**Finding the right file in a codebase too big to grep.** This is the
real win and it grows with the codebase. On DBeaver — 11 786 files,
125 000 chunks — wsindex put the right file in the top three for four out
of four identifier questions. `ripgrep` found one of the four and buried
two more in lists of over twenty files. That is the case you already know:
you grep a name, you get two hundred hits, and you start reading.

**One question, several repositories.** Searching six repos at once
without remembering which one holds what is not a trick, it is just what
the tool does. If your docs live in one repo and the code in another, both
answer.

**Knowing why a file is missing.** `wsindex explain path/to/file` tells
you whether it was indexed, as what, and if not, which rule left it out.
In the trial it was asked about 240 files and answered correctly every
time. Use it the moment anything looks wrong.

**Re-indexing that costs nothing.** Running `wsindex index` again after
no changes takes under a second and does not get slower as the corpus
grows — 0.68 s on the 125 000-chunk workspace. You can put it in a hook
and forget it.

## What it does not give you yet

**Asking in plain English does not work.** This is the honest headline.
The trial asked 102 questions phrased the way a person thinks — *"is
there logic that skips the wipe when nothing changed since last time"* —
deliberately using none of the words in the answer file. wsindex found
the answer in its top three for **one** of them.

`ripgrep` found **none** of the 102, so the questions are fair and the
need is real. But today wsindex does not fill it. The model underneath is
`all-MiniLM-L6-v2`, a general-purpose sentence model, and it does not
cross from English to code.

**Two defaults will cost you answers.** Out of the box, `wsindex search`
returns ten hits and includes commit messages. On a small repository with
a long history the index can be 89% commit messages, and they crowd out
the code — in one measured case the top ten hits for a constant were all
commits, scored within 0.005 of each other. Searching with

```console
$ wsindex search "your question" -k 50 --kind code --kind doc
```

found **28** of those 102 answers instead of 5. Make that your habit.

**If your language is not in the table, you get an empty index and no
warning.** Elixir and Scala have no entry, so those files are not chunked
as text — they are skipped. One workspace indexed 30 files out of 417 and
`wsindex index` said only `files: 30`. Check with `wsindex explain` on any
source file before you trust an index; if it says *no language claims this
suffix*, add a `formats` entry for the repo and re-index.

**Two crashes are waiting in real repositories.** A repository with more
than about 17 900 files ends the run with `OverflowError`. A single commit
message containing a non-UTF-8 byte — a name like *Würkner* written in
1996 — ends it with `UnicodeEncodeError`. Three of the twenty workspaces
hit one of these.

## Is it worth your twenty minutes

**Yes, if** you work across several repositories at once, at least one of
them is large, and you spend real time looking for where something lives.
Install it, index, and use `-k 50 --kind code`. It will save you the
two-hundred-hit grep.

**Not yet, if** what you wanted was to ask questions in words. That is
the thing it is named for and the thing it does not do yet.

**No, if** your workspace is Elixir, Scala, Swift or Objective-C and you
do not want to hand-write a `formats` table first.

## Starting

```console
$ wsindex init myworkspace
$ wsindex add-repo api ~/checkouts/api
$ wsindex add-repo docs ~/checkouts/docs
$ wsindex index
$ wsindex search "retry backoff" -k 50 --kind code
```

`wsindex shell` keeps the model loaded between questions, which is worth
it after the second search — a cold search pays about two seconds to load
the model, and that is most of what you wait for.

`wsindex status` shows what is indexed and at which commit. `wsindex explain <path>` is the first thing to run when an answer looks wrong.
