# wsindex, for whoever decides whether a team adopts it

## What it is, in one paragraph

A command-line tool that indexes the several git repositories a team
works in and answers questions about them with exact file and line
numbers. It runs entirely on the developer's own machine: the model is
local, the index is a file beside the config, nothing is uploaded, and
there is no service to operate. It is one binary, no account, no server,
no per-seat cost.

Everything below is measured on twenty real codebases that are not its
own — 105 repositories from twenty GitHub organizations, chosen to
include the cases where it would struggle. The measurements, and how they
were taken, are in [field-trial.md](field-trial.md).

## What a team gets

**Search that keeps working as the codebase grows.** On the largest
codebase tested — DBeaver, 11 786 files — wsindex put the right file in
the top three for every identifier question asked. `ripgrep`, the tool
people actually use today, found one in four and returned over twenty
files for two more. The value is not that it beats grep on a small repo;
it is that it keeps working when grep stops being usable.

**One question across every repository at once.** Teams that split code,
docs, infrastructure and SDKs across repositories currently rely on
somebody knowing which repo holds what. This removes that dependency, and
it is the reason the tool exists.

**A duplication report that groups by cause.** `wsindex dupes` finds the
same code in two places and collapses the result by the directories
involved, so two copies of a vendored library read as one fact rather
than three hundred findings. It found real duplication in 47 of the
repositories tested.

**Onboarding evidence.** `wsindex domains` reads what a repository is
made of and which files keep changing together across package
boundaries — the pairs where two modules are coupled but not about the
same subject are the ones worth a conversation.

**Nothing leaves the machine.** This is a design principle, not a
setting: no telemetry, no cloud call, no code sent to a model provider.
For a team that cannot send source to a third party, this is the whole
reason to look at it.

## What it costs

|                                          | Measured                              |
| ---------------------------------------- | ------------------------------------- |
| First index, mid-size repo (2 000 files) | 37 seconds                            |
| First index, large repo (11 800 files)   | 4 minutes 18 seconds                  |
| Re-index after no changes                | under a second, whatever the size     |
| Disk, large workspace                    | 349 MB for 125 000 chunks             |
| Ongoing cost                             | none — no server, no licence, no seat |

Setup is three commands per workspace and it is a developer's own
decision; there is nothing central to provision.

## What it does not do

**It does not yet answer questions asked in plain English.** The trial
put 102 questions phrased the way a person thinks, deliberately avoiding
the words used in the code. wsindex placed the right answer in its top
three for one of them.

This is worth reading carefully in both directions. `ripgrep` found
**none** of those 102, so the gap is real and your developers are living
with it today. But wsindex does not close it yet, and anyone selling it to
your team on "ask your codebase a question" would be describing something
that does not work. The reason is identified — a general-purpose sentence
model rather than a code-aware one — and it is fixable, but it is not
fixed.

**It is not finished software, though it is less unfinished than the
trial found it.** Three of the twenty workspaces could not be indexed at
all — two crashed on a repository with more than ~17 900 files, one on a
twenty-year-old commit message containing a non-UTF-8 character — and
three more indexed almost nothing while reporting success. All of that
is fixed and tested. What is not fixed is the class of question below.

**It does not support every language.** Sixteen languages get a real
syntax tree. Elixir, Scala, Swift and Objective-C get nothing at all
unless somebody adds a configuration entry per repository.

## Where it fits, and where it does not

**A good fit:** several repositories, at least one of them large, a
polyglot mix drawn from the supported languages, and a hard requirement
that source code stays on the machine. The bigger and more scattered the
codebase, the more it is worth.

**A poor fit today:** a single small repository — grep is enough; a
workspace in an unsupported language — it will index your READMEs and
nothing else; a team that wants natural-language questions answered —
that is the part that does not work yet.

## How to decide

Pick your largest repository and your most scattered workspace. Install
it, index, and give two developers a week with `-k 50 --kind code` as
their habit. Ask them one question afterwards: *when you needed to find
where something lived, did you reach for this or for grep?*

That is a cheaper experiment than any argument about it, and it is the
one this trial could not run — the twenty codebases were strangers'. On
your own code, with your own people, the answer may differ, and it is the
only answer that decides anything.
