# Contributing to WSIndex

Thanks for your interest! WSIndex is a learning-first project; contributions and
experiments are welcome.

## Development setup

You need [uv](https://docs.astral.sh/uv/). Then:

```bash
uv sync                     # create the venv and install dev tools
uv run pre-commit install   # enable git hooks (optional but recommended)
```

## Everyday workflow

Tasks are defined with [poethepoet](https://poethepoet.natn.io/) in
`pyproject.toml`; `uv run poe --help` lists them.

```bash
uv run poe check   # ruff (lint) + mypy (types) + pytest — run this before pushing
uv run poe fmt     # auto-format with ruff
uv run poe run     # run the wsindex CLI
```

Run everything the pre-commit hooks would run:

```bash
uv run poe hooks
```

## If you touch search

`poe check` cannot tell you whether search still finds the right file —
it has no corpus and no model. That is what `poe relevance` is for:

```bash
uv run poe relevance                                          # 60 questions, ~5 min
uv run poe relevance -- --check scripts/relevance_baseline.json   # fail on a drop
```

Sixty questions written by testers who explored a workspace with reading
and grep only and never ran wsindex, against repositories pinned to the
sha they were written at, with a `ripgrep` control on every one. Six of
the twelve per workspace describe a behaviour using no word that appears
in the answer file.

**The questions in `scripts/acceptance_corpus/` are not to be edited to
make a run pass** — the same rule `scripts/acceptance.py` already lives
by, and for the same reason. If a change makes the numbers better, move
the baseline with `--save` and say so in the commit message. If it makes
them worse, that is the finding.

Two tiers. A pull request gets the routine one — 60 questions, five
minutes — and the weekly run adds `dbeaver` and `icsharpcode` against
`scripts/relevance_baseline_full.json`. Those two are the large codebases
where this beats grep outright (on dbeaver, 6 of 12 answers in the top
three against ripgrep's 1 of 12), so they are the half most worth
guarding and the half too slow to guard on every push.

CI runs this automatically on any pull request touching the pipeline, the
store, the embedders, the ranker, the chunkers or the corpus itself — the
paths between a question and an answer. `poe acceptance` is a different
thing and a good one: an end-to-end smoke test across both backends, not
a measure of whether answers are right.

## Standards

- **Formatting & linting:** `ruff` (config in `pyproject.toml`). Run
  `uv run poe fmt` before committing; CI runs `ruff format --check`.
- **Types:** `mypy` in `strict` mode. All functions must be fully annotated.
- **Tests:** `pytest`. Add a test for any new behavior; keep tests fast and
  offline (no network, no heavy models).
- **Python:** target 3.11+ and keep the code cross-version clean
  (CI runs on 3.11, 3.12, 3.13).

## Pull requests

1. Branch off `main`.
1. Make your change with a matching test.
1. Ensure `uv run poe check` passes locally.
1. Open a PR; CI must be green before merge.

See the design docs (`design/concept.md`, `design/brd.md`,
`design/architecture.md`) for where the
project is heading.
