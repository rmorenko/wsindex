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
