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

Use the `make` shortcuts (all wrap `uv run …`):

```bash
make check   # ruff (lint) + mypy (types) + pytest — run this before pushing
make fmt     # auto-format with ruff
make run     # run the wsindex CLI
```

Run everything the pre-commit hooks would run:

```bash
make hooks
```

## Standards

- **Formatting & linting:** `ruff` (config in `pyproject.toml`). Run `make fmt`
  before committing; CI runs `ruff format --check`.
- **Types:** `mypy` in `strict` mode. All functions must be fully annotated.
- **Tests:** `pytest`. Add a test for any new behavior; keep tests fast and
  offline (no network, no heavy models).
- **Python:** target 3.11+ and keep the code cross-version clean
  (CI runs on 3.11, 3.12, 3.13).

## Pull requests

1. Branch off `main`.
2. Make your change with a matching test.
3. Ensure `make check` passes locally.
4. Open a PR; CI must be green before merge.

See the design docs (`CONCEPT_en.md`, `BRD_en.md`, `ARCH_en.md`) for where the
project is heading.
