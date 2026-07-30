# WSIndex

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

**WSIndex** is a learning-first Python CLI that indexes a developer workspace —
multiple repositories, with **code and configs parsed via AST** and **documents
indexed as text** — and serves semantic search over it. It uses
[Tensorus v1](https://github.com/tensorus/v1) as its vector database and is
designed to grow from an educational project into a product.

> **Status — skeleton only.** This repository currently ships the **project
> scaffold and toolchain**. The package contains a single placeholder module
> whose only job is to prove that every tool (uv, ruff, mypy, pytest, pre-commit,
> CI) works end to end. The real indexing engine, described in the design docs,
> is implemented next.

## Design docs

- Concept — [CONCEPT_en.md](CONCEPT_en.md)
- Business requirements — [BRD_en.md](BRD_en.md)
- Architecture — [ARCH_en.md](ARCH_en.md)

_(Russian originals: `CONCEPT_ru.md`, `BRD_ru.md`, `ARCH_ru.md`.)_

## Requirements

- [uv](https://docs.astral.sh/uv/) — package & environment manager
- Python 3.11+ (uv fetches it automatically)
- `make` (optional, for the shortcuts below)

## Quickstart

```bash
uv sync            # create the venv and install dev tools
make check         # ruff + mypy + pytest
uv run wsindex     # run the placeholder CLI
```

Without `make`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Make targets

| Target | Description |
| --- | --- |
| `make install` | Sync dependencies (`uv sync`) |
| `make lint` | Lint with ruff |
| `make fmt` | Format with ruff |
| `make typecheck` | Type-check with mypy |
| `make test` | Run pytest |
| `make check` | Lint + type-check + test |
| `make run` | Run the `wsindex` CLI |
| `make hooks` | Run all pre-commit hooks |
| `make clean` | Remove caches and build artifacts |

## Project layout

```
.
├── pyproject.toml            # uv + hatchling, ruff, mypy, pytest config
├── Makefile                  # dev shortcuts
├── .pre-commit-config.yaml   # ruff, mypy, hygiene hooks
├── .github/workflows/ci.yml  # CI matrix (Python 3.11–3.13)
├── src/wsindex/__init__.py   # placeholder module (single file for now)
└── tests/test_wsindex.py     # toolchain smoke test
```

## Toolchain

Package management **uv** · lint & format **ruff** · types **mypy (strict)** ·
tests **pytest** · hooks **pre-commit** · CI **GitHub Actions**.

## License

MIT — see [LICENSE](LICENSE).
