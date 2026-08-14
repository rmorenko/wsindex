
.PHONY: help install lint fmt fmt-check spell typecheck test cov check run hooks install-hooks clean up down logs
.DEFAULT_GOAL := help

# git ls-files (not a bare glob) so generated files like uv.lock and anything
# under .venv never reach the formatters.
YAML_FILES := $(shell git ls-files '*.yml' '*.yaml')
TOML_FILES := $(shell git ls-files '*.toml')
JSON_FILES := $(shell git ls-files '*.json')
XML_FILES := $(shell git ls-files '*.xml' '*.xsd')

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies (uv sync)
	uv sync

lint:  ## Run linter (ruff check)
	uv run ruff check .

fmt:  ## Format code, docs and configs (ruff w/ safe fixes, mdformat, taplo, yamlfix, json.tool, xmllint)
	uv run ruff check --fix .
	uv run ruff format .
	uv run mdformat *.md
	uv run taplo fmt $(TOML_FILES)
	uv run yamlfix $(YAML_FILES)
	@for f in $(JSON_FILES); do uv run python -m json.tool --indent 2 "$$f" "$$f.tmp" && mv "$$f.tmp" "$$f"; done
	@for f in $(XML_FILES); do xmllint --format "$$f" --output "$$f"; done

fmt-check:  ## Check formatting without changing files
	uv run ruff format --check .
	uv run mdformat --check *.md
	uv run taplo fmt --check $(TOML_FILES)
	uv run yamlfix --check $(YAML_FILES)
	@for f in $(JSON_FILES); do uv run python -m json.tool --indent 2 "$$f" | diff -u "$$f" - || exit 1; done
	@for f in $(XML_FILES); do xmllint --format "$$f" | diff -u "$$f" - || exit 1; done

spell:  ## Spell check (codespell)
	uv run codespell

typecheck:  ## Type check (mypy)
	uv run mypy src tests

test:  ## Run tests (pytest, no coverage)
	uv run pytest

cov:  ## Run tests with coverage, fail under 90%
	uv run pytest --cov --cov-report=term-missing

check: fmt lint spell typecheck cov  ## Auto-format, then lint + spelling + types + tests w/ coverage

run:  ## Run CLI (uv run wsindex)
	uv run wsindex

acceptance:  ## Full MVP acceptance on a real corpus, writes acceptance_report.md
	uv run python scripts/acceptance.py

hooks:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

install-hooks:  ## Install the git pre-commit hook
	uv run pre-commit install

up:  ## Start Tensorus (docker compose up -d)
	docker compose up -d

down:  ## Stop Tensorus (docker compose down)
	docker compose down

logs:  ## Tail compose logs (docker compose logs -f)
	docker compose logs -f

clean:  ## Remove caches, build artifacts and index
	rm -rf .mypy_cache .ruff_cache .pytest_cache .wsindex build dist
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
