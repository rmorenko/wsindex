
.PHONY: help install lint fmt fmt-check typecheck test check run hooks install-hooks clean up down logs
.DEFAULT_GOAL := help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies (uv sync)
	uv sync

lint:  ## Run linter (ruff check)
	uv run ruff check .

fmt:  ## Format code (ruff format)
	uv run ruff format .

fmt-check:  ## Check formatting without changing files
	uv run ruff format --check .

typecheck:  ## Type check (mypy)
	uv run mypy src tests

test:  ## Run tests (pytest)
	uv run pytest

check: lint fmt-check typecheck test  ## Full pre-commit check: lint + format + types + tests

run:  ## Run CLI (uv run wsindex)
	uv run wsindex

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
