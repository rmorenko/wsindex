.PHONY: help install lint fmt typecheck test check run hooks clean
.DEFAULT_GOAL := help

help:  ## Показать это сообщение
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Установить зависимости (uv sync)
	uv sync

lint:  ## Проверка линтером (ruff check)
	uv run ruff check .

fmt:  ## Форматирование кода (ruff format)
	uv run ruff format .

typecheck:  ## Проверка типов (mypy)
	uv run mypy src

test:  ## Запуск тестов (pytest)
	uv run pytest

check: lint typecheck test  ## Полная проверка: линтер + типы + тесты

run:  ## Запустить CLI (uv run wsindex)
	uv run wsindex

hooks:  ## Прогнать pre-commit по всем файлам
	uv run pre-commit run --all-files

clean:  ## Удалить кэши, артефакты сборки и индекс
	rm -rf .mypy_cache .ruff_cache .pytest_cache .wsindex build dist
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
