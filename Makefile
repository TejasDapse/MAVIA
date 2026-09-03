.PHONY: help install install-all sync lint fmt type test cov clean data doctor

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install core deps + dev tools
	uv sync

install-all: ## Install every optional extra (vision, memory, agents, report, dashboard, eval)
	uv sync --all-extras

lint: ## Ruff check
	uv run ruff check src tests scripts

fmt: ## Ruff format + autofix
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

type: ## Mypy
	uv run mypy

test: ## Run unit tests (skips slow/integration)
	uv run pytest -m "not slow and not integration"

cov: ## Tests with coverage report
	uv run pytest --cov --cov-report=term-missing

data: ## Download MVTec AD into data/mvtec_ad
	uv run python scripts/download_mvtec.py

doctor: ## Show which parts of the stack are configured
	uv run mavia doctor

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
