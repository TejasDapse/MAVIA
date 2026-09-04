.PHONY: help install install-all sync lint fmt type test cov clean data doctor dashboard memory docker docker-run

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

dashboard: ## Launch the Streamlit operations dashboard
	uv run streamlit run src/mavia/dashboard/app.py

memory: ## Build and index the defect-history corpus
	uv run python scripts/build_memory.py --recreate

docker: ## Build the container image
	docker build -f docker/Dockerfile -t mavia:latest .

docker-run: ## Run the dashboard in Docker (mounts data, models, artifacts)
	docker run --rm -p 8501:8501 \
		-v "$$PWD/data:/app/data:ro" \
		-v "$$PWD/models:/app/models:ro" \
		-v "$$PWD/artifacts:/app/artifacts" \
		--env-file .env mavia:latest

doctor: ## Show which parts of the stack are configured
	uv run mavia doctor

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
