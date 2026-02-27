.PHONY: help install dev lint format typecheck test test-cov \
       pipeline train evaluate serve docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────

install:  ## Install package
	pip install -e .

dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

# ── Quality ──────────────────────────────────────────────────────────

lint:  ## Run ruff linter + mypy
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:  ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:  ## Run mypy only
	mypy src/ --ignore-missing-imports

# ── Testing ──────────────────────────────────────────────────────────

test:  ## Run tests
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Pipeline ─────────────────────────────────────────────────────────

pipeline:  ## Run full pipeline (download → split → train → evaluate)
	seqthreat pipeline -c configs/pipeline.yaml

download:  ## Generate synthetic dataset
	seqthreat download -c configs/pipeline.yaml

split:  ## Stratified split
	seqthreat split -c configs/pipeline.yaml

train:  ## Train classifier
	seqthreat train -c configs/pipeline.yaml

evaluate:  ## Evaluate on test set
	seqthreat evaluate -c configs/pipeline.yaml

# ── Serving ──────────────────────────────────────────────────────────

serve:  ## Start FastAPI dev server
	uvicorn src.serving.app:app --reload --port 8000

serve-prod:  ## Start production server
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 4

# ── Docker ───────────────────────────────────────────────────────────

docker:  ## Build Docker image
	docker build -t seqthreat .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 -v $(PWD)/models:/app/models seqthreat

# ── Cleanup ──────────────────────────────────────────────────────────

clean:  ## Remove caches and artifacts
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/__pycache__ tests/__pycache__
	find . -name '*.pyc' -delete
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean  ## Remove data, models, and MLflow artifacts
	rm -rf data/ models/ mlruns/
