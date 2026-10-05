PYTHON ?= python3.12
VENV ?= .venv
PY   := $(VENV)/bin/python

.PHONY: help install lock lint format typecheck test test-live check ingest serve docker-build docker-run clean snapshot verify-snapshot index test-model serve-dense

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create venv, install package + dev tools, set up git hooks
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.lock
	$(PY) -m pip install --no-deps -e .
	$(VENV)/bin/pre-commit install

lock:  ## Re-pin dependencies after editing pyproject.toml (commit both lockfiles)
	$(PY) -m pip install -q uv
	$(VENV)/bin/uv pip compile pyproject.toml --python-version 3.12 --generate-hashes -o requirements.lock
	$(VENV)/bin/uv pip compile pyproject.toml --extra dev --python-version 3.12 -o requirements-dev.lock

lint:  ## Ruff lint + format check
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format:  ## Auto-fix lint issues and format
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

typecheck:  ## mypy --strict
	$(VENV)/bin/mypy

test:  ## Unit tests with coverage gate (no network)
	$(PY) -m pytest --cov --cov-report=term-missing --cov-fail-under=85

test-live:  ## Smoke tests against the real GOV.UK API
	$(PY) -m pytest -m network --no-cov

check: lint typecheck test  ## Everything CI runs

ingest:  ## Fetch (cached) + parse + chunk -> data/processed/
	$(PY) -m ukmoney_rag.ingest --sources configs/sources.toml

serve:  ## Run the API on :8080 over the committed data/snapshot (BM25 only)
	$(VENV)/bin/ukmoney-serve

serve-dense:  ## Same, plus /search?mode=dense (needs `make index` first)
	UKMONEY_DENSE=1 $(VENV)/bin/ukmoney-serve

IMAGE ?= ukmoney-rag:local

docker-build:  ## Build the production image (needs `make ingest` first)
	docker build --build-arg GIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo local) -t $(IMAGE) .

docker-run:  ## Run the production image on :8080
	docker run --rm -p 8080:8080 $(IMAGE)

clean:  ## Remove caches and build artefacts (keeps data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

snapshot:  ## Freeze data/processed into the committed snapshot (then review the diff + commit)
	mkdir -p data/snapshot
	cp data/processed/chunks.jsonl data/processed/manifest.json data/snapshot/
	@echo "corpus_sha256=$$(sha256sum data/snapshot/chunks.jsonl | cut -d' ' -f1)"

verify-snapshot:  ## Fail if the snapshot's bytes don't match what the eval set was written against
	@test "$$(sha256sum data/snapshot/chunks.jsonl | cut -d' ' -f1)" = "$$(cat eval/CORPUS_SHA256)" \
	  && echo "snapshot OK" || (echo "snapshot != eval/CORPUS_SHA256" && exit 1)

index:  ## Build the dense index from the committed snapshot
	$(VENV)/bin/ukmoney-build-index --data-dir data/snapshot --out indexes --model-cache .cache/models

test-model:  ## Slow tests that download/run the real embedding model
	$(PY) -m pytest -q -m model --no-cov
