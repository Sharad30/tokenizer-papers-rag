# One command each to install, test, lint and run the pipeline end to end.
# Every target goes through `uv run`, so it uses the locked environment.

RAG := uv run rag
SMOKE_LIMIT := 2

.PHONY: help install test lint format check smoke grobid-up grobid-down clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create .venv from uv.lock (dev group included)
	uv sync

test:  ## Run the test suite
	uv run pytest

lint:  ## Check lint rules and formatting without changing files
	uv run ruff check .
	uv run ruff format --check .

format:  ## Apply ruff formatting and safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

check: lint test  ## Lint + tests: run before every commit

# Runs every stage in pipeline order on a tiny --limit. In Phase 0 the stages
# are stubs; each phase replaces one, and smoke keeps the whole chain honest.
smoke:  ## Run the whole pipeline on a tiny --limit
	$(RAG) --help > /dev/null
	$(RAG) config show > /dev/null
	$(RAG) collect all --limit $(SMOKE_LIMIT)
	$(RAG) parse --limit $(SMOKE_LIMIT)
	$(RAG) dedup --limit $(SMOKE_LIMIT)
	$(RAG) index --limit $(SMOKE_LIMIT)
	$(RAG) eval --limit $(SMOKE_LIMIT)
	$(RAG) report smoke

# GROBID (Phase 2) runs in Docker. The image comes from config.yaml so the
# version is pinned in one place.
GROBID_CONTAINER := rag-grobid
GROBID_IMAGE = $(shell uv run python -c "from rag.common.config import load_config; print(load_config().parse.grobid.image)")

grobid-up:  ## Start the GROBID service on port 8070 (needs Docker running)
	docker run -d --rm --init --ulimit core=0 --name $(GROBID_CONTAINER) -p 8070:8070 $(GROBID_IMAGE)
	@echo "GROBID starting; ready when http://localhost:8070/api/isalive returns true"

grobid-down:  ## Stop the GROBID service
	docker stop $(GROBID_CONTAINER)

clean:  ## Remove caches only; never touches data/, results/ or eval/
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
