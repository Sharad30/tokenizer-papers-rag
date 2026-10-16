# One command each to install, test, lint and run the pipeline end to end.
# Every target goes through `uv run`, so it uses the locked environment.

RAG := uv run rag
SMOKE_LIMIT := 2

.PHONY: help install test lint format check smoke clean

help:  ## List targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-8s %s\n", $$1, $$2}'

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
	$(RAG) collect --limit $(SMOKE_LIMIT)
	$(RAG) parse --limit $(SMOKE_LIMIT)
	$(RAG) dedup --limit $(SMOKE_LIMIT)
	$(RAG) index --limit $(SMOKE_LIMIT)
	$(RAG) eval --limit $(SMOKE_LIMIT)
	$(RAG) report smoke

clean:  ## Remove caches only; never touches data/, results/ or eval/
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
