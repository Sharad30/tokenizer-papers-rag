# One command each to install, test, lint and run the pipeline end to end.
# Every target goes through `uv run`, so it uses the locked environment.

RAG := uv run rag
SMOKE_LIMIT := 2

.PHONY: help install test lint format check smoke ui grobid-up grobid-down clean

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

ui:  ## Open the showcase UI (one page per stage) in the browser
	uv run streamlit run ui/app.py

# Runs every stage in pipeline order on a tiny --limit. It uses a copy of
# config.yaml in .smoke/: every path resolves against the config file's
# directory, so all outputs land in .smoke/ and the real manifest, chunks
# and 45-minute dense index are never overwritten. The raw download cache
# is shared (symlink) so smoke doesn't refetch what's already cached.
# Needs GROBID running (`make grobid-up`) and network access.
SMOKE_DIR := .smoke
SMOKE_RAG = uv run rag --config $(SMOKE_DIR)/config.yaml

smoke:  ## Run the whole pipeline on a tiny --limit, in .smoke/
	rm -rf $(SMOKE_DIR) && mkdir -p $(SMOKE_DIR)/data
	cp config.yaml $(SMOKE_DIR)/ && cp -R prompts $(SMOKE_DIR)/
	ln -s ../../data/raw $(SMOKE_DIR)/data/raw
	$(SMOKE_RAG) --help > /dev/null
	$(SMOKE_RAG) config show > /dev/null
	$(SMOKE_RAG) collect all --limit $(SMOKE_LIMIT)
	$(SMOKE_RAG) parse all --limit $(SMOKE_LIMIT)
	$(SMOKE_RAG) dedup --limit $(SMOKE_LIMIT)
	$(SMOKE_RAG) index all --limit $(SMOKE_LIMIT)
	$(SMOKE_RAG) eval validate
	$(SMOKE_RAG) eval run --name smoke --limit $(SMOKE_LIMIT)
	@f=$$(ls -t $(SMOKE_DIR)/results/*-smoke.json 2>/dev/null | head -1); \
	if [ -n "$$f" ]; then $(SMOKE_RAG) report $$f; else echo "report: no smoke results (no questions yet)"; fi

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
