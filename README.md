# tokenizer-papers-rag

A retrieval-augmented chatbot over research papers on LLM tokenizers: corpus
collection, section-aware parsing and chunking, version and near-duplicate
handling, an embedding comparison backed by evidence, and grounded multi-turn
chat with citations and abstention. Plain Python, no RAG frameworks.

> Status: Phase 0 (scaffold). Every pipeline command exists but is a stub.
> See `PLAN.md` for the phases.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (uv installs it if missing).

```bash
uv sync                 # create .venv from uv.lock
uv run rag --help       # list commands
make check              # lint + tests
make smoke              # run every stage on --limit 2
```

Copy `.env.example` to `.env` and fill in keys when a phase needs them.

## Pipeline

| Command | Reads | Writes |
|---|---|---|
| `rag collect` | arXiv, Semantic Scholar, ACL Anthology | `data/raw/`, `data/manifest/` |
| `rag parse` | `data/raw/` | `data/parsed/` |
| `rag dedup` | `data/parsed/` | `data/works/` |
| `rag index` | `data/parsed/`, `data/works/` | `data/index/` |
| `rag eval` | `data/index/`, `eval/` | `results/` |
| `rag report <run>` | `results/<run>.json` | `docs/figures/` |
| `rag chat` | `data/index/` | terminal |
| `rag inspect paper\|work\|query` | stage outputs | terminal |

Every stage takes `--limit N` and writes a `stats.json` next to its outputs.
All settings live in `config.yaml` (`uv run rag config show` prints the
validated version). Use `--config PATH` or `RAG_CONFIG` to point at another config.

## Repository map

- `docs/architecture.md`: block diagram of the whole system and where to look for each block
- `src/rag/`: the package, one folder per stage plus `common/`
- `tests/`: pytest suite (`make test`)
- `DECISIONS.md`: every significant choice with alternatives and evidence
- `JOURNAL.md`: predictions and results, written before and after each experiment
- `notes/`: teaching notes per phase · `probes/`: small concept experiments

## Quick start, expected runtimes, results

To be written in Phase 10.
