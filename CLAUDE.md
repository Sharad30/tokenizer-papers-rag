# CLAUDE.md

Standing instructions for Claude Code in this repository. Read this, `PLAN.md` and `DECISIONS.md` at the start of every session.

## Project

A RAG application and chatbot over research papers on LLM tokenizers: corpus collection, section-aware parsing and chunking, version and near-duplicate handling, an embedding comparison with evidence, and a grounded multi-turn chatbot with citations and abstention. Based on an AI-engineer take-home brief.

This is primarily a learning project. Sharad is building it to understand RAG deeply enough to defend every technique and every line of code, and to write about what he learned afterwards. Optimise for code he can understand, explain and change by hand, and for decisions backed by measurements. Speed and polish come second.

## Key files

- `PLAN.md`: the phased plan. Work only on the current phase.
- `STUDY.md`: Sharad's study guide. Not for you to act on, but useful context for what he knows.
- `DECISIONS.md`: one entry per decision: context, options, choice, why, evidence. You draft entries; Sharad reviews them.
- `JOURNAL.md`: Sharad's journal. You may add entry skeletons; never fill in his predictions, interpretations or conclusions.
- `notes/phase-N-<name>.md`: teaching notes you write at the end of each phase.
- `config.yaml`: every tunable value lives here.

## How to work

- One phase at a time, in the order in `PLAN.md`. Don't start work from a later phase.
- Begin each phase in plan mode. Propose files, stage contract (inputs, outputs, schemas), decisions with 2–3 alternatives each and their trade-offs, tests, and the concept probe. Wait for approval before implementing.
- Don't silently make a significant choice. If something comes up mid-phase that the plan didn't cover, stop and ask, with options.
- Before relying on any external API, rate limit, library API or model-loading detail, check current documentation. Don't rely on memory for these.
- Items marked "You do" in `PLAN.md` are Sharad's hands-on tasks. Don't do them. Where code is needed for them, leave a clearly marked `TODO(sharad)` stub with a short description.
- When Sharad asks why something works the way it does, explain with reference to the actual code (file and function) and the underlying concept.

## End of each phase

Stop and do all three:
1. **End-of-phase report:** what was built, decisions made and alternatives considered, how to run and inspect it, what the tests cover, known limitations.
2. **Teaching note** in `notes/phase-N-<name>.md`: each concept from the phase explained as it is implemented here, with file/function references, common pitfalls, and what to look at in the inspect output.
3. **Update the Progress section** at the bottom of this file.

## Code style

- Python 3.11+, managed with `uv` (`uv add`, `uv run`; never bare `pip`).
- Plain Python. No LangChain, LlamaIndex or similar frameworks: every mechanism should be visible in this repo.
- Small modules and functions; clear over clever. Type hints everywhere; pydantic models for every schema that crosses a stage boundary.
- Comments explain why, not what.
- No magic numbers: thresholds, k values, sizes, model names and weights go in `config.yaml`.
- Every stage reads only from the previous stage's output folder and writes to its own, plus a `stats.json`.
- Every CLI stage supports `--limit N` for fast end-to-end runs.
- CPU-only. Docker is available (used for GROBID).

## Tests

- pytest. Unit tests for pure logic (normalisers, metrics, chunker, dedup rules, citation parsing).
- Mock all network calls with recorded fixtures. Test fixtures must be synthetic or from CC-BY papers.
- Metric functions get tests against hand-computed examples, including edge cases.
- Run `make test` before every commit; don't commit failing tests.

## Git and commits

The commit history is part of the deliverable: a reader should be able to follow how the project progressed from the log alone. Commit regularly without being asked.

**When to commit**
- After each logical unit of work that runs and passes tests: typically every 15–45 minutes of work, never one large commit per phase.
- Before starting a risky change, so there's a clean point to return to.
- At the end of each phase: commit the teaching note and Progress update, then tag it `phase-N-done`, and push.
- Push to the remote at least at the end of every phase.

**What goes in one commit**
- One purpose per commit. Code and its tests together; experiment results and figures in their own commit; docs and decisions in their own commit.
- Don't mix a refactor with a feature or a fix.
- Stage specific files (`git add <paths>`), never `git add -A` or `git add .` blindly.
- If there are uncommitted changes you didn't make, they're Sharad's: don't commit them. Tell him so he can commit his own work himself.

**Message format** (Conventional Commits)

```
<type>(<scope>): <summary in imperative mood, max ~72 chars>

<body: what changed and why; wrap at ~72 chars>
```

- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, and `exp` for experiment runs and results.
- Scopes: `collect`, `parse`, `dedup`, `index`, `retrieve`, `chat`, `eval`, `report`, `common`, `config`, `cli`, `probes`, `decisions`, `journal`, `notes`, `deps`.
- Summary says what the change does ("add", "fix", "replace"), not "update code" or "wip".
- Body explains why, plus anything a reader can't see from the diff. Skip it only for trivial commits.
- `exp` commits put the key numbers in the body, the config that changed, and the results file path.

**Examples**

```
feat(collect): add arXiv client with rate limiting and response cache

Throttles to one request every 3 s per arXiv API terms; caches raw
responses under data/raw/arxiv so reruns make no network calls.
```

```
test(eval): add hand-computed cases for Recall@5 and MRR

Covers ties, no relevant result, and multiple gold spans per question.
```

```
exp(dedup): ablate exact-ID merge against baseline

dup@5 0.41 -> 0.22; Recall@5 0.63 -> 0.64 (CI overlaps).
Config: dedup.exact_id=true, other dedup steps off.
Results: results/2026-10-18-dedup-exact-id.json
```

```
docs(decisions): record chunk size choice and alternatives
```

Avoid: `wip`, `fix stuff`, `update`, `changes`, `final`, or messages that only repeat file names.

**Never**
- Squash, amend pushed commits, rebase shared history, or force-push.
- Commit PDFs, downloaded data, indexes, or `.env`. Keys come from environment variables; keep `.env.example` current.
- Commit with failing tests.

## Evaluation integrity

- Never edit gold labels in `eval/questions.jsonl` or `eval/pairs.jsonl`. They are Sharad's. Report problems with them instead.
- Every results file records the config and git commit that produced it.
- Report every run, including ones that look worse. Never drop an unfavourable run or cherry-pick questions.
- If a threshold or setting is tuned on the eval set, say so in the results and in `DECISIONS.md`.

## External sources

- Respect rate limits and terms of each source (arXiv, Semantic Scholar, ACL Anthology, OpenReview). Throttle conservatively, cache every response, send a descriptive user agent.
- Record each paper's licence. Never commit PDFs; the download script is the deliverable.

## Progress

Update at the end of each phase.

- [x] Phase 0: Scaffold (2026-10-16; Sharad's "You do" tasks still open)
- [x] Phase 1: Corpus collection (2026-10-23; audit labels and other "You do" tasks still open)
- [x] Phase 2: Parse and chunk (2026-10-25; Sharad's inspection and chunk-size tasks still open)
- [ ] Phase 3: Evaluation set
- [ ] Phase 4: Baseline retrieval + eval harness
- [ ] Phase 5: Versions and near-duplicates
- [ ] Phase 6: Embedding comparison + hybrid
- [ ] Phase 7: Grounded multi-turn chat
- [ ] Phase 8: Answer evaluation + failure analysis
- [ ] Phase 9: Deep dives
- [ ] Phase 10: Assignment write-up + reproducibility
- [ ] Phase 11: Journey write-up

Current phase: 3 (not started). Notes for the next session:
- Phase 2 built `rag parse` (grobid, structure, chunk, mark, all) and `rag inspect paper`. First full run: 1,781 of 1,979 accepted papers indexed (197 no PDF, 1 scanned), 93,031 chunks in `data/parsed/chunks/chunks.jsonl`. See `notes/phase-2-parse.md`.
- GROBID: `make grobid-up` (Docker Desktop must be running; image `grobid/grobid:0.9.1-crf` from config). TEI files in `data/parsed/tei/` are the cache, so reruns only re-send failures.
- Chunks: bge-small tokenizer, target 300 / max 448 / overlap 40; header stored separately (`Chunk.header`) and must be prepended at embedding time in Phase 4. References are in `data/parsed/papers/references.jsonl`, never chunked.
- `rag parse mark` writes `included_in_index` into the manifest; rerun it after `rag collect manifest`. It skips after a `--limit` run.
- D-013 to D-021 are `proposed`. PyMuPDF is pinned `<2` (2.0 pulls in an ONNX layout stack).
- Sharad's open "You do" items: Phase 2 inspection of 5 papers and the chunk-size comparison (journal skeletons added); Phase 1 audit labels, D-005 rule, seed query, journal predictions; Phase 0 `TODO(sharad)` config field and first journal entry.
- Chunks include linked copies and kept v1s on purpose; Phase 5 dedups.
- Typer 0.27 vendors Click privately (`typer._click`): don't `import click`. Local uv is 0.11.7.
