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
- Scopes: `collect`, `parse`, `dedup`, `index`, `retrieve`, `chat`, `eval`, `report`, `common`, `config`, `cli`, `probes`, `decisions`, `journal`, `notes`, `deps`, `showcase`, `ui`.
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

- Never edit gold labels in `eval/gold/questions.jsonl` or `eval/pairs.jsonl`. They are Sharad's. Report problems with them instead.
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
- [x] Phase 3: Evaluation set (2026-10-26; tooling only: Sharad's questions pending)
- [x] Phase 4: Baseline retrieval + eval harness (2026-10-26; baseline numbers wait for Sharad's questions)
- [x] Phase 5: Versions and near-duplicates (2026-10-28; resolver precision/recall and Recall/MRR columns wait for Sharad's labelled pairs and questions)
- [x] Phase 6: Embedding comparison + hybrid (2026-10-29; silver numbers only: gold Recall/MRR and Sharad's recommendation wait for his questions)
- [x] Phase 7: Grounded multi-turn chat (2026-10-26; local qwen3:8b default; Sharad's prompt and year-filter tasks open)
- [x] Phase 8: Answer evaluation + failure analysis (2026-11-02; tooling, runs and transcript done: Sharad's labels (he is the judge) and failure analysis open)
- [ ] Phase 9: Deep dives
- [ ] Phase 10: Assignment write-up + reproducibility
- [ ] Phase 11: Journey write-up

Current phase: none in progress. Order changed by Sharad on 2026-10-26: Phase 3 tooling -> 4 -> 7 (chat on the baseline retriever), then Sharad inspects each stage's quality one by one (and writes the eval questions), then Phases 5 and 6. Phases 5, 6 and 8 are done (8 awaits Sharad's labels and failure analysis); ask Sharad which phase is next. Notes for the next session:
- Phase 8 (D-046): `rag eval answers` (gate off; thresholds simulated), `rag eval label` (Sharad is the judge: claims, answers, rewrites), `rag eval answer-report`, `rag eval transcript --turns probes/phase-8-transcript.txt`. First run `results/2026-11-02-answers.jsonl` (b233815): 5/5 unanswerable refused by the model, 1/23 false abstention (C4-T2 unrewritten), 38% uncited claims. `docs/phase8-failures.md` is Sharad's (TODO(sharad)). Worktree runs need a fresh `config-abs.yaml` every time config.yaml changes.
- Eval set (2026-11-02): `eval/gold/questions.jsonl` has 28 questions built from Sharad's LLM-drafted `eval/gold/questions.draft.jsonl` (untracked, his) by `rag eval check-papers|resolve|absent|finalize` (D-045, `src/rag/eval/draft.py`). At Sharad's request Claude picked the spans (`eval/gold/claude_resolution.yaml`, applied by `eval/gold/resolve_by_claude.py`); provenance says so; Sharad's review pending. 14 skipped (`eval/gold/GOLD_BUILD.md`): MEGABYTE and Byte Latent Transformer aren't in the corpus; U01/U03/U07 are answerable. The mix is short (multi_paper 7/8, followup 4/8, unanswerable 4/6, conversations 2/3), so `validate` fails on the mix only. Phase 6's gold numbers can now be run: `rag eval ablate --name embeddings`.
- Phase 6 built three dense indexes side by side (`index.dense_models`: bge_small deployed, bge_base, specter2; `rag index dense --model <name>`, `Retriever.load(..., dense=<name>)`), an optional reranker (`retrieve.rerank`, off; bge-reranker-base over the fused top 50, before collapse), `rag eval silver` (`eval/silver/`, 66 questions, D-044: never pooled with gold) and `rag eval ablate --name embeddings` (pairs in `eval.ablations.embeddings.pairs`). Results `results/2026-10-29-embeddings-ablation.md`; probe `docs/figures/phase6-terms.md`; D-043 proposed: keep bge-small hybrid, reranker pending gold (Sharad decides whether to enable it). See `notes/phase-6-embeddings.md`.
- SPECTER2 runs in `uv run --isolated` via `src/rag/retrieve/specter2_worker.py` (adapters pins transformers 4.57). Its cosines bunch near 0.75: switching the deployed model needs `probes/phase-7-gate.py` and `rag dedup run` rerun.
- Long background jobs: run with `HF_HUB_OFFLINE=1 caffeinate -i ...`. A sleep mid-download hung a worker on a socket read for hours. Builds from a worktree need a fresh absolute-path copy of `config.yaml` (config sections change).
- Phase 5 built `rag dedup run|pairs|score` (`src/rag/dedup/`): 1,781 records -> 1,298 works (identifiers give 1,414; +125 title, +12 MinHash merges; chain guard refused 0). Outputs in `data/works/` (gitignored; `review.md` lists fuzzy merges). The Retriever loads the works when `retrieve.dedup.enabled` (refuses them if built from other chunks): canonical mask, per-work cap 1, survey demotion within top k, MMR off (`retrieve/collapse.py`, D-041). Rerun `rag dedup run` after any re-chunk or `rag index dense` (restatements need the dense vectors). `rag eval ablate --name dedup` runs the ablation (probe queries in `probes/phase-5-queries.txt`; adds Recall/MRR once questions exist). The dup@5 judge = identifiers + Sharad's `eval/pairs.jsonl` (D-042), never the resolver. See `notes/phase-5-dedup.md`.
- Showcase UI side track (2026-10-27, D-037/D-038): `make ui`. Pages so far: system diagram, Collect, Parse and chunk. Pattern for a new stage page: logic in `src/rag/showcase/<stage>.py` (tested, no streamlit import), rendering in `ui/pages/<stage>.py`, register it in `ui/app.py`, add a builder to `BUILDERS` in `showcase/cli.py` if a fresh clone lacks the data. Numbers come from stage outputs, never `stats.json`; examples are sampled with a seed ("Show others"). Regenerate snapshots with `rag showcase snapshot`. 3 manifest records were decided on text that later changed (the page flags them).
- Phase 7 built `rag chat` (`src/rag/chat/`): rewrite -> hybrid search (k 6) -> retrieval gate (best dense cosine < 0.70) -> context [C1..] -> generate -> NOT_SUPPORTED gate -> citations renumbered by code. Prompts are files in `prompts/`. Transcripts in `data/chat/*.jsonl` (gitignored). See `notes/phase-7-chat.md`, `docs/phase7-demo.md`.
- LLM: Ollama `qwen3:8b` by default (`ollama pull qwen3:8b` done; ~7-17 s per answer). `--provider anthropic` needs ANTHROPIC_API_KEY in .env (`claude-haiku-5-5`). The anthropic SDK uses its own httpx2: respx can't mock it; tests inject a fake client.
- The gate threshold 0.70 comes from `probes/phase-7-gate.py` (probe queries, not gold). A new embedding model (Phase 6) needs the probe rerun.
- `make smoke` now runs in `.smoke/` (a config copy; data/raw symlinked) and needs GROBID and network.
- Chat now gets one chunk per work (Phase 5); hits carry `work_id` and `hidden` chunk ids, unused by chat so far.
- Phase 4 built `rag index bm25|dense|all` (data/index/bm25, data/index/dense/bge-small-en-v1.5), the Retriever (`rag.retrieve.retriever.Retriever.load(config, {"hybrid"})`, `.retrieve(query, k, mode, allow)`), `rag inspect query`, `rag eval run --name`, and `rag report`. See `notes/phase-4-retrieve.md`.
- Phase 7 chat should use `Retriever.retrieve(..., allow=...)` (the year filter hook is Sharad's task) and the top dense cosine (`Hit.scores["dense"]`) for the abstention gate, never RRF scores.
- Full dense rebuild takes 45 min on CPU. The Retriever refuses a stale index (chunks sha256 or settings changed). Build from a clean tree (or a clean worktree with `--config`) so stats carry a clean commit.
- Parse clean-up done 2026-10-27 (D-039, `src/rag/parse/cleanup.py`): caption labels stripped from GROBID's figDesc, bibliography text cut from body sections, figure labels/equations/running heads folded into the previous section. Re-chunked (92,820 chunks) and re-indexed. Residue on the Parse page: 76 / 18 / 164 chunks. Thresholds in `parse.clean`. `docs/architecture.md` is the system block diagram.
- Phase 3 built `rag eval find` (unranked grep over chunk text) and `rag eval validate` (schema, indexed papers, span -> chunk mapping, conversations, mix). `eval/gold/questions.jsonl` doesn't exist yet: Sharad writes it (`eval/README.md`).
- Phase 2 outputs: `rag parse` and `rag inspect paper`; 1,781 of 1,979 accepted papers indexed, 92,820 chunks in `data/parsed/chunks/chunks.jsonl`. See `notes/phase-2-parse.md`.
- Chunks: bge-small tokenizer, target 300 / max 448 / overlap 40; header stored separately (`Chunk.header`) and must be prepended at embedding time in Phase 4. References are in `data/parsed/papers/references.jsonl`, never chunked.
- GROBID is only needed for `rag parse grobid`/`all` (`make grobid-up`, Docker Desktop running). `structure`, `chunk`, `inspect` and `eval` work without it.
- `rag parse mark` writes `included_in_index` into the manifest; rerun it after `rag collect manifest`. It skips after a `--limit` run.
- D-013 to D-042 are `proposed`. PyMuPDF is pinned `<2`.
- Sharad's open "You do" items: Phase 5 label `eval/pairs.jsonl` from `eval/pairs_candidates.jsonl` then `rag dedup score`, review `data/works/review.md`, per-work cap 1 -> 2 and rerun the ablation (journal skeletons exist), Phase 5 prediction entry; Phase 7 system-prompt change (journal skeleton) and `--year-min` filter (`TODO(sharad)` in `chat/cli.py`); Phase 4 Recall@10 (`TODO(sharad)` in `eval/metrics.py`), hand-computed Recall@5/MRR for 2 questions, Phase 5/6 journal predictions; Phase 3 questions; Phase 2 inspection of 5 papers and the chunk-size comparison; Phase 1 audit labels, D-005 rule, seed query, journal predictions; Phase 0 `TODO(sharad)` config field and first journal entry.
- Chunks include linked copies and kept v1s on purpose; Phase 5 masks non-canonical records at query time (no rebuild).
- Typer 0.27 vendors Click privately (`typer._click`): don't `import click`. Local uv is 0.11.7.
