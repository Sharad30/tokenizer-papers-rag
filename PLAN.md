# Tokenizer-literature RAG: build and learning plan

A phased plan for building the take-home with Claude Code, designed for two goals at once:

1. **Finish the assignment** to the standard the brief describes (core track).
2. **Understand the concepts deeply and capture the journey**, so you can explain every choice and later write about the interesting things you found (learning layer + exploration track).

Each phase has a file-based contract (what it reads and writes), its own tests, a way to inspect what it produced, concepts to learn, a hands-on task without AI, and optional experiments. Any phase can be re-run, tested and improved on its own, and every change shows up in the eval numbers.

---

## Three tracks, three outputs

| Track | What | Time | Output |
|---|---|---|---|
| Core | Phases 0–8: a working, measured system | ~2.5–3 days | The repo |
| Exploration | Phase 9: 2–3 deep dives from the ★ ideas | ~1–1.5 days | Experiments with figures |
| Write-up | Phases 10–11 | ~1 day | Assignment write-up + a journey write-up |

Do the core track first. The deep dives need a working eval harness and chatbot to measure against. Small ★ experiments inside a phase can be done in passing when cheap; the big ones wait for Phase 9.

If you want to stay close to 2.5 days, do the concept probes in every phase and one deep dive.

---

## The learning loop (applies to every phase)

**Study first.** Before each phase, do its study block in `STUDY.md` (about 45–90 minutes; the Phase 0 domain primer is longer) and write a short pre-note in your own words. This is what lets you judge the alternatives Claude Code proposes in plan mode and write a real prediction.

**Predict, run, explain.** Before running any experiment or eval, write your prediction in `JOURNAL.md` ("I expect hybrid to beat dense by ~0.1 Recall@5 because..."). After running, record the result and explain the gap. Your wrong predictions are the most valuable material for the journey write-up, and the habit forces you to have a mental model before seeing numbers.

**Three living documents.**
- `DECISIONS.md`: what was chosen and why (options, choice, evidence). Feeds the assignment write-up.
- `JOURNAL.md`: dated entries: predictions, results, surprises, dead ends, questions. Feeds the journey write-up. Your words, not Claude Code's.
- `notes/phase-N-<name>.md`: a teaching note Claude Code writes at the end of each phase. It explains the phase's concepts as they are implemented, mapped to the file and function where each lives, plus common pitfalls. This is your study guide for defending the code.

**Every experiment produces a figure.** `rag report <run>` writes a markdown table and plots to `docs/figures/`. By the end you'll have a folder of figures ready for the write-ups.

**Concept probes.** Each phase includes a small script or notebook (`probes/`) that makes one concept visible: BM25 parameters, MinHash accuracy, chunk-length dilution. Ten minutes each. They build intuition faster than reading.

---

## How to use this plan with Claude Code

0. Do the phase's study block in `STUDY.md` and write the pre-note.
1. One phase at a time. Start each phase in plan mode, review the plan, then let it implement. If you can't judge an alternative it proposes, take it to Claude chat and get quizzed on the trade-off before approving.
2. At the end of each phase Claude Code stops and gives an end-of-phase report and writes the teaching note.
3. You do the phase's hands-on task, answer the "Understand" questions out loud, and write a journal entry. If you can't answer something, ask Claude Code to explain it with reference to the code. Don't move on until you can.
4. Commit small and often with clear messages. Never squash.

Prompt template for each phase:

```
We're starting Phase N (<name>) from PLAN.md. Read PLAN.md, DECISIONS.md,
JOURNAL.md and the outputs of earlier phases first. Enter plan mode and propose:
the files you'll create or change, the stage contract (inputs/outputs + schemas),
the decisions to make with 2-3 alternatives each, the tests, and the concept
probe. Verify any external API, rate limit or model-loading detail against
current docs before coding. Wait for my approval before implementing. Commit
after each logical step.

When done, stop and:
1. Give me the end-of-phase report: what was built, decisions + alternatives,
   how to run and inspect it, what the tests cover.
2. Write notes/phase-N-<name>.md: each concept from this phase explained as it
   is implemented here, with file/function references, pitfalls, and what to
   look at in the inspect output.
3. Add a JOURNAL.md entry skeleton for any experiment with "Prediction (Sharad):"
   left blank. Never fill in my predictions or interpretations.
```

---

## Global conventions (set up in Phase 0)

**Stage contracts.** Every stage reads only from the previous stage's output folder and writes to its own:

```
data/
  raw/          # Phase 1: API responses (cached), PDFs
  manifest/     # Phase 1: corpus manifest (committed: CSV/JSON)
  parsed/       # Phase 2: per-paper sections, chunks, references
  works/        # Phase 5: work_id clusters, restatement links
  index/        # Phase 4/6: BM25 + dense indexes, one folder per model
eval/
  questions.jsonl   # Phase 3 (committed)
  pairs.jsonl       # Phase 5 labelled duplicate pairs (committed)
results/            # every eval run: JSON + markdown table (committed)
docs/figures/       # plots from rag report (committed)
notes/              # teaching notes per phase (committed)
probes/             # concept probe scripts/notebooks (committed)
```

Each stage also writes a `stats.json` (counts, failures, timings). Data, PDFs and indexes are gitignored.

**One CLI, one config.** A `rag` CLI (Typer) with a subcommand per stage and a single `config.yaml` for every knob. No magic numbers in code. Every results file records the config and git commit that produced it, so any number can be traced and reproduced.

**Fast loop.** Every stage takes `--limit N` so the whole pipeline runs end to end on 5 papers in minutes.

**Inspection commands.** `rag inspect paper <id>`, `rag inspect work <work_id>`, `rag inspect query "<q>"` (ranked results with BM25, dense, fused and rerank scores side by side). Looking at intermediate outputs is where most understanding comes from.

**Code style.** Plain Python, no LangChain/LlamaIndex: every mechanism is visible and yours to explain. Small modules, type hints, pydantic schemas, pytest with mocked network calls. Test fixtures are synthetic or from CC-BY papers only.

**Reproducibility.** Pin a collection cutoff date in config (for example `max_date: 2026-10-15`). Keys in `.env` (gitignored); commit `.env.example`.

---

## Phase overview

| Phase | Name | Track | Main output |
|---|---|---|---|
| 0 | Scaffold | Core | Repo, CLI, config, tests, journal files |
| 1 | Corpus collection | Core | Manifest + PDFs, relevance audit |
| 2 | Parse and chunk | Core | Section-aware chunks, references store |
| 3 | Evaluation set | Core | 30–40 labelled questions |
| 4 | Baseline retrieval + eval harness | Core | Baseline Recall@5, MRR, dup@5, report tooling |
| 5 | Versions and near-duplicates | Core | work_id clusters, ablation table |
| 6 | Embedding comparison + hybrid | Core | Model comparison, recommendation |
| 7 | Grounded multi-turn chat | Core | CLI chatbot with citations and abstention |
| 8 | Answer evaluation + failure analysis | Core | Groundedness, abstention, transcript, failures |
| 9 | Deep dives | Exploration | 2–3 experiments with figures |
| 10 | Assignment write-up + reproducibility | Write-up | README, 2–3 page write-up |
| 11 | Journey write-up | Write-up | Your blog-style write-up |

Phases 1 and 2 have long-running jobs (downloads, GROBID). Start them in the background and write eval questions (Phase 3) while they run.

---

## Phase 0: Scaffold

**Goal.** A repo where every later phase has a place to go, and running, testing and reporting are one command each.

**Build.**
- `uv` project, Python 3.11+, `src/` layout: `collect/`, `parse/`, `dedup/`, `index/`, `retrieve/`, `chat/`, `eval/`, `report/`, `common/` (config, logging, schemas, IO).
- `rag` CLI with stub subcommands; typed pydantic config from `config.yaml`.
- pytest, a `Makefile` with `test`, `smoke`, `lint`.
- `.gitignore`, `.env.example`, empty `DECISIONS.md` and `JOURNAL.md` (with entry templates), `notes/`, `probes/`, README skeleton.
- Public GitHub repo created now; first commit pushed.

**Tests.** Config loads and validates; every subcommand's `--help` works.

**You do.** Read the structure and config schema. Add one config field yourself and use it in a stub command. Write your first journal entry: what you expect to be hardest and why.

**Exit.** `uv run rag --help` and `make test` pass on a fresh clone.

---

## Phase 1: Corpus collection

**Goal.** A rerunnable pipeline that finds candidate papers, decides relevance with an explicit rule, downloads PDFs, and records every decision in a manifest.

**Concepts to learn.**
- Citation-graph expansion (snowball sampling): why it finds papers keyword search misses, and its bias toward well-cited work.
- Filter evaluation: precision, recall, and estimating recall from an audit sample.
- Scholarly identifiers: arXiv ID and version, DOI, Semantic Scholar ID, ACL ID, and how they link.
- Rate limiting, caching, idempotent pipelines.
- *Probe:* plot accepted papers by year and source overlap. See what each source contributes and where coverage is thin.

**Build.**
- **Topic boundary**, written in DECISIONS.md before coding: include papers where tokenization is a primary contribution or a main experimental variable (BPE, WordPiece, Unigram/SentencePiece, byte/character-level and tokenizer-free models, vocabulary size, multilingual fairness and fertility, tokenization effects on arithmetic, code, reasoning). Exclude papers that only use a tokenizer.
- **Sources.**
  - arXiv API: seed keyword queries over title/abstract, date-bounded. At least 3 s between calls (verify current terms).
  - Semantic Scholar: resolve each paper; fetch `externalIds`, venue, year, publication types; snowball one hop through references and citations of anchor papers. API key if available; throttle conservatively.
  - ACL Anthology metadata (official Python library or BibTeX export) for venue versions.
  - OpenReview: optional.
- **Relevance filter, two stages.** Keyword rules on title + abstract give accept / reject / ambiguous; ambiguous go to a second judge (embedding similarity to a written topic description, or an LLM with the written rubric). Every reject keeps a reason code.
- **Multiple manifestations kept on purpose.** Keep arXiv + venue versions as separate records; for papers with substantial revisions also fetch v1. Record exactly how and why in DECISIONS.md, so the Phase 5 before/after is honest.
- **Download** with caching, resume, retries. Record each paper's licence.
- **Manifest** (CSV + JSON): id, source, title, authors, year, venue, url, arXiv id + version, DOI, S2 id, ACL id, licence, relevance decision, reason, included_in_index (filled later), download status.

**Tests.** Rate limiter spacing; re-runs hit the cache; manifest schema validation; keyword rules on hand-written titles.

**Run and inspect.** Counts per source and overlap, accept/reject breakdown by reason, download failures. Knobs: seed queries, anchor papers, snowball depth, rules, second-stage threshold.

**You do.** Write the inclusion rule in your own words. Label an audit sample of about 50 papers (25 accepted, 25 rejected); compute the filter's precision and rough recall. Add one seed query and re-run.

**Understand.**
- Why snowball instead of only keyword search, and what bias does it introduce?
- How would you know if your filter is too strict?
- Why does pinning a cutoff date matter for "rerunnable"?

**Explore.**
- ★ **External recall check.** Take the bibliographies of 2–3 recent tokenization surveys, keep the in-scope references, and measure what fraction your corpus contains. This gives a recall estimate from an independent source rather than your own judgment, and it's an unusually concrete number to report.

**Exit.** Manifest committed; audit numbers in DECISIONS.md.

---

## Phase 2: Parse and chunk

**Goal.** Clean, section-aware chunks that are good retrieval units and precise citation targets.

**Concepts to learn.**
- PDF structure extraction: how GROBID turns a PDF into TEI XML (layout tokens to labelled zones: header, sections, tables, references), and why PDFs lose structure in the first place.
- Chunking trade-offs: precision vs context, the dilution of an embedding as a chunk covers more topics.
- Embedding max sequence length and silent truncation.
- Contextual chunk headers: why prepending "Title > Section" helps a chunk that says "our method" mean something on its own.
- *Probe:* take a paragraph that answers a specific question; embed it alone, then padded with more and more surrounding text (128 → 1024 tokens). Plot cosine similarity to the question. You'll see dilution, and a flat line past the model's max length (truncation).

**Build.**
- GROBID via Docker (CPU-friendly image; check the current tag). PDF to TEI XML.
- TEI to structured paper: title, abstract, section tree, paragraphs, tables, figure captions, formulas, references, and in-text citation markers (which bibliography entry each sentence cites; needed for the restatement deep dive).
- PyMuPDF fallback for GROBID failures, flagged as lower parse quality.
- **Element handling.** Abstract: own chunk. Paragraphs: packed within a section, never across sections. Tables: own chunk, caption + linearised rows; large ones split by rows with header repeated. Figure captions: own chunks or attached to nearby text (decide). Formulas: short inline kept; long display replaced with a placeholder. References: not indexed; stored in `references.jsonl`.
- **Chunk sizing.** Token count with the embedding model's tokenizer; target ~300, hard max below the model limit including the header; small overlap within a section only.
- **Contextual header** stored separately and prepended at embedding time (so it can be switched off for the ablation).
- **Chunk metadata.** chunk_id, paper record id, arXiv version, section path and number, element type, page, token count, cited references.

**Tests.** Synthetic TEI fixtures: sections extracted; no chunk crosses a section; no chunk exceeds the max; references never in chunks; tables produce table chunks with captions; citation markers resolved to bibliography entries.

**Run and inspect.** Parse success rate, GROBID vs fallback, chunks per paper, token-length histogram, element-type counts. Knobs: chunk size, overlap, table splitting, captions, formula threshold.

**You do.** Open 5 random papers' chunks next to their PDFs and log problems in the journal. Change chunk size, re-run on `--limit 20`, compare stats.

**Understand.**
- Why must a chunk never cross a section boundary here?
- What happens silently when a chunk exceeds the embedding model's max length?
- Why exclude references from the index but keep them on disk?

**Explore** (run once the harness exists in Phase 4).
- ★ **Chunk granularity and header ablation.** Recall@5 vs chunk size (150/300/500 tokens), with and without contextual headers. One clear figure with a clear story.
- ★ **Table representation.** For lookup questions answered by tables, compare raw markdown rows vs a one-line generated description per table vs row-wise records. Tables are where "what vocabulary size did X use?" lives, and they're usually handled badly.

**Exit.** All accepted papers parsed or flagged; `included_in_index` filled in.

---

## Phase 3: Evaluation set

**Goal.** 30–40 labelled questions that make every later comparison meaningful. You write these; Claude Code builds the tooling.

**Concepts to learn.**
- Test-collection design (the Cranfield paradigm): fixed questions + relevance judgments as a reusable instrument.
- Pooling bias: judgments made by looking at one system's results favour that system.
- Question taxonomy (lookup, multi-hop/multi-document, conversational, unanswerable) and what each type tests.
- Why gold should be robust to re-chunking.

**Build (Claude Code).**
- Schema (`eval/questions.jsonl`): id, type, question, conversation_id + turn, oracle standalone rewrite (follow-ups), answerable flag, gold list of `{paper / work identifiers, section, verbatim span}`, `duplicate_sensitive` flag, notes on known restatements.
- Span-finder helper: grep/fuzzy search over parsed text to locate spans for questions you've already written. Not the retriever, no ranking, so no pooling bias.
- Validator: every gold span maps to at least one chunk; unmapped spans listed; type mix checked.

**Question mix (about 36).** ~12 specific lookups (several answered by tables); ~8 multi-paper; ~8 follow-up turns across 3–4 conversations, each with an oracle rewrite; at least 6 unanswerable (in-domain but not covered, false premise, paper outside the corpus, a number that doesn't exist); several duplicate-sensitive questions.

**Tests.** Schema validation; spans map; type counts meet the mix.

**You do.** All of it. Write questions from the papers first, then locate spans. If an LLM drafts candidates, you verify and rewrite each, and say so in the write-up.

**Understand.**
- Why write questions before seeing retriever output?
- Why are spans more robust than chunk IDs as gold?
- What makes an unanswerable question a good test rather than an easy one?

**Explore** (run in Phase 6).
- ★ **Do synthetic evals agree with hand-written ones?** Generate a few hundred questions from random chunks with an LLM. Rank all retrieval configurations on both sets and measure rank agreement (Kendall's tau). If they agree, synthetic sets are a cheap way to get statistical power; if they don't, you've learned something about what synthetic questions miss. Either answer is interesting.

**Exit.** Validator passes; eval set committed.

---

## Phase 4: Baseline retrieval and eval harness

**Goal.** A first working retriever and a harness + report tool that scores any configuration the same way. This produces the "before" numbers.

**Concepts to learn.**
- BM25: term-frequency saturation (k1), document-length normalisation (b), IDF. Why it's still hard to beat on exact terms.
- Dense bi-encoders: contrastive training, cosine similarity, why query and passage are embedded independently.
- Exact vs approximate nearest-neighbour search, and why exact is right at this scale.
- Reciprocal Rank Fusion: why rank-based fusion avoids score normalisation; what the k constant (commonly 60) does.
- Metrics: Recall@k, MRR, nDCG; paired bootstrap confidence intervals.
- *Probe:* compute BM25 by hand for one query on three chunks and check against the library; vary k1 and b and watch rankings change. Run RRF on a toy example with k = 1 vs 60.

**Build.**
- **Dense index**, one general model (for example `BAAI/bge-small-en-v1.5` with its query instruction), exact search (numpy or FAISS flat).
- **BM25 index** (`bm25s` or `rank_bm25`), tokenisation that keeps terms like "SentencePiece" intact.
- **Retriever interface**: `retrieve(query, k, filters) -> ranked chunks with per-retriever scores`. Dense, BM25, hybrid via RRF.
- **Eval harness** writing `results/<run>.json`; `rag report` writing markdown tables and figures.
- **Metrics, defined in DECISIONS.md and docstrings.**
  - Recall@5: fraction of gold spans found in the top-5 chunks; also paper-level Recall@5.
  - MRR: reciprocal rank of first relevant chunk, answerable questions only.
  - Duplicate rate @5: fraction of top-5 slots whose work already appears higher, or which restate a higher result; plus share of queries with at least one redundant slot.
  - Follow-ups scored three ways: raw follow-up, system rewrite (Phase 7; stubbed now), oracle rewrite.
  - Per-question table; paired bootstrap CIs for differences between runs.
- **Duplicate ground truth** comes from trustworthy identity only (exact arXiv IDs, DOIs, Semantic Scholar links) plus your Phase 5 labelled pairs. The fuzzy resolver is being tested, so it can't be the judge.

**Tests.** Metrics against hand-computed toy examples (ties, no relevant result, multiple gold spans); RRF toy; deterministic index rebuild.

**Run and inspect.** Baseline table for dense, BM25, hybrid. `rag inspect query` on 5 questions.

**You do.** Compute Recall@5 and MRR by hand for 2 questions and check against the harness. Add Recall@10 to the harness yourself. Write predictions for Phase 5 and 6 results in the journal now, before you build them.

**Understand.**
- Why RRF instead of a weighted sum of scores?
- Find one question where BM25 wins and one where dense wins. Why?
- With about 30 answerable questions, how big a difference is meaningful?

**Explore.**
- ★ **Tokenizer-aware retrieval diagnostics** (start here, finish in Phase 9). The domain is about tokenization, and dense retrievers have tokenizers too. Measure how each embedding model's tokenizer fragments the corpus's key terms ("SentencePiece", "ByT5", "fertility", language names), tag questions by fragmentation, and compare BM25 vs dense hit rates by tag. Hypothesis: dense retrieval fails most on exactly the terms its tokenizer splits worst, which is why hybrid search helps here.

**Exit.** Baseline results and first figures committed; you can explain each number.

---

## Queued before Phase 5: chunker fixes (added 2026-10-27)

Three Phase 2 defects found by the Parse page's checks, fixed in one batch so the corpus is re-chunked and re-indexed once (~45 min dense rebuild). Start in plan mode: each fix has options to agree first.

1. **Repeated caption labels** (5,260 chunks, "Figure 2: Figure 2."): `chunker._labelled` prepends a label GROBID's caption already starts with. Skip the prefix when the caption already carries it.
2. **Reference-list text in body chunks** (~180 paragraphs, e.g. `arxiv:2507.23248v2#0034`): GROBID leaves some bibliography entries in a body section. Drop or re-route those paragraphs in `structure`, so "references are never indexed" (D-021) holds.
3. **Non-headings taken as section headings** (~689 chunks: "Fig. 1", "( a )", a formula): fold them into the previous section. The check also flags real short headings ("BPE"), so the rule needs care.

Each fix gets a unit test from a real failing case. **Exit:** the Parse page's three check counts drop (to ~0 for 1 and 2), `rag index all` rebuilt from a clean tree, snapshots regenerated, before/after counts recorded in an `exp` commit.

---

## Phase 5: Versions and near-duplicates

**Goal.** Stop duplicates crowding the top-5 without merging distinct papers, and show the effect with numbers.

**Concepts to learn.**
- Identity vs similarity: two papers can be semantically near-identical yet distinct.
- Shingling and Jaccard similarity; MinHash (the chance two MinHash values match equals the Jaccard similarity).
- LSH banding: with b bands of r rows, the candidate probability is 1 − (1 − s^r)^b, an S-curve you tune.
- Union-find clustering and its risk: transitive chains (A≈B, B≈C) can merge A and C that aren't similar.
- Maximal Marginal Relevance: trading relevance against novelty.
- *Probe:* MinHash estimate vs exact Jaccard as the number of permutations grows; plot the LSH S-curve for your chosen b and r and mark your threshold.

**Build.**
- **Index-time identity resolution** (records to `work_id`), in order of confidence: same arXiv ID; linked IDs via Semantic Scholar `externalIds`; normalised-title fuzzy match + author overlap + year proximity (`rapidfuzz`); MinHash/LSH on full-text shingles (`datasketch`) as confirmation, not sole signal.
- Union-find with a cluster-level sanity check against transitive chaining. Each cluster gets a `work_id`, a canonical member (rule recorded), and aliases kept for citation.
- Index only canonical members' chunks; aliases in metadata.
- **Surveys and restatements (not merged).** Flag surveys (title cues + S2 publication types). Mark a survey chunk as restating work W when its in-text citation points to W and it is highly similar to a W chunk.
- **Query-time collapse.** Per-work cap in the top-5 (config), demotion of restating chunks when the restated work is present, optional MMR.
- **Labelled pairs** (`eval/pairs.jsonl`): candidates sampled across similarity bands (very high, borderline, topically similar but distinct); label 40–60; report resolver precision and recall.
- **Edge-case policy** in DECISIONS.md: workshop vs extended journal version, same-author follow-ups, companion papers.

**Ablation table.**

| Condition | Dup@5 | Recall@5 | MRR |
|---|---|---|---|
| No handling (Phase 4 baseline) | | | |
| Exact-ID merge only | | | |
| + fuzzy metadata + MinHash merge | | | |
| + query-time per-work cap | | | |
| + survey demotion / MMR | | | |

**Tests.** Synthetic records: v1/v3 merge; preprint vs venue with changed title merges; same-author follow-up does not; two different fertility papers do not; transitive chain caught; MinHash above/below threshold; per-work cap.

**Run and inspect.** Cluster size distribution; every multi-member cluster from fuzzy/MinHash steps listed for review; `rag inspect work` on borderline merges. Knobs: title threshold, author overlap, MinHash threshold, LSH b/r, per-work cap, MMR lambda, survey demotion.

**You do.** Label the pairs. Review every fuzzy merge and record false merges honestly. Change the per-work cap from 1 to 2 yourself and re-run.

**Understand.**
- Why is semantic similarity not identity? Use the fertility example.
- Why handle versions at index time but restatements at query time?
- How can union-find merge two papers that aren't similar, and how did you guard against it?
- Why can't the resolver's own clusters be the ground truth for dup@5?

**Explore.**
- ★ **Finding-level provenance.** Use in-text citation markers to link survey sentences to the primary work they summarise. In answers, cite the primary source and mention the survey as secondary ("A shows X [A §5]; summarised in Survey B [B §3]"). Measure how often a survey chunk in the top-5 is replaced or joined by its primary source.
- ★ **What merging hides.** For clusters with several versions, diff the numbers in abstracts and results tables across versions. How many papers changed their reported results between v1 and the final version? Surface it in answers when relevant ("v1 reported X; the final version reports Y"). This turns the merge policy's risk into a feature and a finding.

**Exit.** Ablation table and resolver precision/recall committed; chosen configuration recorded.

---

## Phase 6: Embedding comparison and hybrid search

**Goal.** Choose an embedding setup with evidence, under the deployed configuration (Phase 5 dedup on), so the comparison tests the decision being made.

**Concepts to learn.**
- Bi-encoder vs cross-encoder: independent vs joint encoding, and the recall/precision two-stage pattern.
- Asymmetric retrieval and instruction prefixes (`query:` / `passage:`, BGE's query instruction): why they exist and what happens if you forget them.
- Domain-specific embeddings: how SPECTER-style models are trained on citation links, and why that optimises paper-to-paper similarity rather than question-to-passage retrieval.
- Public benchmark scores vs your own eval: why they can disagree.
- CPU latency and throughput: batch size, sequence length, model size.
- *Probe:* cosine similarities between "SentencePiece" / "sentence piece" / "Unigram LM" / "BPE" across models; Recall@5 with prefixes on vs off.

**Build.**
- Models (verify loading and prefixes on each model card): general small (`BAAI/bge-small-en-v1.5`), general base (`BAAI/bge-base-en-v1.5` or `intfloat/e5-base-v2`), scientific (SPECTER2 with the ad-hoc query adapter for queries and the proximity adapter for documents), optional newer small model if CPU time allows.
- Conditions: each model dense-only and hybrid; BM25-only; best + cross-encoder reranker (`BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`) over the top 30–50.
- Measure Recall@5, MRR, dup@5, corpus embedding time, query latency p50/p95 on CPU, index size, dimensions.
- Paired bootstrap CIs and per-question wins/losses between the top two configurations.

| Config | R@5 | MRR | Dup@5 | Embed time | p50 latency | Dims |
|---|---|---|---|---|---|---|

**Tests.** Prefix formatting per model; every index covers the same chunk set.

**You do.** Interpret the table and write the recommendation yourself, including when you'd choose differently (for example, if latency mattered more). Compare with the predictions you wrote in Phase 4.

**Understand.**
- Why does keyword search help on this corpus in particular?
- What does a reranker do that a bi-encoder can't, and what does it cost?
- Two models differ by 0.03 Recall@5 with overlapping CIs: what do you recommend and why?

**Explore.**
- **Reranker depth sweep.** Rerank top-10/20/50/100: plot Recall@5 against latency. Shows where extra reranking stops paying.
- **Adaptive fusion.** Do lookup questions want more BM25 weight and conceptual ones more dense weight? Try per-type fusion weights and see whether a simple router beats fixed RRF.
- Run the Phase 3 synthetic-eval agreement experiment here.

**Exit.** Recommendation in DECISIONS.md; config updated.

---

## Phase 7: Grounded multi-turn chat

**Goal.** A CLI chatbot that resolves follow-ups, cites paper and section for every claim, and says plainly when the corpus doesn't support an answer.

**Concepts to learn.**
- Conversational query rewriting ("condense question"), and alternatives like HyDE (embedding a hypothetical answer).
- Context assembly: ordering, how much to include, and the "lost in the middle" effect.
- Grounding vs correctness: an answer can be faithful to bad context, or correct but unsupported.
- Citation by ID: letting code, not the LLM, produce paper and section names.
- Selective answering: abstaining as a deliberate trade-off between coverage and risk.

**Build.**
- **LLM client abstraction** chosen in config: Anthropic API, a local model via Ollama, or an OpenAI-compatible free tier; a mock client for tests.
- **Query rewriter**: follow-up + recent history to a standalone query; skipped on the first turn; always logged.
- **Retrieval** with the Phase 6 config; optional year/venue filters.
- **Context assembly**: chunks labelled `[C1]`, `[C2]`, ...
- **Generation prompt**: answer only from context, cite a chunk ID after each claim, say the corpus doesn't support it when that's the case.
- **Citation post-processing**: code maps IDs to "Title (Year), §Section"; cited IDs not in context are rejected or flagged.
- **Abstention, two layers.** Retrieval gate on the reranker score or top dense cosine (not RRF scores, which are rank-based and not comparable across queries), then the model's explicit "not supported" path.
- **CLI**: `rag chat` with conversation state, `--show-sources`, and a debug view showing the rewrite and retrieved chunks.

**Tests.** Rewriter on fixtures with the mock client; citation parsing and mapping; invalid-citation detection; abstention gate; conversation state across turns.

**You do.** Change the system prompt and record the effect in the journal. Add a year filter flag to `rag chat` yourself.

**Understand.**
- Why rewrite the query instead of embedding the whole history?
- Why do citations use chunk IDs mapped by code rather than titles written by the LLM?
- Why can't you threshold on RRF scores?

**Explore.**
- ★ **Evidence for and against.** The brief's example wants counter-evidence (a compute-cost paper) next to supporting results. For evidence-style questions, decompose into aspect sub-queries (supporting results, counter-evidence, cost, fertility), retrieve for each, fuse. Measure distinct works and gold coverage on multi-paper questions vs a single query.
- **Rewrite vs HyDE** for follow-ups: retrieval quality against the oracle rewrite.

**Exit.** Working chat; LLM choice recorded in DECISIONS.md.

---

## Phase 8: Answer evaluation and failure analysis

**Goal.** Honest numbers on how the chatbot behaves, plus the required transcript and failure analysis.

**Concepts to learn.**
- LLM-as-judge and its biases (self-preference, verbosity, position); why to audit the judge.
- Claim decomposition for faithfulness (FActScore-style): scoring each claim, not the whole answer.
- Risk–coverage curves for selective answering.
- Error taxonomy: attributing a failure to a pipeline stage.

**Build.**
- Run every eval question (conversations in order) through chat; save outputs.
- **Groundedness**: answers split into claims with citations; judge labels each supported / partially / unsupported. Report supported-claim and uncited-claim rates. Judge from a different model family if possible.
- **Judge audit**: label about 20 judgments by hand; report agreement.
- **Abstention**: rate on unanswerable; false abstention on answerable; threshold sweep. Say plainly if the threshold may be overfitted to few unanswerables.
- **Follow-ups**: rewrite correctness and retrieval with system vs oracle rewrite.
- **Transcript**: one conversation of at least 5 turns with a follow-up and an unanswerable question.
- **Failure analysis**: at least 3 wrong answers, each traced to a stage with evidence from the inspect commands.

**You do.** Write the failure analysis yourself.

**Understand.**
- What are the limits of LLM-as-judge here, and how did you check?
- Which single change would most improve the system, according to your failures?

**Explore.**
- ★ **Risk–coverage curves for abstention.** Compare gates (top cosine, reranker score, top-1 vs top-5 score margin, LLM self-assessment) on one plot. Add "counterfactual unanswerables" made by perturbing answerable questions (swap the paper name, change the number asked about) to get more than a handful of negatives.

**Exit.** Result files, transcript and failure analysis committed.

---

## Phase 9: Deep dives (exploration track)

Pick 2–3 ★ ideas and take them from a small experiment to a finding. Choose the ones that surprised you most during the core track. The journal will tell you.

**Strongest candidates.**

| Deep dive | Why it's worth talking about |
|---|---|
| Tokenizer-aware retrieval diagnostics (Phase 4) | On-theme: the retriever has the same problem the literature studies. Links a measurable cause (term fragmentation) to a design choice (hybrid search). |
| What merging hides (Phase 5) | Turns the dedup policy's main risk into a finding: how often results change across versions. |
| Finding-level provenance (Phase 5) | Goes beyond "dedupe results" to attributing claims to primary sources. Directly targets the highest-weighted section of the brief. |
| Evidence for and against (Phase 7) | Directly addresses the brief's own example of what a good answer looks like. |
| Synthetic vs hand-written evals (Phase 3) | A general lesson about evaluation that applies to any RAG project. |
| Risk–coverage abstention (Phase 8) | Treats "say when you don't know" as a measurable trade-off rather than a prompt instruction. |

**Template for each deep dive** (one section in `JOURNAL.md` plus figures):
1. Question
2. Hypothesis and your prediction (written before running)
3. Method and data (including synthetic data, clearly labelled)
4. Result with figure(s) and confidence intervals where possible
5. Interpretation: what explains the result, including where the prediction was wrong
6. What it changes in the system, if anything
7. Limits: what this experiment can't tell you

**Exit.** Each chosen deep dive has a complete entry, figures, and committed code and results.

---

## Phase 10: Assignment write-up and reproducibility

**Goal.** Someone else can run it and understand why it's built this way.

**Build.**
- README: setup in a few commands (`uv sync`, start GROBID, `rag collect`, `rag parse`, `rag index`, `rag eval`, `rag chat`), a `--limit` quick start and expected runtimes.
- Write-up (2–3 pages): topic boundary and corpus stats, parsing and chunking, duplicate strategy and ablation, embedding comparison and recommendation, chat design, answer-eval results, limitations, assumptions. Deep-dive results summarised briefly.
- Final manifest with `included_in_index`.
- Fresh-clone test: clone into a new folder and follow the README exactly.

**You do.** The fresh-clone run and a read-through of the commit history.

**Exit.** Repo in submittable state.

---

## Phase 11: Journey write-up

**Goal.** A write-up in your own voice about what you built and learned, aimed at someone who builds RAG systems.

**Suggested structure.**
1. The problem, and why literature search is a harder RAG case than it looks (versions, restatements, exact terms, evidence questions).
2. The system in one diagram.
3. Three to four things that were non-obvious, each with a figure: what you expected, what you measured, why.
4. What didn't work, and what you learned from it.
5. What you'd do next.

**Sources.** `JOURNAL.md` (predictions vs results, surprises), `docs/figures/`, deep-dive entries, `DECISIONS.md`.

**Exit.** Draft written. Claude can help edit, but the observations and claims are yours.

---

## Side track: showcase UI (added 2026-10-27)

A Streamlit app (`make ui`) with one page per stage: its key numbers, how it decides, and examples drawn live from the data (D-037, D-038). Built page by page as Sharad reviews each stage, starting with Collect. Logic in `src/rag/showcase/` (tested), rendering in `ui/`. Numbers are computed from the stage outputs, never from `stats.json`; a committed snapshot (`rag showcase snapshot`) covers data a fresh clone doesn't have.

---

## Rehearsal for live changes (no AI)

Practise each of these once:
- Add a year or venue filter to retrieval and chat.
- Change the per-paper cap in the top-5 and re-run the eval.
- Swap the embedding model via config and rebuild one index.
- Change the RRF k constant and compare.
- Add a new metric (for example nDCG@5) to the harness.
- Add a collection seed query or source.
- Change chunk size, rebuild and re-evaluate on `--limit`.

---

## Open decisions

- Generation model (Phase 7): Anthropic API, local via Ollama, or a free-tier API. Pluggable, so this can wait.
- Judge model (Phase 8): ideally a different family from the generator.
- Which 2–3 deep dives: decide after Phase 8, based on what surprised you.