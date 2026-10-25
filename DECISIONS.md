# Decisions

One entry per significant choice. Claude Code drafts entries as `proposed`;
Sharad reviews and marks them `accepted` (or rewrites them). Entries are never
deleted: a changed decision gets a new entry, and the old one is marked
`superseded by D-NNN`.

## Template

```
## D-NNN: <title>
- Date / Phase:
- Status: proposed | accepted (Sharad, YYYY-MM-DD) | superseded by D-NNN
- Context: what forced a choice
- Options considered:
  1. <option>: pros / cons
  2. <option>: pros / cons
  3. <option>: pros / cons
- Choice:
- Why:
- Evidence: results file / commit / probe, or "none yet: judgement call"
- Tuned on eval set?: yes / no
- Revisit if:
```

---

## D-001: Package and CLI layout
- Date / Phase: 2026-10-16 / Phase 0
- Status: proposed
- Context: Every phase needs a place for its code and a command to run it; PLAN.md asks for one Typer CLI with a subcommand per stage.
- Options considered:
  1. One package `rag` (src layout); each stage's command lives in `rag/<stage>/cli.py`, and `rag/cli.py` only registers them. Pros: stage code and its CLI are side by side; the root file stays a readable table of contents. Cons: one extra file per stage.
  2. All commands in one `cli.py`. Pros: one file to read. Cons: grows with every phase and mixes all stages' imports.
  3. Distribution-named package `tokenizer_papers_rag`. Pros: uv's default. Cons: long imports, and the name differs from the `rag` command.
- Choice: 1. Distribution name `tokenizer-papers-rag`, import name and command `rag` (via `tool.uv.build-backend.module-name`).
- Why: Matches PLAN.md's stage list one to one; each phase touches mostly its own folder.
- Sub-choice: the config is loaded **lazily** (the root callback stores the path; `AppState.config()` loads it on first use). Click runs the root callback even for `rag <cmd> --help`, so loading eagerly would break help outside the repo root. A bad config still fails at the start of any real command. Alternatives were eager loading in the callback, or each command calling `load_config` itself (repeats the `--config` handling in every command).
- Evidence: none yet: judgement call. `tests/test_cli.py::test_help_works_for_every_command` runs help from an empty directory.
- Tuned on eval set?: no
- Revisit if: a stage needs more than one command (then give it a sub-Typer like `inspect`).

## D-002: Config format and strict validation
- Date / Phase: 2026-10-16 / Phase 0
- Status: proposed
- Context: Every tunable value must live in `config.yaml`, and every results file must record the config that produced it.
- Options considered:
  1. YAML + pydantic models with `extra="forbid"`; sections added per phase when decided. Pros: typos fail loudly; no invented values. Cons: config.yaml grows phase by phase, so the final shape isn't visible yet.
  2. Full skeleton for all phases now. Pros: the whole shape is visible. Cons: full of placeholder numbers chosen before any evidence, which defeats "decisions backed by measurements".
  3. pydantic-settings with env-var overrides. Pros: easy overrides. Cons: a second source of truth that results files would also need to record.
- Choice: 1.
- Why: A silently ignored key (`chunk_szie: 300`) would make a run use the default while the config claims otherwise, which is the worst kind of eval bug.
- Related: `config_hash` hashes everything except `paths`. Resolved paths are absolute and differ between clones, but where files live doesn't change what a run computes.
- Evidence: `tests/test_config.py`, `tests/test_provenance.py`.
- Tuned on eval set?: no
- Revisit if: we need per-run overrides for sweeps (then add a `--set key=value` that is recorded in the results file).

## D-003: Stub commands exit 0 and print their contract
- Date / Phase: 2026-10-16 / Phase 0
- Status: proposed
- Context: Commands exist before their stages are built.
- Options considered:
  1. Print `[stub] rag <cmd> (Phase N): reads -> writes, limit=N` and exit 0. Pros: `make smoke` chains the full pipeline now and shows the stage contracts. Cons: a stub "succeeds" without doing anything.
  2. Exit non-zero "not implemented". Pros: can't be mistaken for success. Cons: smoke can't run past the first stub.
- Choice: 1. The `[stub]` prefix makes it obvious in output.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: never needed after Phase 7, when every stub is gone.

## D-004: Tooling: ruff now, type checker later
- Date / Phase: 2026-10-16 / Phase 0
- Status: proposed
- Context: Type hints are required everywhere; the question is whether to enforce them yet.
- Options considered:
  1. ruff (lint + format) only; add mypy when real logic lands. Pros: little friction while there is almost no logic. Cons: hints are unchecked for now.
  2. ruff + mypy from day one. Pros: hints checked from the start. Cons: setup cost and stub-package noise before there's logic worth checking.
  3. No tooling. Cons: style drift, and the commit history gets noisy.
- Choice: 1. Python pinned to 3.11 (`.python-version`), the declared floor, so 3.12-only features can't slip in.
- Update (2026-10-23, Phase 1): `requires-python` raised to `>=3.11.4`, because `acl-anthology` declares `>3.11` and uv can't resolve it against a range that includes 3.11.0. The local interpreter is 3.11.15, so nothing changes in practice.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: Phase 1 adds API clients and schemas. That's a good point to add `mypy --strict` on `src/`.

## D-005: Topic boundary (what counts as a tokenizer paper)
- Date / Phase: 2026-10-23 / Phase 1
- Status: proposed
- Context: The relevance filter needs an explicit rule written before any filtering code, so the rule decides the corpus rather than the code's quirks.
- Boundary (from PLAN.md, verbatim): include papers where tokenization is a primary contribution or a main experimental variable (BPE, WordPiece, Unigram/SentencePiece, byte/character-level and tokenizer-free models, vocabulary size, multilingual fairness and fertility, tokenization effects on arithmetic, code, reasoning). Exclude papers that only use a tokenizer.
- Inclusion rule in Sharad's words: TODO(sharad)
- How it is applied: keyword rules on title + abstract (accept / reject / ambiguous, each with a reason code), then embedding similarity to this boundary text for the ambiguous ones (D-008).
- Modality scope (decided by Sharad, 2026-10-23): **text only**, meaning natural language and code. Other uses of "tokenization" are out even with a tokenizer term in the title. That covers images, video, speech units, audio, proteins/DNA/molecules, graphs, recommendation and locations. Options considered: text only (chosen; focused, matches the brief), text + biology sequences (shared algorithms, different questions), any modality (much larger, incoherent corpus). Implemented as title exclusion terms, reason `off_topic_term`.
- Why it was needed: the first full discovery (19k candidates) showed "tokenization" is now a generic ML word. In a random 25 of the title-term accepts, about 7–8 were protein, graph, vision, recommendation or location papers.
- Evidence: the audit sample (precision and rough recall), to be added after labelling.
- Tuned on eval set?: no
- Revisit if: the audit shows systematic misses in one sub-topic.

## D-006: HTTP layer: hand-written polite client on httpx
- Date / Phase: 2026-10-23 / Phase 1
- Status: proposed
- Context: Four sources with different rate rules (arXiv 1 request / 3 s on one connection; Semantic Scholar shared unauthenticated pool; ACL and OAI with no published limit). Every response must be cached so reruns are free and reproducible.
- Options considered:
  1. Hand-written `PoliteClient` on httpx: per-host minimum interval, file cache keyed by request, retry loop with exponential backoff on 429/5xx/timeouts. Pros: every mechanism is visible and testable with a fake clock. Cons: about 120 lines to own.
  2. requests + requests-cache + tenacity. Pros: less code. Cons: cache keys, expiry and retry behaviour are hidden in library defaults.
  3. Source client libraries (`arxiv`, `semanticscholar`). Pros: quickest. Cons: rate limiting and caching are out of our control, and the library sits between us and the API we're learning.
- Choice: 1. `httpx.HTTPTransport(retries=)` only retries connection errors (per httpx docs), so the 429/5xx retry loop is ours anyway.
- Evidence: none yet: judgement call. Tests cover spacing, cache hits and retries.
- Tuned on eval set?: no
- Revisit if: we add many more sources.

## D-007: ACL Anthology via the official library
- Date / Phase: 2026-10-23 / Phase 1
- Status: proposed
- Context: ACL venue versions matter for Phase 5, and some tokenization papers (workshops, LREC, shared tasks) are not on arXiv.
- Options considered:
  1. `acl-anthology` library, scanning titles and abstracts locally. Pros: finds ACL-only papers; clean typed records. Cons: ~120 MB git clone; requires Python >3.11 (see D-004 update).
  2. ACL records only via Semantic Scholar `externalIds.ACL`. Pros: no new dependency. Cons: ACL-only papers that S2 doesn't link are missed.
  3. BibTeX export (`anthology+abstracts.bib.gz`, ~42 MB) plus a parser. Pros: one file. Cons: undocumented export; we'd write and test a BibTeX parser.
- Choice: 1, cloned to `data/raw/acl-anthology/` (gitignored) via `Anthology.from_repo(path=...)`.
- Licence: the library has no licence field. The Anthology's policy: material from 2016 on is CC BY 4.0, earlier material CC BY-NC-SA 3.0. Recorded as `licence_source=acl_year_rule`.
- Evidence: none yet: judgement call. The probe's source-overlap figure will show what ACL adds.
- Tuned on eval set?: no
- Revisit if: ACL contributes almost nothing beyond arXiv.

## D-008: Second-stage relevance judge: embedding similarity
- Date / Phase: 2026-10-23 / Phase 1
- Status: proposed (judge type chosen by Sharad, 2026-10-23)
- Context: Keyword rules can't settle papers that mention tokenization without studying it.
- Options considered:
  1. Embedding similarity of title + abstract to the D-005 boundary text with `BAAI/bge-small-en-v1.5`. Pros: CPU, no key, deterministic. Cons: one threshold, a blunt instrument near the boundary.
  2. LLM with the written rubric. Pros: best on borderline cases, readable reasons. Cons: key, cost, non-deterministic.
  3. Manual review queue. Pros: most honest. Cons: hundreds of papers to read.
- Choice: 1. No query instruction (the model card reserves it for short retrieval queries; this is passage-to-passage similarity), `normalize_embeddings=True`, so cosine = dot product.
- Threshold: starts at a value set by eye on a handful of clearly-in and clearly-out abstracts; the audit then measures it. Any change after seeing audit labels will be recorded here.
- Calibration (2026-10-23): first 60 results of three seed queries (`subword`, `fertility_fairness`, `tokenization_effects`), 180 papers. Keyword rules: 88 title_term, 60 no_topic_term, 31 ambiguous, 1 off_topic_term.
  - Title-accepted papers score a median cosine of 0.793 (minimum 0.669).
  - The 31 ambiguous papers span 0.652–0.832 with **no clean gap**. Clearly in-scope papers mostly sit high (*Broken Tokens?* 0.817, *BrahmicTokenizer-131K* 0.799, *Bolmo: Byteifying…* 0.794), but in- and out-of-scope papers interleave between 0.73 and 0.75: *Optimizing LLMs for Italian: Reducing Token Fertility* 0.743 (in) sits next to *OncoNoteBERT* 0.744 and *AudioCodecBench* 0.733 (out).
  - Threshold set by eye to **0.745**. Expect errors in both directions near it.
  - Model check: bge-small in sentence-transformers 6.2 defines no default prompt (`prompts: {query: '', document: ''}`), so no instruction is applied silently.
- Passing-mention rule (Sharad chose "stricter evidence first", 2026-10-23): before the judge, an abstract-only match needs topic terms in at least 2 sentences (`min_abstract_sentences_with_term`), else reject `passing_mention`. Options considered: this rule (chosen), raising the threshold only (blunt, given no clean gap), rejecting all ambiguous (would lose real papers whose titles don't say "tokenizer", e.g. *Bolmo: Byteifying…*). On the 18,826 resolved records this cut the judge's input from ~4,000 to 908 (2,234 passing mentions). The threshold stays 0.745: it was set before this rule existed and has not been re-tuned.
- Evidence: audit precision/recall (pending).
- Tuned on eval set?: no (the audit sample is separate from the Phase 3 eval set).
- Revisit if: the audit shows most errors come from the ambiguous band.

## D-009: Keep v1 as its own record only when the abstract was revised
- Date / Phase: 2026-10-23 / Phase 1
- Status: proposed
- Context: PLAN asks for v1 of substantially revised papers, so Phase 5 can test version merging and "what merging hides".
- Options considered:
  1. Keep v1 when the v1 abstract differs from the latest one (string similarity below a config threshold). Pros: direct evidence of revision. Cons: misses papers that changed only their results tables.
  2. Keep v1 for every multi-version paper. Pros: simple. Cons: lots of trivial typo-fix versions.
  3. Keep v1 when there are ≥3 versions or the gap is ≥6 months. Pros: no extra fetch. Cons: a noisy proxy.
- Choice: 1.
- Evidence: none yet: judgement call. Stats will report how many v1s were kept.
- Tuned on eval set?: no
- Revisit if: Phase 5 finds result changes the abstract rule missed.

## D-010: The manifest records every candidate, including rejects
- Date / Phase: 2026-10-23 / Phase 1
- Status: proposed
- Context: PLAN: "records every decision in a manifest"; the audit must sample rejects.
- Options considered:
  1. Every candidate with its decision and reason code. Pros: auditable, honest. Cons: thousands of rows (rejects moved to a gzipped file if the CSV passes ~10 MB).
  2. Accepted + ambiguous only; rejects summarised in stats. Pros: small. Cons: rejects can't be audited.
- Choice: 1. Amended after the first full run (Sharad, 2026-10-23): accepted records in `manifest.csv` + `manifest.json`; rejects in `rejected.csv.gz` **without abstracts** (IDs, title, reason, matched terms and score kept, so every reject is still auditable and its abstract can be looked up by ID). `audit-sample` reads the full local records, so labels still see abstracts.
- Alternatives at the amendment: rejects gzipped with abstracts (8.7 MB, ~17 MB total, re-added as a binary blob on each rerun); accepted only, rejects as counts (smallest, but rejects can't be audited from the repo).
- Evidence: 19,044 rows came to 30 MB CSV + 42 MB JSON, almost all reject abstracts. Accepted (1,979): 3.3 MB CSV + 4.6 MB JSON. Rejects without abstracts, gzipped: 2.5 MB.
- Tuned on eval set?: no
- Revisit if: the size becomes a problem for git.

## D-011: OpenReview skipped
- Date / Phase: 2026-10-23 / Phase 1
- Status: proposed
- Context: PLAN marks OpenReview optional. Anonymous reads to `api2.openreview.net` now return a 403 challenge, so it would need an account and login.
- Choice: skip. ICLR papers usually also appear on arXiv.
- Evidence: a live anonymous request returned 403 `ChallengeRequiredError` (2026-10-23).
- Tuned on eval set?: no
- Revisit if: the external recall check finds in-scope OpenReview-only papers missing.

## D-012: Snowball one hop, both directions, from anchors only
- Date / Phase: 2026-10-23 / Phase 1
- Status: proposed
- Context: Keyword search misses papers that don't use the expected terms; citation links find them.
- Options considered:
  1. One hop, references and citations of the anchor papers. Pros: PLAN's default; finds both foundations and follow-ups. Cons: biased toward well-cited work and toward the anchors' sub-topics; highly cited anchors bring thousands of citing papers, mostly rejects.
  2. References only. Pros: small. Cons: misses everything newer than the anchors.
  3. Two hops. Pros: wider net. Cons: tens of thousands of candidates, almost all rejected.
- Choice: 1. The keyword filter runs on the title and abstract returned with each citation.
- Evidence: none yet. The source-overlap probe shows what snowballing adds.
- Tuned on eval set?: no
- Revisit if: the external recall check (★ in PLAN) shows systematic gaps.

## D-013: GROBID CRF image, not the full deep-learning image
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Context: GROBID 0.9.1 (2026-08-04) ships two images. This machine is an arm64 Mac (10 cores, 16 GB RAM, ~32 GB free disk), CPU only.
- Options considered:
  1. `grobid/grobid:0.9.1-crf`: 0.5 GB, native arm64 build, fast on CPU. Cons: lacks the most accurate deep-learning models (references, headers, some zones).
  2. `grobid/grobid:0.9.1-full`: 14.8 GB, deep-learning + CRF. Cons: amd64 only (emulated here), deep-learning models slow on CPU, half the free disk.
  3. Layout-model parsers (Marker, Nougat). Cons: need a GPU in practice; PLAN names GROBID.
- Choice: 1.
- Why: the only option that runs natively and quickly here; quality is checked by Sharad's 5-paper inspection.
- Evidence: Docker Hub tag listing and GROBID Docker docs, 2026-10-25.
- Tuned on eval set?: no
- Revisit if: the inspection shows poor section or reference extraction. CRF vs full on 20 papers is a candidate Phase 9 deep dive.

## D-014: GROBID consolidation off
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Context: `consolidateHeader` defaults to 1, which makes GROBID call CrossRef for every paper.
- Options considered:
  1. `consolidateHeader=0`, `consolidateCitations=0`. Pros: no hidden external calls; the manifest already has authoritative metadata. Cons: reference entries keep only what GROBID extracted.
  2. Defaults (header 1). Pros: richer metadata. Cons: an external call per paper that we neither throttle nor cache.
- Choice: 1.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: Phase 5 dedup needs better reference metadata (DOIs, arXiv IDs).

## D-015: PyMuPDF as the fallback parser
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Context: some PDFs come back from GROBID empty (204) or with an error (500).
- Options considered:
  1. PyMuPDF (named in PLAN). Pros: good text blocks and reading order. Cons: AGPL-3.0; if the repo ever gets a licence, it must be AGPL-compatible.
  2. pypdf (BSD-3). Pros: permissive. Cons: plainer text, worse column handling.
  3. No fallback; failures are only flagged. Pros: simplest. Cons: drops papers.
- Choice: 1. Fallback papers are marked `parser=pymupdf` and treated as lower quality (one flat section per page, no references). Pinned to `>=1.28.2,<2`: PyMuPDF 2.0 makes its layout package (onnxruntime, networkx, pymupdf4llm) a required dependency, and plain text blocks need none of it.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: licensing matters, or the fallback share is large.

## D-016: Count tokens with the embedding model's own tokenizer
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Context: chunk limits must hold for the model that embeds them; anything past its 512-token limit is silently dropped.
- Options considered:
  1. The tokenizer of `BAAI/bge-small-en-v1.5` (Phase 4 baseline). Pros: exact. Cons: limits are model-specific.
  2. tiktoken. Cons: a different vocabulary, so counts are wrong for this model.
  3. Words × 1.3. Cons: an approximation that can overflow silently.
- Choice: 1. Header tokens and the 2 special tokens count toward the limit.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: Phase 6 adds a model with another vocabulary (SPECTER2 uses SciBERT's): re-count and check truncation then.

## D-017: Pack paragraphs within a section, overlap by whole sentences
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Context: PLAN fixes "never across sections", a ~300-token target and small overlap; the packing method is open.
- Options considered:
  1. Greedy paragraph packing to `target_tokens`, long paragraphs split on sentences, overlap = previous chunk's last sentences up to `overlap_tokens`. Pros: never cuts a sentence or citation marker. Cons: chunk sizes vary.
  2. Fixed token windows with token overlap. Pros: uniform. Cons: cuts sentences and citation markers.
  3. One chunk per paragraph. Pros: precise boundaries. Cons: many tiny chunks with weak embeddings.
- Choice: 1 (target 300, max 448, overlap 40).
- Evidence: none yet; the chunk-size ablation in Phase 4 measures it.
- Tuned on eval set?: no
- Revisit if: the Phase 4 granularity ablation favours another size.

## D-018: Figure captions as their own chunks
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Options considered:
  1. Own chunks. Pros: self-contained; "Figure 3 shows …" answers live there; paragraph chunks stay focused. Cons: short chunks.
  2. Attached to the paragraph that first cites the figure. Pros: context. Cons: dilutes that chunk.
  3. Dropped. Cons: loses content.
- Choice: 1.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: caption chunks crowd the top-k with low-value hits.

## D-019: Keep short display formulas, replace long ones with a placeholder
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Context: GROBID's formula text is often garbled; long garbled formulas add noise tokens.
- Options considered:
  1. Keep ≤ `formula_max_chars` (120), else `[equation (n)]`. Pros: short formulas carry meaning, long noise is dropped. Cons: a threshold to choose.
  2. Always a placeholder. Cons: loses short, useful formulas.
  3. Always keep. Cons: noise.
- Choice: 1. Inline formulas stay in the text.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: questions about formulas fail in the eval.

## D-020: Tables as caption + markdown pipe rows
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Options considered:
  1. Caption + pipe rows, split by rows with the header row repeated. Pros: compact, readable. Cons: cells far from the header lose their column name.
  2. "header = value" records per row. Pros: each cell keeps its column. Cons: longer.
  3. One generated description per table. Cons: needs an LLM; deferred to the PLAN exploration.
- Choice: 1, as a starting point; PLAN's ★ table-representation exploration compares them once the harness exists.
- Evidence: none yet. CRF table-cell extraction is weak; the stats show how often tables have no rows.
- Tuned on eval set?: no
- Revisit if: the table exploration says so.

## D-021: `included_in_index` is written back into the manifest
- Date / Phase: 2026-10-25 / Phase 2
- Status: proposed
- Context: PLAN's Phase 2 exit asks for `included_in_index` to be filled in; stages normally write only their own folder.
- Options considered:
  1. A final `rag parse mark` step updates the committed manifest. Pros: the manifest stays the single record of the corpus. Cons: one documented exception to "write only your own folder".
  2. Status only in `data/parsed/chunks/status.jsonl`. Pros: folders stay pure. Cons: gitignored, so repo readers can't see what was indexed.
- Choice: 1. True when the paper has at least one chunk.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: later stages need finer status (e.g. dedup exclusions in Phase 5).
