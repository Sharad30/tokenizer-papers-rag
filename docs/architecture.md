# Architecture: the blocks and how they connect

The system has three parts:

- an **offline pipeline** that builds the corpus and the indexes once;
- an **eval loop** that scores retrieval against gold spans;
- the **online chat**, which answers one question at a time.

Each block reads only the previous block's output folder and writes its
own, plus a `stats.json`. Every setting is in `config.yaml`.

```
 OFFLINE PIPELINE (run once; rerun when a stage changes)
 ═══════════════════════════════════════════════════════

  arXiv API    Semantic Scholar    ACL Anthology
      └──────────────┼───────────────────┘
                     ▼
 ┌─────────────────────────────────────────┐
 │ 1 COLLECT                rag collect    │  discover → resolve → filter →
 │   src/rag/collect/                      │  licence → download → manifest
 └────────────────────┬────────────────────┘
                      │ data/manifest/manifest.json (every candidate + decision)
                      │ data/raw/pdfs/
                      ▼
 ┌─────────────────────────────────────────┐
 │ 2 PARSE + CHUNK          rag parse      │  grobid → structure → chunk → mark
 │   src/rag/parse/      (GROBID in Docker)│
 └────────────────────┬────────────────────┘
                      │ data/parsed/chunks/chunks.jsonl  (92,820 chunks)
                      │ data/parsed/papers/  (sections, references)
                      ▼
 ┌─────────────────────────────────────────┐
 │ 5 DEDUP  (not built yet)  rag dedup     │  records → works (versions,
 │   src/rag/dedup/                        │  venue copies, restatements)
 └────────────────────┬────────────────────┘
                      │ data/works/   (today the indexes read chunks directly)
                      ▼
 ┌─────────────────────────────────────────┐
 │ 4 INDEX                  rag index      │
 │   src/rag/index/                        │
 │   ├─ BM25   (bm25s, our tokeniser)      │──▶ data/index/bm25/
 │   └─ dense  (bge-small-en-v1.5, CPU)    │──▶ data/index/dense/bge-small-en-v1.5/
 └────────────────────┬────────────────────┘    (meta.json: chunks sha256 + settings)
                      │
                      ▼
 ┌─────────────────────────────────────────┐
 │ RETRIEVER   src/rag/retrieve/retriever  │  shared by eval, inspect and chat
 │   query ─┬─ BM25 top 100 ──┐            │
 │          └─ dense top 100 ─┴─ RRF (k=60)│──▶ hits, each with both ranks/scores
 │   + best BM25 / dense score in corpus   │──▶ top_scores (for the gate)
 └──────────┬───────────────────┬──────────┘
            │                   │
 EVAL LOOP  ▼                   ▼  ONLINE CHAT (one turn)
 ═══════════════════            ═══════════════════════════════════════════
 ┌──────────────────────┐   ┌──────────────────────────────────────────────┐
 │ 3 EVAL SET           │   │ 7 CHAT                       rag chat        │
 │  eval/questions.jsonl│   │   src/rag/chat/                              │
 │  (Sharad writes;     │   │                                              │
 │   gold = verbatim    │   │  user text ─▶ a REWRITE (turn ≥ 2)  LLM call │
 │   spans)             │   │                │ standalone query            │
 │  rag eval find       │   │                ▼                             │
 │  rag eval validate   │   │              b RETRIEVE hybrid, k = 6        │
 └──────────┬───────────┘   │                │                             │
            ▼               │                ▼                             │
 ┌──────────────────────┐   │              c GATE 1: best dense cosine     │
 │ 4 HARNESS            │   │                < 0.70 ? ──yes──▶ abstain     │
 │  rag eval run        │   │                │ no                          │
 │  span → chunk map,   │   │                ▼                             │
 │  Recall@5, MRR@10,   │   │              d CONTEXT [C1]..[C6], best first│
 │  dup@5, bootstrap    │   │                │ (no chat history)           │
 └──────────┬───────────┘   │                ▼                             │
            │ results/*.json│              e GENERATE    LLM call          │
            ▼               │                │ qwen3:8b (Ollama) or Haiku  │
 ┌──────────────────────┐   │                ▼                             │
 │ REPORT   rag report  │   │              f GATE 2: NOT_SUPPORTED ?       │
 │  tables, paired      │   │                │ no      (yes ─▶ abstain)    │
 │  diffs → docs/figures│   │                ▼                             │
 └──────────────────────┘   │              g CITATIONS by code: renumber,  │
                            │                drop invalid, Sources from    │
                            │                metadata, count uncited       │
                            │                │                             │
                            │                ▼                             │
                            │   answer + Sources ─▶ terminal               │
                            │   whole turn ───────▶ data/chat/<ts>.jsonl   │
                            └──────────────────────────────────────────────┘

 Cross-cutting: src/rag/common/ (config, io, provenance, stats, logging),
                rag inspect paper | query (look at any stage), probes/ (one
                concept experiment per phase), prompts/ (editable LLM prompts)
```

Block numbers are the PLAN.md phases. Phase 6 (embedding comparison +
hybrid tuning) changes the **INDEX** and **RETRIEVER** blocks. Phase 8 adds
answer evaluation that reads the chat transcripts.

---

## The blocks, in reading order

### 1. Collect: which papers are in the corpus
- **Does:** finds candidates three ways (arXiv seed queries, an ACL scan, a
  one-hop Semantic Scholar snowball from anchor papers), links them to S2
  ids, filters by keyword rules and then an embedding judge, records
  licences, downloads PDFs.
- **Code:** `src/rag/collect/` (`pipeline.py`, one function per step).
- **Output:** `data/manifest/manifest.json|csv`: every candidate, with
  `relevance`, `reason`, `found_via`, ids and licence.
- **Decisions:** D-005 to D-012. **Note:** `notes/phase-1-collect.md`.
- **Look at:** `data/manifest/stats.json`,
  `docs/figures/phase1-coverage.md`, `phase1-accepted-by-year.png`,
  `phase1-discovery-routes.png`.

### 2. Parse + chunk: PDFs → section-aware chunks
- **Does:** GROBID turns each PDF into TEI XML, with a PyMuPDF fallback.
  Structure extracts sections, captions, tables and references. The chunker
  packs paragraphs within a section to about 300 tokens (max 448), with a
  40-token overlap of whole sentences.
- **Code:** `src/rag/parse/`.
- **Output:** `data/parsed/chunks/chunks.jsonl` (1,781 papers, 92,820
  chunks; the header `Title > Section` is stored separately) and
  `data/parsed/papers/references.jsonl` (never chunked).
- **Decisions:** D-013 to D-021, D-039. **Note:** `notes/phase-2-parse.md`.
- **Look at:** `uv run rag inspect paper <record_id>`,
  `docs/figures/phase2-chunk-stats.md`, `phase2-dilution.md`.
- **Known issues:** the clean-up rules (D-039) fixed repeated caption
  labels, bibliography text in body sections and non-headings taken as
  headings. The residue (76 + 18 + 164 chunks) is listed on the Parse page.

### 3. Eval set: what "right" means
- **Does:** defines gold as a verbatim span in a named paper, mapped to
  chunks at scoring time, so gold survives re-chunking. `find` helps you
  copy spans, and `validate` checks the file.
- **Code:** `src/rag/eval/` (`schemas.py`, `spans.py`, `find.py`,
  `validate.py`).
- **Output:** `eval/questions.jsonl` (yours, not written yet).
- **Decisions:** D-022 to D-025. **Note:** `notes/phase-3-eval.md`.

### 4. Index + retriever + harness: finding chunks, and scoring that
- **Does:**
  - BM25 over compound-aware terms, no stemming;
  - dense bge-small vectors with exact numpy search;
  - hybrid = RRF of both top-100 lists.
  - The harness scores Recall@5, paper Recall@5, MRR@10 and dup@5, with
    bootstrap CIs.
- **Code:** `src/rag/index/`, `src/rag/retrieve/`, `src/rag/eval/harness.py`,
  `metrics.py`, `src/rag/report/`.
- **Output:** `data/index/…`, `results/<date>-<name>.json`.
- **Decisions:** D-026 to D-031. **Note:** `notes/phase-4-retrieve.md`.
- **Look at:** `uv run rag inspect query "<question>"`: both retrievers'
  ranks side by side; `docs/figures/phase4-*.md`.

### 5. Dedup (planned)
- **Does:** clusters records into works: arXiv versions, arXiv ↔ ACL
  copies, restatements.
- **Code:** `src/rag/dedup/` is a stub today.
- **Why it matters now:** duplicates take slots in the top 6, and a model
  citing two copies of one paper creates false corroboration.

### 7. Chat: a grounded answer with citations
- **Does:** steps a–g in the diagram.
- **Code:** `src/rag/chat/` (`engine.py` is the turn; `rewrite.py`,
  `context.py`, `llm.py`, `citations.py`, `transcript.py`).
- **Prompts:** `prompts/answer_system.md`, `prompts/rewrite_system.md`.
- **Output:** the terminal, and `data/chat/<timestamp>.jsonl`.
- **Decisions:** D-032 to D-036. **Note:** `notes/phase-7-chat.md`.
- **Look at:** `uv run rag chat --debug --show-sources`,
  `docs/phase7-demo.md`, `docs/figures/phase7-gate.md`.

---

## Where a bad answer can come from

| Symptom in an answer | Block to check first | How |
|---|---|---|
| A known paper never appears | 1 Collect | is it in `manifest.csv`, accepted, `included_in_index`? |
| Garbled or merged text in a source | 2 Parse | `rag inspect paper <id>` |
| The right paper is indexed but not retrieved | 4 Retriever | `rag inspect query`: BM25 vs dense ranks |
| The same paper cited twice | 5 Dedup | the identity of the two record ids |
| A follow-up retrieves the wrong topic | 7a Rewrite | `--debug` shows the standalone query |
| A wrong refusal, or a confident answer to a false premise | 7c/7f gates | `--debug` best cosine vs 0.70; raw answer |
| A citation that doesn't support its sentence | 7e Generate | `--show-sources`; the prompt in `prompts/` |
