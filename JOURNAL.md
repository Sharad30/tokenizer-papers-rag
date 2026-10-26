# Journal

Sharad's journal: predictions, results, surprises, dead ends, questions.
Write the prediction **before** running anything. Claude Code may add entry
skeletons, but never fills in predictions, interpretations or conclusions.

## Template

```
## YYYY-MM-DD: <title>
Phase:  ·  Type: prediction | result | surprise | dead end | question

### Prediction (Sharad):

### Why I expect this (Sharad):

### What happened:
(numbers, results file path, commit)

### Explaining the gap (Sharad):

### Follow-ups:
```

---

## 2026-10-16: What I expect to be hardest, and why
Phase: 0  ·  Type: prediction

### Prediction (Sharad):

### Why I expect this (Sharad):

### What happened:
(fill in at the end of the core track)

### Explaining the gap (Sharad):

### Follow-ups:

## 2026-10-23: Corpus size and which source contributes most
Phase: 1  ·  Type: prediction

### Prediction (Sharad):
(write before reading data/manifest/stats.json or docs/figures/phase1-*)

### Why I expect this (Sharad):

### What happened:
(numbers from data/manifest/stats.json and docs/figures/phase1-coverage.md, commit)

### Explaining the gap (Sharad):

### Follow-ups:

## 2026-10-23: Relevance filter audit: precision and recall
Phase: 1  ·  Type: prediction

### Prediction (Sharad):
(precision and rough recall you expect, before labelling data/manifest/audit_sample.csv)

### Why I expect this (Sharad):

### What happened:
(your precision / recall from the labelled sample)

### Explaining the gap (Sharad):

### Follow-ups:

## 2026-10-25: Which element types parse worst
Phase: 2  ·  Type: prediction

### Prediction (Sharad):
(tables, figure captions, formulas, section headings, references: which parse worst, and why? Write before opening data/parsed/ or docs/figures/phase2-*)

### Why I expect this (Sharad):

### What happened:
(from data/parsed/*/stats.json and your 5-paper inspection below)

### Explaining the gap (Sharad):

### Follow-ups:

## 2026-10-25: Five papers' chunks next to their PDFs
Phase: 2  ·  Type: result

### Papers inspected (Sharad):
(5 random record ids; `uv run rag inspect paper <id>` next to data/raw/pdfs/<id>.pdf)

### Problems found (Sharad):
(per paper: missed or bogus headings, merged sections, broken tables, garbled formulas, lost text, wrong pages)

### What happened:

### Follow-ups:

## 2026-10-25: Changing the chunk size
Phase: 2  ·  Type: prediction

### Prediction (Sharad):
(change parse.chunk.target_tokens, re-run `rag parse chunk --limit 20`: what happens to chunks per paper and the length histogram?)

### Why I expect this (Sharad):

### What happened:
(stats from data/parsed/chunks/stats.json before and after)

### Explaining the gap (Sharad):

### Follow-ups:

## 2026-10-26: Which question type will be hardest for retrieval
Phase: 3  ·  Type: prediction

### Prediction (Sharad):
(lookup, multi_paper, followup, unanswerable: which scores worst at Recall@5 / abstention, and by how much? Write before the first `rag eval run` in Phase 4)

### Why I expect this (Sharad):

### What happened:
(per-type numbers from the Phase 4 results file, commit)

### Explaining the gap (Sharad):

### Follow-ups:
