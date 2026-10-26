# Phase 3 teaching note: the evaluation set

Every later comparison (BM25 vs dense, dedup on vs off, one embedder vs
another) is only as good as the questions it's scored on. Phase 3 builds the
**instrument**: a question format, a way to find evidence without bias, and a
validator. The questions themselves are Sharad's to write.

The order was changed on 2026-10-26: tooling now, questions after the chat
prototype works. So this phase closes with `rag eval validate` reporting "no
questions yet".

```
eval/questions.jsonl  ──validate──>  report (errors / warnings / mix)
        ▲
   copy spans
        │
rag eval find "<text>"  <── data/parsed/chunks/chunks.jsonl
```

Code: `src/rag/eval/` (`schemas.py`, `spans.py`, `find.py`, `validate.py`, `cli.py`).
How to write questions: `eval/README.md`.

---

## 1. Test collections: the Cranfield idea

**Where:** `schemas.Question`, `eval/README.md`.

A test collection has three parts: a fixed corpus, fixed questions, and
fixed judgments of which text answers each question. Because all three are
fixed, any two systems can be scored on exactly the same task, and a
difference in Recall@5 means a difference in the systems.

This is why the question file is committed, never edited by tools, and
validated rather than generated.

## 2. Pooling bias, and why the finder doesn't rank

**Where:** `find.find`, D-024.

TREC judged only the documents that participating systems retrieved (the
"pool"). Relevant documents no system found were never judged, so they
count as non-relevant, and a new system that finds them is *penalised*.

The same trap here: if you labelled gold by looking at the retriever's top
results, gold would contain only what that retriever finds. It would then
score well and look better than a model that finds other valid evidence.

The guards:
- **Write the question first**, from the paper.
- **Find evidence with grep, not a retriever.** `find` matches text and
  lists hits in corpus order (paper, then chunk position). It never scores,
  so nothing it shows is "what a model likes".

## 3. Gold as spans, not chunk ids

**Where:** `spans.map_span`, D-022, probe `phase-3-span-robustness.py`.

The gold for a question is `{record_id, span}`: verbatim text in a named
paper. Which chunks contain that text is computed **at scoring time** and
never stored.

The probe shows why. Re-chunking the BPE paper at `target_tokens` 100, 150,
300 and 448:

| | Maps at every size |
|---|---|
| Span gold | 5 / 5 sentences |
| Chunk ids labelled at 300 | 0 / 5 sentences |

Chunk ids are positions (`#0016`), so any change to chunking (the Phase 2
chunk-size task, a Phase 6 model with another tokenizer) moves every one of
them. Stale chunk-id gold doesn't fail loudly: it scores the wrong chunks.

Spans work because of a Phase 2 property: **chunks never split a sentence**
(D-017; the only exceptions are 549 over-long sentences). So a span within
one sentence always lands whole in some chunk. Some spans map to two chunks,
because sentence overlap copies a chunk's last sentences into the next. That
is correct: either chunk answers the question.

## 4. Matching: exact after normalisation

**Where:** `spans.normalise`, `spans.suggest`, D-023.

`normalise` applies, in order:
- NFKC (ligature "ﬁ" → "fi");
- casefold;
- unified quotes and dashes;
- table pipes dropped;
- whitespace collapsed.

A span then matches a chunk if it is a substring of that chunk's text, after
both are normalised.

Why not fuzzy matching? A fuzzy threshold is one more tunable, and on short
spans ("32k merges") it can match the wrong passage. Instead:
- spans are **copied from chunk text** (the finder's excerpt), so exact
  matching works;
- when a span doesn't map, `suggest` uses `difflib` to show the closest
  passage in that paper, as a **repair hint only**. It never counts as a
  match.

Pitfall: GROBID's surviving hyphenation ("prob-lem"). A span typed from the
PDF ("problem") won't match. That's another reason to copy from the finder.

## 5. Question types, and what each one tests

**Where:** `schemas.QuestionType`, `schemas.UnanswerableKind`, D-025.

| Type | Tests | Typical failure it exposes |
|---|---|---|
| `lookup` | finding one passage (often a table) | dense models missing exact numbers/terms |
| `multi_paper` | covering evidence across papers | top-k filled by one paper or its duplicates |
| `followup` | query rewriting (Phase 7) | "that model" retrieved literally |
| `unanswerable` | abstention (Phase 7–8) | confident answers from loosely related text |

A set made only of lookups would make BM25 look great and say nothing about
rewriting or abstention. That's why the validator checks the mix.

**Follow-ups carry an `oracle_rewrite`:** the standalone question a perfect
rewriter would write. Phase 4 scores follow-ups three ways (raw, the
system's rewrite, the oracle), which separates "the rewriter failed" from
"retrieval failed".

**Good unanswerable questions are hard** (SQuAD 2.0): in-domain, and close
to something the corpus does say. The `unanswerable_kind` values name the
recipes:
- `false_premise`: assumes something the papers contradict;
- `nonexistent_detail`: asks for a number no paper reports;
- `outside_corpus`: asks about a real paper we didn't collect;
- `not_covered`: on topic, but no paper answers it.

"What is the capital of France?" tests nothing, because retrieval scores
are low anyway.

## 6. Duplicate-sensitive questions

**Where:** `spans.elsewhere`, the duplicate warning in `validate._check_gold`.

The BPE definition ("iteratively replaces the most frequent pair of bytes")
appears in **12 other records** besides the BPE paper:
- arXiv and ACL copies of the same paper;
- versions kept from Phase 1;
- later papers that restate it.

For such a question the top-5 can be all copies of one finding. Phase 5's
dup@5 metric measures exactly this, but only for questions flagged
`duplicate_sensitive`. The validator warns when a span appears elsewhere and
the flag is off.

## 7. What the validator checks

**Where:** `validate.validate_file`.

| Check | Level |
|---|---|
| Line parses; per-question rules (`Question._consistent`) | error, with line number |
| Unique ids | error |
| Gold paper in the manifest and `included_in_index` | error |
| Span maps to ≥ 1 chunk of its paper | error, with a `suggest` hint |
| Conversation turns are 1..n | error |
| Mix below `eval.mix_min` minimums | error |
| Span long or multi-sentence (may straddle chunks) | warning |
| `section` not in the mapped chunk's path | warning |
| Span found in other papers, `duplicate_sensitive` false | warning |

A missing file is not an error, because that's the expected state until the
questions exist.

---

## What to look at

- `uv run rag eval validate`: the have/min table.
- `uv run rag eval find "byte pair encoding" --paper acl:P16-1162`: hits by
  section, excerpts you can copy.
- `uv run rag eval find "<text>"` without `--paper`: how many papers restate
  the same thing. A good way to spot duplicate-sensitive questions.
- `docs/figures/phase3-span-robustness.md`: the probe table.

## Common pitfalls (summary)

| Pitfall | Guard here |
|---|---|
| Gold labelled from retriever output (pooling bias) | write questions first; unranked `find` |
| Gold goes stale when chunking changes | spans mapped at scoring time, never stored |
| Span typed from the PDF doesn't match the parsed text | copy from `find`; `suggest` hint |
| Span crosses chunks | one-sentence spans; warning above `max_span_words` or on sentence breaks |
| Easy unanswerables | `unanswerable_kind` recipes, in-domain only |
| One-type eval set | mix minimums checked |
| Duplicates inflate or deflate scores unnoticed | `elsewhere` warning + `duplicate_sensitive` |
