# Phase 6 teaching note: picking an embedding model with evidence

Phase 6 asks which model should embed the chunks, whether keyword search
should stay next to it, and whether a reranker is worth its cost. It
measures each choice under the deployed configuration (Phase 5 duplicate
handling on).

```
rag index dense --model bge_small | bge_base | specter2   ─▶ data/index/dense/<model>/
rag eval silver                                           ─▶ eval/silver/questions.jsonl (+ meta.json)
uv run python probes/phase-6-terms.py                     ─▶ docs/figures/phase6-terms.{png,md}
rag eval ablate --name embeddings                         ─▶ results/<date>-embeddings-ablation.{json,md}

query ─▶ Retriever.search ─▶ BM25 + dense ─▶ RRF ─▶ [rerank top 50] ─▶ collapse (D-041) ─▶ top k
```

Code:
- `src/rag/index/config.py`: `DenseConfig`, `IndexConfig.dense_models`;
- `src/rag/retrieve/dense.py`: `SentenceTransformerEncoder`, `Specter2Encoder`, `make_encoder`;
- `src/rag/retrieve/specter2_worker.py`;
- `src/rag/retrieve/rerank.py`;
- `src/rag/retrieve/retriever.py`: `Retriever.search`;
- `src/rag/eval/silver.py`;
- `src/rag/eval/ablation.py`.

Decisions: D-043 (the setup, proposed) and D-044 (silver set).

Numbers (silver, 66 questions; provisional until `eval/questions.jsonl`
exists):

| Condition | Recall@5 | MRR@10 | p50 latency | Embed time | Dims |
|---|---|---|---|---|---|
| BM25 only | 0.742 | 0.729 | 14 ms | – | – |
| bge-small dense | 0.470 | 0.399 | 14 ms | 58 min | 384 |
| bge-small hybrid (deployed) | 0.636 | 0.596 | 14 ms | 58 min | 384 |
| bge-base hybrid | 0.667 | 0.605 | 25 ms | 129 min | 768 |
| SPECTER2 dense | 0.136 | 0.099 | 26 ms* | 76 min | 768 |
| SPECTER2 hybrid | 0.667 | 0.445 | 26 ms* | 76 min | 768 |
| bge-small hybrid + reranker | 0.818 | 0.795 | 3.3–4.0 s | 58 min | 384 |

\* SPECTER2's query encoding is timed inside its warm worker; a live system would
also pay the process start and model load (seconds) unless the worker stays up.

---

## 1. Bi-encoders and cross-encoders

**Bi-encoders** (bge, SPECTER2) encode the query and each passage
*separately* into vectors. A passage's vector doesn't depend on the query,
so the whole corpus is embedded once, offline (`rag index dense`). At query
time the system encodes one string and takes 92,820 dot products
(`DenseIndex.scores`). That takes milliseconds, but the model never sees the
query and the passage together. Everything it knows about their match
has to fit through one 384- or 768-number bottleneck.

A **cross-encoder** (bge-reranker-base) reads `query [SEP] passage` as
*one* input, so every query word can attend to every passage word, and it
outputs a single relevance score. It is far more precise, but nothing can be
precomputed: each (query, passage) pair is a full transformer pass. Over
92k chunks that would take hours per query.

**The two-stage pattern** combines them:
1. the cheap stage (BM25 + bi-encoder, fused) finds a short list with high recall;
2. the expensive stage reorders only that list.

Here, `rerank(ranked, query, texts, scorer, depth)` in `retrieve/rerank.py`
rescores the fused top 50 and leaves the rest below in their old order.
In `Retriever.search` the order is: fuse → rerank → collapse (Phase 5).
Reranking before collapse matters: the per-work cap should keep each
paper's *best* chunk by the better score.

**What it bought here:** Recall@5 +0.182 [0.091, 0.288] over the deployed
hybrid, 13 questions better and 1 worse. **What it costs:** about 4 s per query on CPU (50
query–passage pairs of up to 512 tokens each), against 14 ms without it.

Also worth seeing: with the reranker, bge-small and bge-base tie on every
question. Their top 5s still differ on 77 of 106 queries, but both
candidate lists contain the gold chunk, and the cross-encoder puts it at
the same rank. Once the reranker runs, the bi-encoder's only job is to
get the answer into the top 50. A smaller model does that as well as a
bigger one.

## 2. Asymmetric retrieval and the query instruction

A question ("How does SentencePiece handle whitespace?") and the passage
that answers it look nothing alike: one is short and interrogative, the
other long and declarative. bge was trained with an instruction prefixed
to *queries only*: "Represent this sentence for searching relevant
passages: ". Passages get no prefix. The prefix tells the model which
side of the asymmetry it is encoding.

Here:
- `DenseConfig.query_prompt` holds the prefix.
- `SentenceTransformerEncoder.encode_query` passes it, and `encode_documents` doesn't.
- The documents in the index carry no prefix, so turning the prompt off
  needs no re-embedding: condition `small_dense_no_prompt` simply encodes
  queries without it.

**Result:** dense Recall@5 0.470 with the prefix, 0.394 without:
+0.076 [0.015, 0.152], 6 questions better and 1 worse. Forgetting the
prefix is a silent bug. Nothing crashes, and recall just drops. The test
`tests/retrieve/test_dense.py` checks that the prefix goes on queries and
not on documents.

SPECTER2 handles the asymmetry with **adapters**, not a prefix: small
trained layers swapped into the same base model.
- Queries go through `allenai/specter2_adhoc_query`.
- Passages go through `allenai/specter2`, the "proximity" adapter.
- The input is `title [SEP] text`. That's why `DenseConfig.header_separator`
  is `[SEP]` for SPECTER2 and `\n` for bge (`retrieve/text.py:chunk_text`).

## 3. Domain models: what SPECTER2 was trained to do

"Scientific" sounds like a fit for a corpus of papers. But SPECTER2 was
trained on **citation links**: a paper should embed close to the papers it
cites. That is *paper-to-paper* similarity, i.e. "is this about the same
research topic?". Our task is *question-to-passage*: does this paragraph
answer this specific question?

The probe (`probes/phase-6-terms.py`, `docs/figures/phase6-terms.md`)
shows the consequence:
- **Term pairs:** SPECTER2 scores every pair between 0.80 and 0.98:
  - "SentencePiece / BPE" (different methods): 0.874;
  - "SentencePiece / sentence piece" (the same thing): 0.976;
  - "fertility / birth rate": 0.905.
  Everything in the field looks related, because to a citation model it is.
  bge separates different methods more (SentencePiece / BPE 0.53).
- **Exact terms in the top 5:**
  - SPECTER2 dense, 0.28 of the top 5;
  - bge-small, 0.60;
  - BM25, 0.68.
  SPECTER2 returns chunks on the general topic, not the named method.
- **On silver:**
  - SPECTER2 dense alone: Recall@5 0.136.
  - In hybrid, BM25 rescues recall (0.667), but MRR falls by 0.151
    [-0.265, -0.035] against bge-small hybrid. Its dense list pushes
    loosely related chunks up through RRF.

A domain model wins when its training objective matches your task, not
when its training data matches your corpus. SPECTER2 would be the right
tool for "find papers like this one".

Practical costs show up too:
- **Separate environment:** `adapters` pins transformers 4.57 while the
  project uses 5.x, so SPECTER2 runs in its own `uv run --isolated`
  environment (`Specter2Encoder._call` → `specter2_worker.py`).
- **Truncation:** SciBERT's tokenizer cut 44 chunks at 512 tokens.
- **Ranking:** its card ranks by L2 distance while we use cosine. The probe
  shows the top 10s overlap 92% on 2,000 sampled chunks, so normalising
  barely changes rankings.

## 4. Why keyword search helps on this corpus

Tokenizer papers name things:
- methods: SentencePiece, BPE-dropout, MaxMatch;
- models: ByT5, CANINE;
- languages and datasets;
- coined terms: fertility, Parity-aware BPE.

A dense model maps text to *meaning*, so a rare name it never saw in
training gets blurred toward its nearest familiar neighbours. BM25 matches
the *string*, and a rare string has a high IDF, so it counts for a lot
(`retrieve/bm25.py`).

- Hybrid vs dense (bge-small): +0.167 [0.091, 0.258], 11 questions better
  and 0 worse.
- In the probe, BM25 is the only retriever that finds CANINE, Hindi and
  Parity-aware BPE in all of its top 5.

**But on silver, BM25 alone beats every hybrid** (−0.106 for bge-small
hybrid vs BM25). Don't read that as "drop the dense model" yet:
- The silver questions were written *from* their passage. Their content
  words overlap it at a median of 0.75 (D-044), which is ideal for BM25.
- Real questions paraphrase, and that is where dense retrieval earns its
  place.

This is the comparison most likely to change on gold.

## 5. A silver set, and what it can and can't tell you

With no gold questions yet, `rag eval silver` had qwen3:8b write one
question per sampled chunk:
- one chunk per canonical work;
- seed 13;
- temperature 0, so generation repeats: all 66 also appear, word for
  word, in an earlier run with a looser filter (which kept 75).

Filters in `silver.py:make_question` drop questions that point at their
source ("the paper", "the table", "the proposed…", `SOURCE_REFERENCE`) and
evidence that doesn't map to the chunk (`spans.map_span`). 66 of 90 survived.

Know its biases before trusting a number (D-044):
- **one gold chunk** per question, so another passage that also answers it
  counts as a miss;
- **lookups only**;
- **passage wording**, which favours BM25;
- **one LLM's idea** of a question.

Differences between conditions are more trustworthy than absolute levels,
and only where the paired interval excludes 0.

## 6. Reading the comparison: paired differences

Two conditions answer the *same* questions, so compare them question by
question.
- `ablation.paired` takes a − b per question and bootstraps the mean of
  those differences (`metrics.paired_bootstrap_diff`).
- It also counts how many questions each wins.
- `eval.ablations.embeddings.pairs` in `config.yaml` names the comparisons
  the decision rests on.

**Overlapping intervals:** bge-base hybrid vs bge-small hybrid is +0.030
[-0.045, 0.121], 5 questions better and 3 worse. The interval contains 0, so 66 questions can't
separate them. Pay 2.2× embedding time, 2× disk and 2× latency for a
difference you can't measure? The default should be the cheaper model,
until more questions show a gap.

## 7. CPU latency and throughput

What sets the cost:
- **Model size:** bge-base is 2× bge-small's dims and ~3× its parameters;
  embedding took 129 vs 58 min, and query p50 is 25 vs 14 ms.
- **Sequence length:** attention cost grows with tokens². Padding a batch
  to its longest member wastes compute, so sentence-transformers sorts by
  length. `specter2_worker.py` does the same (longest first, rows put back
  in order).
- **Batch size:** bigger batches use the CPU's vector units better, up to memory.
- **Memory pitfall** (from this phase): `last_hidden_state[:, 0, :].numpy()`
  is a *view*, which keeps each batch's whole `32 × 512 × 768` tensor alive.
  The first SPECTER2 build was killed at 17k of 92k chunks. `.copy()`
  keeps only the CLS rows.
- **Reranker:** cost is linear in depth. PLAN's explore item (rerank top
  10/20/50/100, Recall@5 vs latency) shows where extra depth stops paying.

## 8. Public benchmarks vs your own eval

bge-base beats bge-small on MTEB retrieval (53.25 vs 51.68 on the
bge-small-en-v1.5 model card), and SPECTER2
is the reference model for scientific-document benchmarks (SciRepEval).
Neither ranking held here:
- MTEB averages many domains and query styles, none of them "questions
  about tokenizer papers with exact method names";
- SciRepEval's tasks are mostly paper-level (citation prediction,
  classification), not passage QA.

A benchmark tells you which models to try. Your own eval, on your corpus
and your questions, tells you which to pick.

## What to look at

- **The table:** `results/2026-10-29-embeddings-ablation.md`, with the
  paired comparisons at the bottom. Per-question rows, including the top 10
  chunk ids, are in the JSON.
- **One query across models:** `uv run rag inspect query "How does
  SentencePiece handle whitespace?" --dense specter2` (or `bge_base`).
  Compare the `dense` rank column with `bm25`.
- **The probe:** `docs/figures/phase6-terms.png`.
- **The silver questions:** `eval/silver/questions.jsonl`. `notes` holds the
  source chunk and the word overlap. Read 10 and ask whether a researcher
  would phrase them that way.
- **UI:** the "4 · Pick an embedding model" page (`make ui`). Its "Try it"
  section runs one query through several retrievers side by side and marks
  results containing an exact term.

## Common pitfalls

| Pitfall | Guard here |
|---|---|
| Query prefix on documents too, or forgotten on queries | `encode_query` vs `encode_documents`; test in `tests/retrieve/test_dense.py`; condition `small_dense_no_prompt` measures it |
| Comparing models on different chunk sets | every index's `meta.json` records the chunks sha256; `Retriever.load` refuses a mismatch |
| A new field in `DenseConfig` making old indexes "stale" | `DenseConfig.settings()` drops defaults |
| Picking a model from leaderboard scores | measured on this corpus |
| Reading a 0.03 gap as a win | paired bootstrap; interval includes 0 → no measurable difference |
| Trusting silver absolute numbers | labelled silver everywhere, never pooled with gold (D-044) |
| SPECTER2's dependency pins breaking the project | isolated `uv run` environment, worker imports nothing from `rag` |
| Holding onto tensors via numpy views | `.copy()` the rows you keep |
| A background job hanging on a network call after the laptop sleeps | `HF_HUB_OFFLINE=1` (models cached) and `caffeinate -i` for long runs |
| Changing the model without the downstream thresholds | the chat gate (0.70, D-036) is a bge-small cosine; `rag dedup run` uses the dense vectors |
