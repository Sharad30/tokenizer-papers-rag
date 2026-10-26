# Phase 4 teaching note: baseline retrieval and the eval harness

Phase 4 turns the 93,031 chunks into something you can query, and builds
the instrument that scores any retrieval configuration the same way.

```
chunks.jsonl ─┬─ rag index bm25  → data/index/bm25/            (sparse term index)
              └─ rag index dense → data/index/dense/bge-small/ (93k x 384 matrix)
                         │
             Retriever.retrieve(query, k, mode, allow)   bm25 | dense | hybrid (RRF)
                ├─ rag inspect query      (look)
                ├─ rag eval run           (score) → results/<date>-<name>.json
                │      └─ rag report      (tables, paired differences, figure)
                └─ Phase 7 chat           (answer)
```

Code: `src/rag/retrieve/`, `src/rag/index/`, `src/rag/eval/` (`metrics.py`,
`identity.py`, `harness.py`, `run.py`), `src/rag/report/report.py`.

Your questions don't exist yet, so `rag eval run` says "no questions yet"
and writes nothing. Everything in it is tested on the synthetic fixture.

---

## 1. BM25: term matching with two knobs

**Where:** `retrieve/bm25.py`, `retrieve/text.py`, D-026, D-027,
probe `phase-4-bm25-rrf.py`.

BM25 scores a chunk by summing, over the query's terms:

```
idf(t)  = ln(1 + (N - df + 0.5) / (df + 0.5))        rare terms count more
tf part = tf / (tf + k1 * (1 - b + b * dl / avgdl))   repeats saturate; long chunks discounted
score   = Σ idf(t) * tf part
```

The probe computes this term by term for three BPE-paper chunks, and the
result matches bm25s to four decimals (2.7386, 0.7763, 0.1521).

- **k1 (1.5)** controls saturation. The 4th mention of a term is worth
  less than the 1st, and a small k1 makes it worth almost nothing.
- **b (0.75)** controls length normalisation. At b = 0, a long chunk wins
  on raw counts. In the probe's synthetic pair, b flips the winner between
  0.5 and 0.75.
- On the three real chunks, **no setting reorders them**, because one chunk
  matches far more query terms. The knobs matter for close calls, not for
  clear winners.

**Tokenisation is where BM25 succeeds or fails here** (`text.terms`):
- Terms keep word-internal `-` `.` `_`, so `gpt-2`, `bpe-dropout` and `byt5`
  stay intact and specific.
- A compound also indexes its parts (`byte-level` → `byte`, `level`), so the
  query "byte level" still matches.
- There is **no stemming**. The cost shows in the probe: "merges" scores
  nothing on chunks that say "merge". The benefit: "tokenization" and "token"
  stay distinct. Stemming is a later ablation.

**Our code does the ranking:** bm25s scores every chunk (`get_scores`), and
`ranking.top_k` sorts with ties broken by corpus position. bm25s doesn't
document its tie order, and a reproducible eval needs one.

## 2. Dense retrieval: one vector per chunk

**Where:** `retrieve/dense.py`, D-028, D-029.

bge-small is a **bi-encoder**: the query and each chunk are embedded
*independently* into 384-dimensional unit vectors, and relevance is their
cosine similarity. Independence is what makes it fast: chunks are embedded
once, at index time; a query is one forward pass plus a matrix product.

- **Exact search.** `DenseIndex.scores` computes `E @ q` over all 93k rows:
  milliseconds, no approximation. Approximate indexes (HNSW) only pay off at
  millions of vectors.
- **The query instruction.** bge was trained with "Represent this sentence
  for searching relevant passages: " in front of queries, never passages.
  sentence-transformers adds nothing on its own for this model (its config
  has no prompts), so `SentenceTransformerEncoder.encode_query` passes the
  prompt explicitly from config. A missing prompt is a silent quality loss,
  not an error.
- **The header.** Each chunk is embedded as `"Title > Section\n" + text`
  (D-029). That's why Phase 2 sized chunks so header + body + 2 ≤ 512: no
  chunk is truncated here.
- **CPU build time:** 45 min for 93,031 chunks (34 chunks/s), slowest at
  the start because sentence-transformers encodes the longest chunks first.
  The matrix is 143 MB.
- **The model loads from the local cache** (`local_files_only=True`, then a
  download only on a miss). Without that, every load asks the HF Hub for
  metadata: an external call each time you start a query or a chat.

## 3. Hybrid: Reciprocal Rank Fusion

**Where:** `retrieve/fusion.py`, D-030.

BM25 scores are unbounded (2.7 here, 10 there); cosines live in [-1, 1].
Adding them needs a normalisation that changes from query to query. RRF
avoids this by using **ranks only**:

```
rrf(d) = Σ over retrievers of 1 / (k + rank(d)),   k = 60
```

- With **k = 1**, a single first place (1/2) ties two third places
  (1/4 + 1/4).
- With **k = 60**, ranks 1 and 3 are worth almost the same (1/61 vs 1/63),
  so a chunk that **both** retrievers like (2/63) beats one that only one
  retriever loves (1/61). The probe's toy table shows both cases.

Each retriever contributes its top 100 (`retrieve.candidates`). A chunk
outside a list gets nothing from that retriever.

**Why you can't threshold RRF scores** (Phase 7's abstention): an RRF score
says where a chunk ranked, not how relevant it is. The best chunk for a
nonsense query still gets 2/61. The harness therefore records the top
**dense cosine** per question for the abstention gate.

## 4. The Retriever and stale indexes

**Where:** `retrieve/retriever.py`, `index/pipeline.py`.

- Every index folder has a `meta.json`: the settings it was built with and
  the **sha256 of chunks.jsonl**.
- `Retriever.load` refuses an index if the chunks changed or the settings
  differ ("run `rag index`"). Without this, re-chunking would leave an index
  whose row 4,512 is a different chunk than the one it embedded, and every
  result would be silently wrong.
- `retrieve(query, k, mode, allow)`: `allow` is a `Chunk → bool` filter. It
  is the hook for the year filter you add in Phase 7.
- Every hit carries **both** retrievers' rank and score, whatever the mode,
  so `rag inspect query` can show where they agree.

## 5. Metrics, and why each one is defined this way

**Where:** `eval/metrics.py`, `eval/harness.py`, D-031.

| Metric | Definition | Why |
|---|---|---|
| Recall@5 | fraction of gold spans with a containing chunk in the top 5 | spans survive re-chunking (D-022); overlap copies count once |
| Paper Recall@5 | fraction of gold papers with any chunk in the top 5 | "found the right paper" even if not the exact passage |
| MRR@10 | 1 / rank of the first chunk containing a gold span | rewards putting evidence first, which the chat needs |
| dup@5 | fraction of top-5 slots whose paper already appeared higher | measures wasted slots: Phase 5's target |

- **Identity is trustworthy only** (`eval/identity.py`). Records are the
  same paper when they share an arXiv id, DOI, S2 id or ACL id, or are
  linked by `derived_from`. This is the judge for duplicates and for
  counting a **linked copy's chunk as a hit**. Without that, returning the
  arXiv copy of an ACL gold paper would score as a miss before Phase 5 even
  starts. The fuzzy resolver Phase 5 builds is what's being tested, so it
  can't also be the judge.
- **Follow-ups are scored three ways:** raw (as asked), the oracle rewrite,
  and the system rewrite (Phase 7). The raw–oracle gap is what query
  rewriting can win at most.
- **Unanswerable questions** get no recall. Their top cosine is kept for
  the abstention gate.

## 6. Is a difference real? Paired bootstrap

**Where:** `metrics.bootstrap_ci`, `metrics.paired_bootstrap_diff`,
`report.paired_table`.

With about 30 answerable questions, one question flipping moves Recall@5 by
about 0.03. The bootstrap resamples the questions 1,000 times
(`eval.bootstrap`) to show how much a mean would move on a different draw
of questions.

- The **95% interval** on one mode's mean is wide: with 30 questions,
  expect ±0.1 or more.
- Comparing two modes, use the **paired** difference: per question,
  hybrid's score minus BM25's, then bootstrap those differences. Questions
  that are hard for both cancel out, so the interval is much narrower than
  comparing two separate intervals. `rag report` marks a difference with
  `*` when its interval excludes 0.

The test `test_paired_diff_cancels_shared_variation` shows the extreme
case: if B = A + 0.1 on every question, the paired interval is exactly
[0.1, 0.1], whatever the spread of A.

## 7. The results file

**Where:** `eval/results.py`, `eval/run.py`.

`results/<date>-<name>.json` records:
- the git commit (with `-dirty` if there were uncommitted changes);
- the config and its hash;
- the questions file's sha256;
- each index's meta;
- per mode, every question's metrics and top-10 chunk ids.

Any number in a report can be traced to the exact code, settings, questions
and index that produced it. `rag report` rebuilds every table from this
file alone.

## 8. The first build, on real queries

`docs/figures/phase4-index-stats.md` has the numbers:
- BM25 builds in 7 s (218k terms); dense in 45 min.
- Loading takes 2.7 s, then about 15 ms per query.
- On five demo queries (not gold), **BM25 and dense share only 2–3 of their
  top 10**. They find different chunks, the situation where fusion helps.
- "What is BPE-dropout?" shows both retrievers' typical failures:
  - BM25 puts the BPE-dropout paper's **figure caption** first, because it
    repeats "BPE-dropout".
  - Dense prefers **other papers' background sections** explaining BPE-dropout.
  - Hybrid mixes both.
- **Duplicates are everywhere.** BM25's top 5 for that query is the ACL and
  arXiv copies of one paper alternating (dup@5 = 0.8). Phase 5 exists for
  this.

Parsing problems from Phase 2 that show up here:
- **Captions repeat their label:** "Figure 4: Figure 4: …". Fixing it
  changes the chunks, which forces a 45-minute re-index, so it waits for
  the next re-chunk.
- **A formula taken as a section heading:** "ϕ(w) ∝ I(w)." has 3 letters, so
  it passes `MIN_HEADING_LETTERS`.

---

## What to look at

- `uv run rag inspect query "<question>"`: the `bm25 rk/score` and
  `dense rk/cos` columns. A "–" means that retriever didn't have the chunk
  in its top 100. Rows with one "–" are where hybrid earns its keep.
- `data/index/*/stats.json` and `meta.json`: chunk counts, vocabulary size,
  build time, the chunks sha256.
- `docs/figures/phase4-bm25-rrf.md`: BM25 by hand, the k1/b sweep, RRF's k.
- Once your questions exist: `uv run rag eval run --name baseline`, then
  `uv run rag report <date>-baseline`. Start with the paired-differences
  table.

## Common pitfalls (summary)

| Pitfall | Guard here |
|---|---|
| Query instruction silently missing | prompt passed explicitly from config; test checks it's on queries only |
| Index rows out of step with chunks | sha256 + settings check in `Retriever.load` |
| Nondeterministic ties | our stable `top_k`; RRF ties by best rank, then position |
| BM25 splitting "gpt-2", "byt5" | compound-aware `terms` |
| Thresholding RRF scores for abstention | top dense cosine recorded instead |
| Duplicates counted as misses or hits arbitrarily | trustworthy identity groups |
| Reading noise as a win | paired bootstrap intervals; `*` only when 0 is excluded |
| Gold going stale under re-chunking | spans mapped at scoring time (Phase 3) |
