# Ablation `embeddings`

Commit 0e564fa880df, 2026-11-02. Question sets: gold (28 questions), silver (66 questions). dup@5 on 40 probe queries (`phase-5-queries.txt`, not gold). Judge: identifiers + 0 labelled pairs. Means with 95% bootstrap intervals.

| Condition | recall@5 (gold) | paper_recall@5 (gold) | mrr@10 (gold) | recall@5 (silver) | paper_recall@5 (silver) | mrr@10 (silver) | dup@5 | copy_dup@5 | latency p50 / p95 | index build | dims | size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 only | 0.087 [0.000, 0.217] | 0.370 [0.196, 0.587] | 0.087 [0.000, 0.217] | 0.742 [0.636, 0.848] | 0.924 [0.864, 0.985] | 0.729 [0.617, 0.829] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 15 / 17 ms | bm25 0 min | – | 73 MB |
| bge-small dense | 0.022 [0.000, 0.065] | 0.576 [0.402, 0.754] | 0.014 [0.000, 0.043] | 0.470 [0.348, 0.591] | 0.788 [0.682, 0.879] | 0.399 [0.285, 0.511] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 14 / 16 ms | bge_small 58 min | 384 | 145 MB |
| bge-small hybrid (deployed) | 0.196 [0.043, 0.370] | 0.522 [0.326, 0.717] | 0.156 [0.029, 0.304] | 0.636 [0.515, 0.758] | 0.909 [0.833, 0.970] | 0.596 [0.482, 0.708] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 14 / 16 ms | bm25 0 min + bge_small 58 min | 384 | 218 MB |
| bge-base dense | 0.109 [0.000, 0.218] | 0.496 [0.319, 0.688] | 0.084 [0.004, 0.178] | 0.455 [0.318, 0.576] | 0.788 [0.697, 0.879] | 0.425 [0.305, 0.539] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 27 / 32 ms | bge_base 129 min | 768 | 288 MB |
| bge-base hybrid | 0.239 [0.087, 0.413] | 0.504 [0.333, 0.703] | 0.210 [0.065, 0.370] | 0.667 [0.545, 0.773] | 0.939 [0.879, 0.985] | 0.605 [0.492, 0.717] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 26 / 30 ms | bm25 0 min + bge_base 129 min | 768 | 361 MB |
| SPECTER2 dense | 0.043 [0.000, 0.130] | 0.449 [0.261, 0.652] | 0.048 [0.000, 0.139] | 0.136 [0.061, 0.227] | 0.439 [0.333, 0.561] | 0.099 [0.041, 0.176] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 28 / 34 ms | specter2 76 min | 768 | 288 MB |
| SPECTER2 hybrid | 0.109 [0.000, 0.239] | 0.514 [0.326, 0.717] | 0.101 [0.000, 0.232] | 0.667 [0.561, 0.773] | 0.909 [0.833, 0.970] | 0.445 [0.356, 0.540] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 28 / 32 ms | bm25 0 min + specter2 76 min | 768 | 361 MB |
| bge-small hybrid + reranker | 0.217 [0.043, 0.391] | 0.489 [0.301, 0.696] | 0.186 [0.052, 0.332] | 0.818 [0.727, 0.909] | 0.955 [0.894, 1.000] | 0.795 [0.697, 0.886] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 3519 / 4704 ms | bm25 0 min + bge_small 58 min | 384 | 218 MB |
| bge-base hybrid + reranker | 0.232 [0.087, 0.406] | 0.475 [0.290, 0.678] | 0.201 [0.068, 0.344] | 0.818 [0.727, 0.909] | 0.955 [0.894, 1.000] | 0.795 [0.697, 0.886] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 3848 / 4665 ms | bm25 0 min + bge_base 129 min | 768 | 361 MB |
| bge-small dense without query prompt | 0.152 [0.022, 0.304] | 0.609 [0.435, 0.775] | 0.124 [0.015, 0.247] | 0.394 [0.273, 0.500] | 0.727 [0.621, 0.833] | 0.335 [0.231, 0.433] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 14 / 17 ms | bge_small 58 min | 384 | 145 MB |

**Top two by recall@5 (gold):** bge-base hybrid vs bge-base hybrid + reranker. Paired difference recall@5 0.007 [-0.116, 0.138], mrr@10 0.009 [-0.127, 0.146]; recall@5 higher on 2 questions, lower on 2, equal on 19. An interval that includes 0 means the data can't tell them apart.

**Paired comparisons (gold, a minus b):**

| a | b | recall@5 | paper_recall@5 | mrr@10 | a higher / lower / equal |
|---|---|---|---|---|---|
| bge-small hybrid + reranker | bge-small hybrid (deployed) | 0.022 [-0.130, 0.174] | -0.033 [-0.170, 0.069] | 0.030 [-0.121, 0.182] | 2 / 2 / 19 |
| bge-base hybrid | bge-small hybrid (deployed) | 0.043 [0.000, 0.130] | -0.018 [-0.130, 0.054] | 0.054 [0.000, 0.141] | 1 / 0 / 22 |
| SPECTER2 hybrid | bge-small hybrid (deployed) | -0.087 [-0.217, 0.000] | -0.007 [-0.065, 0.036] | -0.054 [-0.185, 0.065] | 0 / 2 / 21 |
| bge-small hybrid (deployed) | BM25 only | 0.109 [0.000, 0.239] | 0.152 [0.022, 0.304] | 0.069 [0.000, 0.170] | 3 / 0 / 20 |
| bge-small hybrid (deployed) | bge-small dense | 0.174 [0.043, 0.348] | -0.054 [-0.101, -0.011] | 0.141 [0.022, 0.301] | 4 / 0 / 19 |
| bge-small dense | bge-small dense without query prompt | -0.130 [-0.261, 0.000] | -0.033 [-0.130, 0.022] | -0.109 [-0.240, -0.016] | 0 / 3 / 20 |

- recall@5: share of a question's gold passages found in the top 5 chunks;
  paper_recall@5: share of its gold papers (any copy) with any chunk there. A gap between
  them means the right paper came back with a different passage of it (the
  per-work cap shows one chunk per paper).
- dup@5 (D-031): share of the top 5 slots whose paper already appeared higher;
  copy_dup@5 counts only another record of the paper (a version or venue copy).
  Judged by identifiers + labelled pairs (D-042): lower bounds until pairs are
  labelled. A per-work cap of 1 makes dup@k 0 by construction.
- Latency: wall time per query on this machine's CPU (encode + search + rerank +
  duplicate handling), after one warm-up query; SPECTER2's query encoding runs in
  its own process and is timed there. Index build: CPU minutes, from stats.json.
- Silver (D-044): questions an LLM wrote from one chunk each. One gold chunk per
  question, so other passages that answer it count as misses; single-paper
  lookups only; wording drawn from the passage favours BM25. Provisional until
  the gold set exists; never pooled with it.

Generated by `rag eval ablate`. Per-query results in the JSON.
