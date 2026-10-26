# Phase 4: first full index build

Built 2026-10-26 from commit `0584d55` (clean worktree), config hash
`71c62f4e…`, chunks.jsonl sha256 `a1b6a1d1…`. Source: `data/index/*/stats.json`
and `meta.json`.

| Index | Chunks | Papers | Size | Build time (CPU) |
|---|---|---|---|---|
| BM25 (bm25s, Lucene, k1 1.5, b 0.75) | 93,031 | 1,781 | 218k terms | 7 s |
| Dense (bge-small-en-v1.5, header prepended) | 93,031 | 1,781 | 93,031 x 384 float32, 143 MB | 45 min (34 chunks/s) |

- Loading both indexes plus the model takes 2.7 s. A query takes about 15 ms
  in any mode, because every mode computes both retrievers' scores so
  `inspect query` can show them side by side.
- Trustworthy identity groups: 1,607 papers for 1,979 manifest records.

## Five demo queries (not gold: illustration only)

| Query | top-10 overlap BM25 / dense | dup@5 BM25 | dup@5 dense | dup@5 hybrid |
|---|---|---|---|---|
| What is BPE-dropout? | 2 | 0.8 | 0.2 | 0.4 |
| How does SentencePiece handle whitespace? | 3 | 0.0 | 0.4 | 0.4 |
| tokenizer fertility for low-resource languages | 2 | 0.2 | 0.4 | 0.2 |
| byte-level models sequence length cost | 2 | 0.4 | 0.4 | 0.4 |
| Unigram language model subword regularization | 3 | 0.4 | 0.0 | 0.2 |

- BM25 and dense agree on only 2–3 of their top 10. They find different
  chunks, which is the case where fusion can help.
- dup@5 counts a top-5 slot as redundant when its paper (identity group)
  already appeared higher. The BPE-dropout query's BM25 top 5 is the ACL and
  arXiv copies of one paper alternating. That's Phase 5's target.
