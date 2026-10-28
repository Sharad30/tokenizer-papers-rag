# Phase 5 teaching note: versions and near-duplicates

Phase 5 decides which records are the same paper, and stops one paper
filling the top results, without merging papers that are only about the
same thing.

```
manifest + chunks ─▶ rag dedup run ─▶ data/works/
                      1 ids      shared arXiv id / DOI / S2 id / ACL id / derived_from
                      2 title    near-exact title + shared authors + close years
                      3 minhash  shared full text (MinHash) + shared authors
                      4 guard    refuse merges that only chain through a third record
                      5 canonical ACL copy > latest arXiv version > most chunks
                      + surveys and the works their chunks restate

query ─▶ Retriever.search ─▶ rank (BM25 + dense, RRF)
                           ─▶ collapse (D-041): hide aliases → cap per work → [MMR] → demote surveys
                           ─▶ top k, each hit with work_id and the chunks it hides
```

Code: `src/rag/dedup/` (`resolve.py`, `minhash.py`, `surveys.py`,
`pairs.py`, `pipeline.py`, `show.py`), `src/rag/retrieve/collapse.py`,
`src/rag/eval/ablation.py`, `src/rag/common/identity.py`. Decisions:
D-040, D-041, D-042.

Numbers: 1,781 records → 1,414 groups by identifiers → 1,298 works
(125 title merges, 12 text merges, 0 refused by the guard). 4 surveys,
36 restating chunks.

---

## 1. Identity is not similarity

**Where:** `dedup/resolve.py` docstring; D-040 option 3.

Two papers that both measure tokenizer fertility on 50 languages have
abstracts that embed almost identically, but they are different work with
different numbers. A same-author follow-up is closer still: "Hard
Non-Monotonic Attention for Character-Level Transduction" (EMNLP 2018) and
"Exact Hard Monotonic Attention for Character-Level Transduction" (ACL
2019) have the same authors and a title ratio of 92.

What makes two records one paper is shared **text**, not shared
**meaning**: the same sentences, tables and numbers. Those two papers share
2% of their 5-word sequences; the arXiv and ACL copies of one paper
usually share more than 90%.

So the resolver never uses embeddings to merge. It uses identifiers, then
titles, then lexical overlap of the full text, and every fuzzy merge also
needs shared authors.

## 2. Shingles and Jaccard

**Where:** `dedup/minhash.py` (`shingles`, `exact_jaccard`).

A paper's text becomes a **set** of word 5-grams ("byte pair encoding
merges the", "pair encoding merges the most", …). Two sets compare by
Jaccard similarity:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

- **Why 5 words:** short shingles are shared by any two papers on the same
  topic ("the", "byte pair encoding"). Long ones only match copied text.
- **Why a set:** repeats don't count, so a chunk overlap of 40 tokens
  doesn't inflate anything.
- **Edge case:** a heavily revised v1 vs its final version can score 0.02.
  The text is mostly rewritten. That is why identifiers are trusted
  outright (step 1): they join the versions that text never could.

## 3. MinHash: estimating Jaccard from short signatures

**Where:** `dedup/minhash.py` (`MinHasher`, `estimate_jaccard`);
`probes/phase-5-minhash.py`.

Exact Jaccard over all 1.6 million pairs of papers is slow, and each set
has thousands of shingles. MinHash compresses each set to `num_perm`
numbers:

1. Hash every shingle to a number (`shingle_hashes`, blake2b, so it's
   stable between runs; Python's `hash()` isn't).
2. Pick a random hash function h(x) = (a·x + b) mod P. Apply it to every
   shingle and keep the **minimum**.
3. Repeat with `num_perm` different (a, b) pairs. That list of minima is
   the signature.

Why it works: under a random h, the minimum of A ∪ B is equally likely to
be any of its elements. The two sets have the same minimum exactly when
that element is in A ∩ B. So

```
P(minA = minB) = |A ∩ B| / |A ∪ B| = J
```

and the share of positions where two signatures agree estimates J, with a
standard error of sqrt(J(1 − J) / num_perm).

**What the probe shows** (`docs/figures/phase5-minhash.md`): mean absolute
error on 200 real pairs is 0.026 at 16 hash functions and 0.008 at 256,
following the theory line. Quadrupling `num_perm` halves the error. P is
2³¹ − 1, so a·x + b stays below 2⁶³ and uint64 arithmetic never wraps
(`PRIME` in `minhash.py`).

## 4. LSH banding: finding candidates without comparing every pair

**Where:** `minhash.lsh_candidates`, `candidate_probability`; the S-curve
figure.

Cut each signature into b bands of r rows. Two papers become a
**candidate pair** if any whole band is identical. For a pair with
Jaccard s:

```
P(one band identical)  = s^r
P(no band identical)   = (1 − s^r)^b
P(candidate)           = 1 − (1 − s^r)^b      an S-curve, steepest near (1/b)^(1/r)
```

Here b = 64, r = 4 (256 hash functions):
- P(candidate) is 0.98 at Jaccard 0.5, the merge threshold. The curve's
  middle is at (1/64)^(1/4) ≈ 0.35, below the threshold on purpose, so
  real copies are rarely missed.
- It is 0.10 at 0.2, so only a few unrelated pairs are compared.

Candidates are then checked with the full signature (`estimate_jaccard`).
Banding only decides which pairs get looked at.

**Hand check** (`tests/dedup/test_minhash.py`): at s = 0.5, b = 32,
r = 4, 1 − (15/16)³² ≈ 0.8732.

## 5. The resolver's steps and thresholds

**Where:** `resolve.resolve`, `_title_edges`, `_text_edges`; `config.yaml`
`dedup:`.

| Step | Merges when | Why this threshold |
|---|---|---|
| ids | shared arXiv id, DOI, S2 id, ACL id, or `derived_from` | the dup@5 judge's own rules; reproduces its 1,414 groups exactly |
| title | `token_sort_ratio` ≥ 97 on normalised titles, author overlap ≥ 0.5, years ≤ 2 apart | at 92 a same-author follow-up appears |
| minhash | estimated Jaccard ≥ 0.5, author overlap ≥ 0.5 | among same-author pairs with no shared ID, none above 0.5 was a different paper (118 of them, all missed copies) |
| close title | ratio 85–97 and Jaccard ≥ 0.2 | "Towards Character-Level Transformer NMT…" → "Towards Reasonably-Sized…": ratio 89, Jaccard 0.22, one paper |

- **Author overlap:** the share of the *shorter* author list found in the
  other (`author_overlap`), so a journal version that adds authors still
  overlaps fully.
- **Normalised titles:** accents and punctuation removed (`normalise`).
  "BabyLM’s" with a curly apostrophe must equal "BabyLM's".

**Edge cases from the data:**
- `acl:2026.abjadnlp-1.2` and `acl:2026.loreslm-1.4` are one paper at two
  workshops (Jaccard 0.86). They merged, which is right for a reader.
- Two near misses are probably one paper: "Improving Consistency in LLM
  Inference…" vs "Improving Self Consistency…" (ratio 79, Jaccard 0.48).
  They are in your labelling candidates rather than solved by loosening a
  threshold.

## 6. Union-find and the chain problem

**Where:** `common/identity.UnionFind`, `resolve._guarded_merge`;
`test_chain_guard_refuses_a_link_between_unrelated_ends`.

Union-find keeps a parent pointer per record. `find` follows the pointers
to the root, and `union` points one root at the other. Groups are
**transitive** by construction: union(A, B) then union(B, C) puts A and C
together, even if A and C share nothing. With fuzzy evidence that is
dangerous: B could be a paper that copies half of A and half of C.

The guard:
1. Records joined by identifiers form **units**, which are trusted.
2. Fuzzy edges are applied strongest first (highest Jaccard, then title).
3. An edge joining two clusters is refused unless **every pair of units**
   across them has direct evidence: a merge edge, or Jaccard ≥ 0.2.

That is complete linkage at the unit level. A revised v1 that shares
little text with the ACL copy doesn't block the merge, because the v1 is
in the same unit as its final version. On this corpus the guard refused
nothing. The test builds the chain on purpose to show it working.

## 7. The canonical record, and why aliases stay in the index

**Where:** `resolve.canonical_member`; `collapse.WorkMap.canonical_mask`.

- **Rule:** the ACL copy (the venue version, the one to cite), else the
  latest arXiv version, else the record with the most chunks. The rule
  that picked it is stored on each `Work`.
- **Aliases** are hidden at query time with a boolean mask, applied before
  ranking (the same mechanism as `allow`), so they never take candidate
  depth.
- **Why not remove them from the index:** a rebuild takes 45 minutes per
  variant. The kept v1s (D-009) are there to answer "what changed between
  versions?" later.
- **Dense results are identical** to a canonical-only index.
- **BM25 differs slightly:** its IDF still counts alias chunks, which makes
  terms in duplicated text look a little more common (D-040).

## 8. Query time: cap, MMR, survey demotion

**Where:** `retrieve/collapse.collapse` and its helpers;
`tests/retrieve/test_collapse.py`.

The order matters, and it is cap → MMR → cut to k → demote:

- **Per-work cap** (`cap_per_work`): at most N chunks per work, by rank.
  Hidden chunks are listed on the hit that kept the work's place, so
  `rag inspect query` shows "+21 more from work:acl:D18-2012". That
  SentencePiece example is the cost of N = 1: the evidence is there but
  only one chunk of it reaches the LLM. Your "You do" task tries N = 2.
- **MMR** (`mmr`, off by default): pick the next chunk by
  λ·relevance − (1 − λ)·(max cosine to chunks already picked). Relevance
  is the fused score scaled to [0, 1]. It reduces near-identical text
  generally, but by meaning, not identity, so it can't replace the cap.
- **Survey demotion** (`demote_restatements`): a survey chunk that restates
  work W moves just below W's chunk when W is also in the top k. The
  first version ran on the whole candidate list. It moved a survey below a
  source ranked about 20th, so the survey left the results and the source
  never came in. Hence "within the top k, last".

A **restatement** (`dedup/surveys.py`) needs both:
- **A citation:** the survey chunk's in-text citation resolves to W
  (bibliography entry → arXiv id, DOI, or title ratio ≥ 95 → record →
  work).
- **Closeness:** cosine ≥ 0.87 to W's closest chunk. That is the 95th
  percentile of a random work's closest chunk, because bge-small cosines
  sit high (median 0.82) even for unrelated work.

Surveys are flagged by **title words only**. S2's "Review" type tags
ordinary research papers too ("The Korean Morphologically Tight-Fitting
Tokenizer…").

## 9. Measuring it: the judge must not be the resolver

**Where:** `eval/identity.py`, `eval/metrics.py`
(`duplicate_rate_at_k`, `copy_duplicate_rate_at_k`), `eval/ablation.py`,
`results/2026-10-28-dedup-ablation.md`.

If the resolver's own works decided what counts as a duplicate, every
false merge would score as a win. The judge is identifiers plus the pairs
you label `same` (D-042). Until you label, copies found only by title or
text count as different papers.

| Condition | dup@5 | copy_dup@5 | copy_dup@5, resolver's view |
|---|---|---|---|
| No handling | 0.470 | 0.255 | 0.420 |
| Exact-ID merge | 0.410 | 0.000 | 0.230 |
| + fuzzy + MinHash | 0.455 | 0.000 | 0.000 |
| + cap 1 | 0.000 | 0.000 | 0.000 |

How to read it:
- **dup@5** (D-031) counts a second chunk of the same paper too, so it is
  mostly "one paper, many chunks", not copies.
- **copy_dup@5** counts only another record of the paper: what merging
  removes.
- **The fuzzy row's dup@5 goes up** (0.410 → 0.455). Hiding an unlabelled
  copy (which the judge thought was a different paper) frees a slot, often
  for another chunk of the same paper. The resolver's view shows the merge
  working (0.230 → 0). Labelled pairs will move the judge toward it.
- **The cap row is 0 by construction.** Its cost can only show in
  Recall@5 and MRR, which need your questions. `rag eval ablate` adds them
  automatically once `eval/questions.jsonl` exists.

Probe queries (`probes/phase-5-queries.txt`) are not gold. They were
written to touch topics with copies and surveys, and nothing was tuned on
them.

## 10. Labelled pairs

**Where:** `dedup/pairs.py`, `eval/README.md` "Labelled pairs".

`rag dedup pairs` sampled 55 pairs from four bands:

| Band | Pairs | What it tests |
|---|---|---|
| ID-linked | 5 | the identifier rules |
| Fuzzy-merged | 15 | merges most likely to be wrong |
| Near misses | 15 | the whole pool of same-author, close-title or some-shared-text pairs left apart |
| Similar abstracts, different authors | 20 | semantic similarity is not identity |

The candidates show the evidence but not the verdict. If you could see
"merged", your label would lean toward agreeing. `rag dedup score` then
gives precision (of the pairs joined, how many are one paper) and recall
(of the real pairs, how many were joined), per band.

---

## What to look at

- **Merges:** `uv run rag inspect work arxiv:2012.15524v1` (Fast WordPiece:
  title changed between versions) and `acl:2020.emnlp-main.203` (joined by
  title + text). Every merge edge shows its evidence.
- **Fuzzy merges:** `data/works/review.md` lists every work built by title
  or text evidence; read these first.
- **Hidden chunks:** `uv run rag inspect query "how does SentencePiece handle
  whitespace"`, then read the "+N more from" lines.
- **Results:** `data/works/stats.json` (edges per step, sizes, refusals),
  `docs/figures/phase5-minhash.md` (the probe),
  `results/2026-10-28-dedup-ablation.md`.
- **UI:** the "3 · Versions and duplicates" page (`make ui`).

## Common pitfalls

| Pitfall | Guard here |
|---|---|
| Merging by embedding similarity | lexical evidence only, always with shared authors |
| A title match alone merging a follow-up | ratio ≥ 97 alone; 85–97 needs shared text |
| Transitive chains through union-find | unit-level complete linkage, strongest edges first |
| Python's `hash()` changing between runs | blake2b shingle hashes |
| Grading the resolver with its own clusters | judge = identifiers + Sharad's labels; resolver view labelled "not ground truth" |
| dup@5 hiding what merging did | copy_dup@5 beside it |
| A cap that looks free because dup@5 hits 0 | its cost shows only in Recall/MRR; "+N more" shows it per hit |
| Demoting a survey below a source that never shows | demotion last, within the top k |
| Stale works after a re-chunk | `WorkMap.load` checks the chunks sha256 |
| Losing the kept v1s | aliases are masked, never deleted |
