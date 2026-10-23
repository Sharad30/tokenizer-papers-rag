# Phase 1 teaching note: Corpus collection

The collect stage turns three external sources into one manifest. Every paper
it considered appears in the manifest with a decision and a reason code.
Each concept below is explained as it is implemented here, with the pitfall
it guards against and what to look at in the outputs.

```
rag collect discover -> resolve -> filter -> licence -> download -> manifest
  data/raw/discovered/  resolved/  filtered/  licensed/  downloaded/  data/manifest/
```

Each step reads the previous step's `records.jsonl` and writes its own, plus
a `stats.json` (`pipeline._stats`). Code: `src/rag/collect/pipeline.py`,
one section per step.

---

## 1. Being a polite, idempotent client

**Where:** `src/rag/collect/http.py` (`RateLimiter`, `ResponseCache`, `PoliteClient._send`).

- **Rate limiting per host.** `RateLimiter.wait(host)` sleeps until
  `min_interval_s[host]` has passed since the *last request to that host*.
  arXiv's terms ask for at most one request every 3 s, so `export.arxiv.org`
  is 3.0. The clock and sleep functions are injected, so
  `tests/collect/test_http.py` checks the spacing with a fake clock and no
  real waiting.
- **Response cache.** `ResponseCache.key` hashes method, URL, params and JSON body.
  Headers are deliberately left out, so an API key can't leak into the cache
  and doesn't change the key. A second identical request never reaches the
  network, which is what makes reruns free (`test_rerun_makes_no_network_calls`).
  Each cache file stores the request next to the body, so you can open one
  and see what it was.
- **Retries.** 429 and 5xx responses, timeouts and connection errors are
  retried with exponential backoff (`backoff_base_s * 2**(n-1)`, capped by
  `max_backoff_s`). A `Retry-After` header wins over our own backoff. We wrote
  this loop ourselves because httpx's own `retries=` only covers failed
  connections, not 429s.
- **`accept` hook.** Some failures arrive as HTTP 200, e.g. an arXiv page that
  comes back empty under load. The caller can reject the body
  (`arxiv._page_is_complete`); a rejected body is retried and never cached.
  **Pitfall:** without this, the empty page would be cached and every rerun
  would "succeed" with missing papers.
- **What happened in practice (no S2 API key):** at 1.1 s spacing, Semantic
  Scholar's shared pool returned 429 on about half of first attempts, and one
  batch call failed all five. The fix (commit `42c2332`) was 3 s spacing,
  8 attempts, and a 120 s cap on any single wait.

## 2. Failing loudly beats truncating quietly

**Where:** `s2.snowball`, the `except HttpError` branch.

In the first smoke run, one citations page ran out of retries. `snowball`
logged a warning and *carried on*, so those papers were silently missing and
`stats.json` showed no failure. Commit `44f6be9` changed this: only an HTTP 400
(the API's undocumented limit on how deep pagination can go) stops pagination
quietly, and anything else re-raises. **The general lesson for a pipeline
like this:** when a step is cached and resumable, failing costs one rerun.
Truncating silently costs you a corpus you can't trust.

## 3. Scholarly identifiers and manifestations

**Where:** `schemas.CandidateRecord`, `s2.lookup_id`, `pipeline._id_keys`, `pipeline._link`.

- A **manifestation** is one concrete version of a paper: an arXiv v1, the
  arXiv latest version, or the ACL venue copy. Each is its own
  `record_id` (`arxiv:1508.07909v5`, `acl:P16-1162`). They are *linked* by
  shared IDs and **never merged here**; merging is Phase 5's job, and its
  before/after numbers are only honest if the duplicates are still there.
- **arXiv IDs** have versions (`v1`…`vN`). `published` in the Atom feed is
  the v1 date, even when the entry is v5 (`arxiv._parse_entry`).
- **Semantic Scholar is a linking layer, not a manifestation.** S2 merges
  versions into one `paperId` whose `externalIds` carry `ArXiv`, `ACL`, `DOI`.
  We checked this live: BPE is one S2 paper with ArXiv `1508.07909` and ACL
  `P16-1162`. That's why `resolve` folds an S2 candidate into an arXiv/ACL
  record it matches, and why `filter` replaces an accepted S2-only record that
  turns out to be on arXiv or ACL with that record
  (`pipeline._materialise_s2`).
- **Linked copies:** an accepted arXiv paper gets its ACL copy and vice versa
  (`pipeline._linked_copies`). The copies *inherit* the decision
  (`DecidedBy.INHERITED`), so they're never judged separately. **Bug found on
  the real run (fixed in `54b4d34`):** "is the copy present?" originally
  looked at every ID of every record. An ACL record that knows its own arXiv
  ID made the arXiv copy look present, so it was never fetched. The check now
  counts only records of that source.

## 4. Snowball sampling and its bias

**Where:** `pipeline._snowball_records`, `s2.snowball`, config `collect.s2`.

One hop out from seven anchor papers: what each cites (`references`) and
what cites it (`citations`). This finds papers that keyword search misses,
because their titles don't use the expected words.

- **Bias 1, towards well-cited work and the anchors' sub-topics.** Areas
  without an anchor (tokenization for code, arithmetic) are reached only by
  keyword queries.
- **Bias 2, recency.** S2 returns citations newest first, and we cap at
  `max_items_per_anchor: 10000`. For BPE and SentencePiece, which have tens of
  thousands of citing papers, the cap keeps the most recent ones. That partly
  explains the 2025–26 spike in the year figure.
- **Most snowballed papers only *use* a tokenizer:** about 11k of the 13.5k
  S2 candidates had no topic term at all (`no_topic_term`).
- `s2.within_cutoff` drops snowballed papers newer than `corpus.max_date`.
  The citations endpoint happily returns papers from after the cutoff.

## 5. A two-stage relevance filter with reason codes

**Where:** `relevance.KeywordRules.decide`, `relevance.EmbeddingJudge`, config `collect.relevance`, D-005, D-008.

The order of the rules matters (`decide`):

1. No topic term in the title or abstract: reject `no_topic_term`.
2. A title term for another modality or domain (image, speech units,
   protein, graph, recommendation, …): reject `off_topic_term`. This is the
   **text-only boundary** you chose; "tokenization" has become a generic ML
   word.
3. A strong topic term in the title: accept `title_term`.
4. Abstract only, with topic terms in fewer than 2 sentences: reject
   `passing_mention`, i.e. "we train with a BPE tokenizer". The rule counts
   *sentences*, not terms, because one phrase like "BPE tokenizer" already
   contains two terms.
5. Otherwise the paper is ambiguous and goes to the embedding judge: the
   cosine similarity of title + abstract to the D-005 boundary text, against
   `embedding_threshold`.

**The embedding judge:**
- bge-small-en-v1.5, **without** the query instruction. The model card
  reserves that for short retrieval queries, and this is passage-to-passage
  similarity.
- `normalize_embeddings=True`, so a dot product equals the cosine
  (`EmbeddingJudge.scores`).
- We confirmed sentence-transformers applies no default prompt for this model.
  In Phase 6, remember that newer versions *can* apply prompts silently.

**Pitfalls seen:**
- **Whole-word matching** means `tokenizers` doesn't match `tokenizer`, so
  every variant has to be listed in config (`test_matching_is_whole_word...`).
  A compound like `BrahmicTokenizer` doesn't match at all.
- **Embedding scores are compressed.** Title-accepted papers had a median of
  0.79, and the ambiguous ones spanned 0.65–0.83 with no clean gap. One
  threshold is a blunt instrument; the audit measures how blunt.
- **The first rule set was too loose.** On the full discovery it would have
  accepted ~2,200 papers on title terms (about 70% on-topic in a 25-title
  sample) and sent ~4,000 to the judge. Tightening (commit `ae1b8a2`) brought
  this down to 1,391 title accepts and 908 for the judge.

## 6. Versions: what a merge would hide

**Where:** `versions.kept_v1`, `pipeline._revised_v1s`, config `collect.versions`, D-009.

For each accepted arXiv paper with more than one version, fetch v1. We
checked that `id_list=…v1` returns v1's *own* abstract. Keep v1 as its own
record when `abstract_similarity` (difflib ratio with `autojunk=False`) is
below 0.95. Measured: the BPE paper's v1 vs v5 abstract scores 0.907, because
its reported gains changed from +0.8/+1.5 to +1.1/+1.3 BLEU. A one-word typo
fix scores ~1.0. **Pitfall:** difflib's default `autojunk` heuristic ignores
frequent characters in long strings and distorts the ratio for abstracts.

## 7. Licences

**Where:** `arxiv.fetch_oai_record`, `acl.licence_for_year`, `s2.to_candidate`.

- **arXiv:** the query API has no licence field. We use OAI-PMH `arXivRaw`
  (`oaipmh.arxiv.org/oai`), one call per accepted paper. It is the slowest
  step after downloads.
- **ACL:** the library has no per-paper field. The Anthology's published
  policy is CC BY 4.0 from 2016 and CC BY-NC-SA 3.0 before that, so
  `licence_source=acl_year_rule`.
- **S2:** `openAccessPdf.license` (codes like `CCBY`).
- PDFs are never committed or redistributed. The download code and the
  `pdf_sha256` in the manifest are what make the corpus reproducible.

## 8. Downloads that resume

**Where:** `download.download`.

- A valid existing file (one starting with `%PDF`) is never fetched again
  (`ALREADY_PRESENT`).
- A response that isn't a PDF, often a publisher landing page behind an
  open-access link, is recorded as `NOT_PDF` and not saved.
- Writes are atomic (`common.io.write_bytes_atomic`).
- Failures are recorded in the record and in `stats.json`; nothing raises.

## 9. The manifest, and what it costs to commit it

**Where:** `manifest.write_manifest`, D-010.

The first full run wrote 19,044 rows: 30 MB of CSV and 42 MB of JSON,
almost all of it reject abstracts. You chose to split it. Accepted records
go to `manifest.json` (exact types, which `read_manifest` loads) and
`manifest.csv`. Rejects go to `rejected.csv.gz` without abstracts
(`read_rejected`), 2.5 MB. The gzip header's mtime is fixed at 0, so
rewriting unchanged data gives identical bytes and no spurious git diff.
`audit-sample` reads the full local records in `data/raw/downloaded/`,
because labelling a reject needs its abstract.

**Pitfall:** the stock Python `.gitignore` has a PyInstaller `MANIFEST`
line. On macOS's case-insensitive filesystem it matched `data/manifest/`,
and `git add` refused the files. `git check-ignore -v <path>` names the
rule responsible.

## 10. The first run in numbers

| | |
|---|---|
| Candidate records | 19,044 (arXiv 4,267 · ACL 2,532 · S2-only 12,245) |
| Accepted manifestations | 1,979, of which 1,761 decided directly |
| Accepted by | title_term 1,391 · embedding 370 (of 908 judged) · linked copy 77 · kept v1 141 |
| Rejected by | no_topic_term 13,630 · passing_mention 2,234 · off_topic_term 663 · embedding 538 |
| PDFs | 1,782 downloaded · 139 no URL · 37 failed · 21 not a PDF |
| Licences (accepted) | arxiv_oai 1,044 · acl_year_rule 678 · s2_open_access 102 · unknown 155 |
| Run time (no S2 key) | discover 9 min · resolve 14 min · licence 45 min · download 79 min |

All seven anchors are accepted and downloaded, and the five with venue
copies have their ACL record. The S2-only download failures are mostly
`doi.org` and publisher links that answer with a landing page or 403. The
three arXiv failures are v2s that return 404.

---

## What to look at

- `data/raw/*/stats.json`: one per step, with counts, `network_calls`,
  `seconds`, `git_commit` and `config_hash`. If you rerun a step, its
  `network_calls` should be 0.
- `data/manifest/manifest.csv` (accepted) and `rejected.csv.gz` (open it
  with `gunzip -c` or straight in a spreadsheet): sort by `reason` to see
  each rule's work. Filter `decided_by = embedding` in both and sort by
  `embedding_score` to see the judge's boundary from each side.
- `docs/figures/phase1-*.png` and `phase1-coverage.md`: what each route
  contributes. A paper found *only* by snowball is one keyword search would
  have missed.
- One cache file in `data/raw/http_cache/export.arxiv.org/`: the exact
  request and response.

## Common pitfalls (summary)

| Pitfall | Guard here |
|---|---|
| Hammering a source / getting blocked | per-host `RateLimiter`, conservative intervals |
| Empty or error page cached as success | `accept` hook, `ArxivError` on error feeds |
| Retries exhausted, papers silently dropped | snowball re-raises; rerun resumes from cache |
| Versions merged before they can be measured | one record per manifestation, linked by IDs |
| Copies of one paper decided differently | linked copies and v1s inherit the decision |
| "tokenization" in non-text papers | text-only title exclusions (`off_topic_term`) |
| Papers that only *use* BPE | `passing_mention` rule |
| Silent cutoff violations | `within_cutoff` on snowball, date window on arXiv |
| Unreproducible corpus | `pdf_sha256`, cached responses, pinned `max_date` |
| Manifest too big for git | accepted in full, rejects gzipped without abstracts |
| A stock `.gitignore` rule hiding committed output | `!data/manifest/`, `git check-ignore -v` |
