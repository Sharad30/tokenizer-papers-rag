# Phase 2 teaching note: Parse and chunk

The parse stage turns each accepted PDF into chunks that follow section
boundaries. A chunk is both the unit retrieval returns and the thing an
answer cites. Each concept below is explained as it is implemented here,
with the pitfall it guards against and what to look at in the outputs.

```
rag parse grobid -> structure -> chunk -> mark
  data/parsed/tei/   papers/   chunks/   (manifest: included_in_index)
```

Each step reads the previous step's folder and writes its own, plus a
`stats.json`. Code: `src/rag/parse/pipeline.py`, one section per step.

---

## 1. Why PDFs lose structure, and what GROBID gives back

**Where:** `grobid.GrobidClient`, `tei.parse_tei`, D-013, D-014.

A PDF is a list of drawing instructions: put these glyphs at these
coordinates. It has no "paragraph", "heading" or "table". Columns, running
headers, footnotes and floating figures are all just positioned text.
Anything structural has to be *inferred* again from layout: font sizes,
positions, line spacing.

GROBID does that inference with a cascade of sequence-labelling models (CRF
in the image we use). Each model labels layout tokens with zones: header vs
body vs references, then heading vs paragraph vs figure inside the body. The
result is TEI XML:

- `<div><head n="3.1">…</head><p><s>…</s></p></div>`: one div per heading. The
  divs are **flat**, and nesting is implied only by the number `3.1`.
- `<s>`: sentences (we ask for `segmentSentences=1`).
- `<ref type="bibr" target="#b12">`: an in-text citation of bibliography entry `b12`.
- `<figure type="table">`: tables, with `<figDesc>` captions and `<row>/<cell>`.
- `coords="page,x,y,w,h"`: where an element sits in the PDF (we ask for it
  on paragraphs and sentences, which gives each chunk its page).

**Choices:**
- The **CRF image** (D-013) runs natively on this arm64 Mac in ~1.5 s per
  PDF. The deep-learning image is amd64-only and slow on CPU.
- **Consolidation off** (D-014). Otherwise GROBID quietly calls CrossRef for
  every paper, an external call we neither throttle nor cache.
- The **title comes from the manifest**, not GROBID's header model, because
  the manifest is authoritative.

**Pitfalls seen on real papers:**
- **Bogus headings.** In the BPE paper, CRF GROBID labelled the formula
  fragment `←h m ).` as a heading, which opened a fake top-level section.
  `tei._sections` now treats a head with fewer than 3 letters as a
  continuation (`MIN_HEADING_LETTERS`).
- **Missed headings.** A heading that GROBID reads as body text ends up
  inside a paragraph chunk ("… disregarded. 2 The notion of Lexico-Semantic
  Unit …").
- **Tables.** CRF table extraction is weak. The BPE paper's Table 1 is
  missing, and Table 2's "rows" are lines of its caption notes.
- **Hyphenation** across line breaks survives in places ("prob-lem").

## 2. Rebuilding the section tree

**Where:** `tei._sections`, `tei._number`.

GROBID's divs are flat, so the tree comes from heading numbers. `"3.1."`
becomes number `3.1` and level 2, and its path is the open headings above
it: `["Subword Translation", "Byte Pair Encoding (BPE)"]`.
- An unnumbered heading ("Limitations", "Conclusion" in some papers) goes to
  level 1.
- A div without a heading continues the previous section. GROBID starts a
  new div after a figure or page break.
- Appendices in `<back><div type="annex">` are kept as sections.
  Acknowledgements and the reference list are not.

## 3. The fallback, and why it is flagged

**Where:** `fallback.parse_pdf`, `pipeline._structure_one`, D-015.

When GROBID returns 204 (no content), an error, or a TEI with no body text,
PyMuPDF reads the PDF's text blocks instead. There is no zone model, so:
- one section per page;
- no tables, figures or references;
- the manifest's abstract stands in for one;
- text after a "References" heading is dropped, so the bibliography never
  reaches the index.

Chunks from these papers carry `parser=pymupdf`. Treat them as lower
quality: their headers say "Page 3", not "Method".

PyMuPDF is pinned below 2.0, because 2.0 makes an ONNX layout stack a
required dependency. Its licence is AGPL.

## 4. Chunking: precision against context

**Where:** `chunker.Chunker._pack`, D-017.

- A chunk that is too small ("We use 32k merges.") lacks the context to be
  found or understood.
- A chunk that is too large covers several topics, and its single embedding
  becomes an average of all of them. That is **dilution**: the part that
  answers your question is a smaller share of the vector, so similarity drops.
  The probe (§6) shows this.

The rules:
1. **Never cross a section.** A section is the author's own topic boundary,
   and a chunk spanning two sections mixes two topics *and* gets a header
   that's wrong for half of it. `_pack` runs per section, so the guarantee
   comes from the structure of the code, not from a check.
2. **Pack whole paragraphs greedily up to `target_tokens` (300).** A
   paragraph that doesn't fit the hard `max_tokens` (448) is split at
   sentence boundaries. A single sentence over the limit (rare: a mangled
   table or a list without punctuation) is cut on tokens and counted in
   `split_sentences`.
3. **Overlap:** the next chunk starts with the previous chunk's last whole
   sentences, up to `overlap_tokens` (40). This is only within a section,
   and never a broken sentence, so citation markers stay intact.
4. **Tables** become a caption plus markdown rows. Large tables are split by
   rows with the caption and header repeated, so each part reads on its own
   (D-020).
5. **Figure captions** are their own chunks (D-018).
6. **Long display formulas** become `[equation]` (D-019). CRF GROBID often
   puts figure text into "formulas", which is noise.
7. **References are never chunked.** They're in `papers/references.jsonl` for
   Phase 5 (shared references are a duplicate signal) and for citation work.
   In the index they would match every query that mentions a famous title.

## 5. Silent truncation, and the size invariant

**Where:** `tokens.HFTokenCounter`, `Chunker.header`, `pipeline._model_tokens`, D-016.

bge-small reads at most 512 tokens and drops everything after that, **with
no error**. So:
- **Count with the model's own tokenizer.** `HFTokenCounter` loads only
  bge-small's WordPiece tokenizer. A word count, or tiktoken, would
  disagree with what the model sees.
- **The header counts.** The "Title > Section" header is stored separately
  and prepended at embedding time (so Phase 4 can switch it off), but it
  still uses the model's budget. The header is capped at
  `512 - 2 - max_tokens` tokens, so `header + body + [CLS] + [SEP] ≤ 512`
  always holds. `chunk` re-checks every chunk and lists violations in its
  stats `failures`. There should be none.
- **Split by offsets, not by detokenising.** `HFTokenCounter.split` uses the
  tokenizer's character offsets to cut the original string. Decoding token
  IDs back to text would lowercase it and mangle spacing.
- **Cut only at word starts** (`tokens.split_at_word_starts`). The first full
  run had 3 chunks of 449–450 tokens against a max of 448. Cutting
  "tok|enizer" and rejoining the pieces inserts a space, which changes the
  text and its token count. In Korean text, BERT's normaliser splits one
  syllable into several tokens with the same start character, and a cut
  between them moved tokens into the next piece. The lesson: **check the
  invariant on real data**. The synthetic tests used a whitespace
  tokenizer, where the bug can't happen.

## 6. Probe: dilution and truncation

**Where:** `probes/phase-2-dilution.py` → `docs/figures/phase2-dilution.png`.

One answering paragraph from the BPE paper is embedded inside more and more
of the paper's text (128 → 1024 tokens) in two ways: answer first, and answer
last. Then each version is compared with a question.
- **Dilution:** similarity falls as unrelated text joins the chunk.
- **Truncation, answer first:** the line goes flat past 512, because
  everything after token 512 is dropped and can't change the vector.
- **Truncation, answer last:** past 512 the answer itself is what gets
  dropped, so similarity collapses. This is the silent failure the size
  invariant prevents.

**What the run showed** (`docs/figures/phase2-dilution.md`):

| Tokens | 128 | 192 | 448 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|
| Answer first | 0.779 | 0.801 | 0.800 | 0.796 | 0.796 | 0.796 |
| Answer last | 0.758 | 0.788 | 0.758 | 0.702 | 0.680 | 0.685 |

- The paragraph alone (156 tokens) scores 0.800.
- **Truncation is clear.** "Answer first" is identical from 512 on, because
  the model never sees the extra text.
- **"Answer last" loses about 0.1** once its answer is cut off.
- **Dilution is milder than expected.** The text around the answer is also
  about BPE, so adding it barely changes the vector. Dilution hurts most
  when a chunk mixes *unrelated* topics, which is exactly what packing
  across sections would do.

## 7. Writing back `included_in_index`

**Where:** `pipeline.mark`, D-021.

`mark` sets `included_in_index` in the committed manifest: true when a paper
has at least one chunk. It is the one place a stage writes outside its own
folder, documented in D-021. It refuses to run after a `--limit` chunk run,
so a smoke test can't dirty the manifest. `rag collect manifest` rewrites
the manifest without the flag, so run `rag parse mark` again after it.

---

## 8. The first full run in numbers

| | |
|---|---|
| Accepted records | 1,979: 1,781 GROBID · 1 failed (scanned, no text) · 197 no PDF |
| GROBID time | 13.5 min, 4 workers, median 1.7 s per PDF |
| Structure | 40,573 sections · 138,443 paragraphs · 5,589 tables · 7,882 figures · 13,586 formulas · 66,588 references |
| Weak spots | 84 papers without numbered headings · 10 without an abstract |
| Chunks | 93,031: paragraph 74,747 · table 8,698 · caption 7,710 · abstract 1,876 |
| Per paper | median 44, max 325 |
| Size checks | 0 over max_tokens · max model input 512 · 549 sentences cut on tokens · 1,682 formula placeholders |

Two things to keep in mind for Phase 4:
- **Duplicates.** The 1,781 indexed papers include the linked copies and
  kept v1s from Phase 1, so many chunks are near-duplicates (arXiv vs ACL
  copy of one paper). That's intentional; Phase 5 measures and removes them.
- **The spike at 448 tokens.** Whole paragraphs are kept if they fit
  `max_tokens`, and large tables fill up to the cap.

## What to look at

- `data/parsed/*/stats.json`:
  - `outcome.*`: GROBID, fallback, failed or no PDF.
  - `median_ms_per_pdf`.
  - `tables_without_rows`.
  - `split_sentences`, `formula_placeholders`.
  - `max_model_tokens`: must be ≤ 512.
- `uv run rag inspect paper <record_id>`: chunks under their headings. Look
  for missed headings (text like "2 The notion of …" inside a chunk), tiny
  chunks from fragment "sections", and tables whose rows are really caption
  lines.
- `docs/figures/phase2-chunk-tokens.png` + `phase2-chunk-stats.md`: the
  length distribution by element type, with the target and max marked.
- `docs/figures/phase2-dilution.png`: §6.

## Common pitfalls (summary)

| Pitfall | Guard here |
|---|---|
| Chunk silently truncated by the embedder | model-tokenizer counts, header budget, per-chunk check in stats |
| Chunk spans two topics | packing runs per section; overlap never crosses one |
| Mid-sentence cuts, broken citation markers | sentence-level packing and overlap |
| References indexed, matching every query | references only in `references.jsonl` |
| Hidden external calls from the parser | GROBID consolidation off |
| Formula fragment taken as a heading | `MIN_HEADING_LETTERS` |
| Garbled long formulas as noise | `[equation]` placeholder above `formula_max_chars` |
| Smoke run dirtying the committed manifest | `mark` skips after a `--limit` run |
| Losing papers GROBID can't read | PyMuPDF fallback, flagged `parser=pymupdf` |
