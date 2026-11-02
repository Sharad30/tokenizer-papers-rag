# How eval/gold/questions.jsonl was built (2026-11-02)

Sharad asked Claude to run the drafted-questions pipeline himself: link each
drafted question to passages in the current corpus, skip questions whose
passage isn't in the current chunks, and note why. So **the gold spans and
labels below were picked by Claude, not by Sharad.** Every question records
this in `provenance`: "llm_drafted; spans and labels picked by Claude at
Sharad's request (2026-11-02)". `origin` says `llm_assisted`, whose
definition assumes Sharad verified each question; that holds only once he
has reviewed them (see "Reviewing" below).

Result: **28 of 42 drafted questions** in `eval/gold/questions.jsonl`. Every
gold span maps to a chunk. 14 were skipped (listed below).

## Commands, in order

```bash
# 1. Which gold papers are in the corpus?
uv run rag eval check-papers
#    -> 55 of 64 gold entries match by id; MEGABYTE and Byte Latent Transformer are missing

# 2. Unanswerable rows: search their verify_absent terms (read-only)
uv run rag eval absent --no-write
#    Narrowed to the one paper a question is about:
uv run rag eval absent --no-write --id C4-T3 --paper arxiv:2402.14903v1
uv run rag eval absent --no-write --id U05 --paper acl:2020.acl-main.170
uv run rag eval absent --no-write --id U06 --paper acl:P16-1162
#    And by hand, for terms the draft lacked:
uv run rag eval find "claude vocabulary" --all-words      # U01
uv run rag eval find "reward model"                      # U03
uv run rag eval find "image tokenizer diffusion" --all-words   # U07

# 3. Candidate passages per gold entry (the list `rag eval resolve` shows):
#    read for every entry, then the picks were written to eval/gold/claude_resolution.yaml

# 4. Apply the picks to the draft (dry run first, then write; keeps a .bak)
uv run python eval/gold/resolve_by_claude.py
uv run python eval/gold/resolve_by_claude.py --write

# 5. Write the eval set from the verified entries and validate it
uv run rag eval finalize          # --force to replace an existing questions.jsonl
```

`eval/gold/resolve_by_claude.py` does what `rag eval resolve` does with a person
at the keyboard, through the same functions (`Corpus.match`, `check_span`,
`with_span`, `absent_hits`). Each pick in `eval/gold/claude_resolution.yaml` names
one or two phrases from the passage. The span is the paper's original text
from the start of the first phrase to the end of the last, on one line. It
is written only if `map_span` finds it in the gold paper, with the canonical
record checked first.

## What changed in the draft

`eval/gold/questions.draft.jsonl` was rewritten. The version before the write is
`eval/gold/questions.draft.jsonl.bak`: Sharad's original draft, which was never
committed.

For each of the 28 resolved questions:
- `gold[].span`, `.record_id` and `.section` were filled;
- `status` was set to `verified`;
- `provenance` was set as above.

Labels that were missing and had to be chosen:

| Question | Added | Why |
|---|---|---|
| U02 | `unanswerable_kind: false_premise` | 250k is mT5's vocabulary; ByT5 uses 256 byte values (gold span says so) |
| U04 | `not_covered` | no indexed paper is from 2027 (manifest max year: 2026); the draft had no terms to search |
| U05 | `nonexistent_detail` | no "Swahili" in either BPE-dropout record |
| U06 | `false_premise` | Sennrich et al. evaluate English→German and English→Russian; no "Hindi" in the paper |
| C4-T3 | `nonexistent_detail` | no cost figures in the Singh & Strouse paper (its one "$" is a digit separator) |
| C4-T3, U05, U06 | `absent_check` (confirmed, scoped to the paper) | corpus-wide counts can't settle a question about one paper |
| U01, U03, U07 | `absent_check` (not confirmed) | the corpus does answer them (below) |

Question text, gists, `expected_behavior`, `duplicate_sensitive` and notes
were not edited.

One tool change came out of this run (commit b057eb7). The draft's turn-1
questions carried an `oracle_rewrite` identical to the question. The schema
forbids a rewrite on turn 1, so `finalize` now drops one that only repeats
the question. A different rewrite would still be refused.

## Skipped questions (status left `draft`)

| Question(s) | Reason |
|---|---|
| L08, L09 | MEGABYTE (2305.07185) / Byte Latent Transformer (2412.09871) aren't in the corpus |
| M03 | 2 of its 5 gold papers are those two |
| C1-T1..T4 | T1, T3 and T4 need those papers; a conversation goes in whole or not at all |
| L13 | MorphPiece's fertility comparison (BPE 1.111, WordPiece 1.095, MorphPiece 1.302) is only in v1 (`arxiv:2307.07262v1`). v2 is the canonical record, so the Phase 5 mask hides v1 from every deployed retriever, and the question would fail by construction. The draft's "Table 3" in v2 is a perplexity table. |
| C2-T2 | the SentencePiece paper never mentions WordPiece (0 chunks); the gold paper needs changing |
| C2-T1, C2-T3 | held back with C2-T2 |
| U01 | **answerable**: a table in the corpus gives Claude 2.1's vocabulary as 65k |
| U03 | **partly answerable**: several chunks describe how an RLHF reward model is trained |
| U07 | **answerable**: the corpus has visual-tokenizer and diffusion papers (the image-tokenizer side of the collection gap) |

## What validate reports

```
                        have   min
  lookup                  13    12
  multi_paper              7     8  <
  followup                 4     8  <
  unanswerable             4     6  <
  duplicate_sensitive      6     3
  conversations            2     3  <
```

- **Mix shortfall:** the four mix errors come only from the skips. Recall/MRR can be computed on the 28 as they are (`rag eval run`, `rag eval ablate`), but PLAN's question mix isn't met yet.
- **Long spans:** L06, L12 and C3-T3 are two sentences each.
- **Copies:** 21 spans also appear in linked copies (arXiv vs ACL) while `duplicate_sensitive` is false. That's expected for copies of the same paper, and the harness already scores any copy. Whether to flag them is Sharad's call.

## Reviewing or changing a pick

- **Read the picks:** `eval/gold/claude_resolution.yaml` lists them; each question's span, record and section are in `eval/gold/questions.jsonl`.
- **Redo one:** `uv run rag eval resolve --id L04` (interactive), then `uv run rag eval finalize --force`.
- **Bring back a skipped question:** fix it in the draft (a new gold paper for C2-T2, new labels for U01/U03/U07, or add MEGABYTE/BLT to the corpus), set `"status": "verified"`, then finalize.
- **Once reviewed:** change `provenance` back to `llm_drafted`, so `origin: llm_assisted` is accurate.
