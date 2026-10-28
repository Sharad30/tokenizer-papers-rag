# Eval set

`questions.jsonl` holds the retrieval and answer eval questions, one JSON
object per line. Sharad writes them (PLAN Phase 3: "You do. All of it");
the tooling only checks them. Never edit gold labels here with a script or
an LLM.

## Workflow

1. **Write the question first**, from reading the papers, before any
   retriever output exists (that avoids pooling bias).
2. **Find the evidence:** `uv run rag eval find "<words from the paper>"`
   (add `--paper <record_id>` to narrow, `--all-words` to match words in any
   order). Hits come in corpus order, never ranked.
3. **Copy a span** from the hit's excerpt into `gold`. Keep it within one
   sentence, the shortest text that supports the answer. Case, quotes, dashes
   and table pipes don't matter; spelling and hyphenation do.
4. **Validate:** `uv run rag eval validate`. It fails on errors and lists
   warnings to look at.

## Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique, e.g. `q001`. |
| `type` | yes | `lookup` (one fact, one paper), `multi_paper` (evidence from ≥ 2 papers), `followup` (turn ≥ 2 of a conversation), `unanswerable`. |
| `question` | yes | As a user would ask it. A followup may say "that model". |
| `conversation_id`, `turn` | together | Conversation name and 1-based turn. Turn 1 is typed by what it asks (e.g. `lookup`). |
| `oracle_rewrite` | followups | The standalone question a perfect rewriter would produce. |
| `answerable` | yes | False only for `unanswerable`. |
| `unanswerable_kind` | unanswerable | `not_covered`, `false_premise`, `outside_corpus` (a real paper we didn't collect), `nonexistent_detail` (a number or result no paper reports). |
| `gold` | answerable | List of `{record_id, section?, span}`. `record_id` is the manifest id; `section` is optional and only checked as a warning. |
| `duplicate_sensitive` | no | True when the answer is restated in other papers or copies. `validate` warns when a span appears elsewhere and this is false. |
| `restatement_notes` | no | Which other papers restate it. |
| `origin` | no | `hand` (default) or `llm_assisted` if an LLM drafted it and you rewrote it. |
| `notes` | no | Anything else. |

## Example lines

```json
{"id": "q001", "type": "lookup", "question": "What does BPE iteratively replace?", "answerable": true, "gold": [{"record_id": "acl:P16-1162", "section": "Byte Pair Encoding", "span": "iteratively replaces the most frequent pair of bytes"}], "duplicate_sensitive": true, "restatement_notes": "the BPE definition is restated in 12 other records"}
{"id": "q020", "type": "unanswerable", "question": "...", "answerable": false, "unanswerable_kind": "false_premise"}
```

## Target mix (config `eval.mix_min`)

At least 12 lookups (several answered by tables), 8 multi-paper, 8
followups across at least 3 conversations, 6 unanswerable, 3
duplicate-sensitive. `validate` prints have vs. minimum.

# Labelled pairs (Phase 5)

`pairs.jsonl` says which pairs of records are the same paper. Sharad labels
them; scripts never write this file. It does two jobs:

- **Judge for dup@5** (D-042): a pair labelled `same` joins the identity
  groups, next to shared identifiers. Copies nobody labelled count as
  different papers, so dup@5 is a lower bound.
- **Resolver precision and recall:** `uv run rag dedup score` compares
  `rag dedup run`'s works with the labels.

## Workflow

1. `uv run rag dedup pairs` writes `pairs_candidates.jsonl`: about 55 pairs
   drawn from four bands (joined by IDs, joined by title or text, near
   misses, similar abstracts by different authors). It shows titles,
   authors, venues, links and the evidence (title ratio, author overlap,
   text Jaccard), but not what the resolver decided.
2. For each candidate, open both papers and decide. Write one line per
   pair to `pairs.jsonl`.
3. `uv run rag dedup score`, then review the disagreements it lists.

## Fields

| Field | Meaning |
|---|---|
| `a`, `b` | The two record ids, copied from the candidate. |
| `same` | True when, for a reader, the two are one paper: a duplicate if both are in a top-5. |
| `kind` | `version`, `venue_copy` (preprint vs published, or one paper at two venues), `extended` (conference vs extended journal version), `follow_up`, `companion`, `survey`, `distinct`. |
| `notes` | Optional: what decided it. |

Whether an `extended` version counts as `same` is a policy call: record it
in D-040 and label consistently.

```json
{"a": "acl:2020.emnlp-main.203", "b": "arxiv:2004.14280v1", "same": true, "kind": "venue_copy", "notes": "title changed for the camera-ready"}
{"a": "acl:D18-1473", "b": "acl:P19-1148", "same": false, "kind": "follow_up"}
```
