# Phase 7 teaching note: grounded multi-turn chat

`rag chat` is the prototype: ask about the tokenizer literature, get an
answer built only from retrieved passages, with every source named by code,
and a plain "the corpus doesn't support this" when that's the truth.

```
you> question
  │ 1 rewrite (turn ≥ 2)           chat/rewrite.py        D-033
  │ 2 search (hybrid, k = 6)       retrieve/retriever.py  Retriever.search
  │ 3 gate 1: best cosine < 0.70?  chat/engine.py         D-036 → abstain, no LLM call
  │ 4 context [C1]..[C6]           chat/context.py        D-034
  │ 5 generate from context only   chat/llm.py + prompts/answer_system.md
  │ 6 gate 2: NOT_SUPPORTED?       chat/engine.py         D-036 → abstain
  │ 7 citations checked by code    chat/citations.py      D-035
  ▼
answer [1][2] + Sources (from metadata) + transcript line (data/chat/*.jsonl)
```

Run it: `uv run rag chat` (interactive; `/reset`, `/quit`), or
`uv run rag chat --once "…" --debug --show-sources`.
Demo: `docs/phase7-demo.md`.

---

## 1. Query rewriting ("condense question")

**Where:** `chat/rewrite.py`, `prompts/rewrite_system.md`, D-033.

A follow-up like "Which of the two is easier to implement?" is meaningless
to a retriever: BM25 sees "two", "easier", "implement", and dense embeds a
vague question. The rewriter gives the LLM the last 3 exchanges and asks
for one standalone query.

Why not embed the whole history instead? The vector would average every
earlier topic, so it drifts toward old questions, and BM25 would match on
words from three turns ago. A rewrite keeps retrieval focused on what's
being asked now.

What the demo showed:
- "How does it compare to the Unigram…" → "How does **BPE-dropout** compare
  to the Unigram…". The best cosine rose from 0.765 to 0.906.
- **A real failure, then a fix.** qwen3:8b first returned "Which of the two
  is easier to implement?" unchanged. I tested two fixes directly against
  the model: putting earlier *standalone* queries in the history didn't
  help; **one worked example in the prompt did**. Small models follow
  examples far better than rules.
- If the rewrite errors or comes back empty, the raw message is used and
  the debug view says so (`RewriteResult.fallback`), so a broken rewriter
  can't silence the chat.
- `rag eval run --rewrite` scores these rewrites (the "system" variant)
  against the raw follow-up and your oracle rewrite, once questions exist.

## 2. Context assembly and "lost in the middle"

**Where:** `chat/context.py`, D-034.

The top 6 hybrid chunks become `[C1]`…`[C6]`, best first, each headed
`Title (Year) > §3.2 Section, p.N`. About 1,000–1,500 prompt tokens.

- **Best first:** models attend most to the start and end of a long
  context, and least to the middle. With 6 short chunks the effect is
  small; it grows with context length. Testing "best at both ends" is a
  Phase 9 experiment.
- **No chat history in the answer prompt.** The generator sees only the
  standalone question and the context. History reaches the answer only
  through the rewrite. Anything the model says must come from the
  passages, not from an earlier answer that might itself be wrong.

## 3. Grounding vs correctness

**Where:** `prompts/answer_system.md`.

These are different things:
- **Grounded but wrong:** the answer faithfully reports a passage that is
  itself wrong, outdated, or about something else.
- **Correct but ungrounded:** the model knows the answer from training, but
  no retrieved passage supports it. The user can't check it.

The prompt asks for grounding: only the passages, a citation after every
factual sentence, and no outside knowledge. Phase 8 measures both
properties separately.

## 4. Citations by id, mapped by code

**Where:** `chat/citations.py` (`resolve`), `context.source_line`, D-035.

The model writes `[C3]`; code does the rest:
- **Renumbers** `[C3] [C1]` → `[1] [2]` in order of first use.
- **Removes ids that weren't in the context** and reports them
  (`CitedAnswer.invalid`). An invented `[C9]` is a grounding failure, so it
  is counted rather than quietly fixed.
- **Builds every source line from metadata:** title and year from the
  manifest, section and page from the chunk. A model asked to write titles
  will sometimes garble or invent them. Here it can't name a paper that
  wasn't retrieved.
- **Counts uncited sentences** (4+ words, no `[Cn]`). The demo's answers had
  2, 3 and 1. qwen3:8b often writes a summary sentence without a citation.
  Phase 8 decides how much that matters.

What code *can't* check: whether `[C2]` actually supports the sentence it
follows. That's Phase 8's groundedness judgment.

## 5. Abstention: two layers

**Where:** `ChatEngine.turn`, `probes/phase-7-gate.py`, D-036.

**Gate 1, retrieval.** If the best dense cosine **anywhere in the corpus**
(`Retrieval.top_scores["dense"]`, computed over all chunks, not just the
hits) is below `chat.abstain.min_dense_cosine`, the chat says so and makes
no LLM call for an answer.

The probe set the threshold from 30 queries written for it (not eval gold):

| Group | Best cosine |
|---|---|
| in-domain | 0.800 – 0.941 |
| near-domain (false premise, made-up models) | 0.700 – 0.838 |
| off-topic | 0.541 – 0.661 |

The midpoint between off-topic and in-domain is 0.73. The gate is set at
**0.70**: 12 in-domain probe queries are few, a wrongly refused real
question is lost, and a wrongly passed one still meets gate 2.

**Why not threshold RRF?** RRF scores encode rank, not relevance: the top
chunk of *any* query gets about 2/61. A cosine says how close the best text
actually is.

**bge cosines are compressed.** Even "What is the plot of Hamlet?" scores
0.54. Absolute values mean little; only the separation between groups
does. A different embedding model needs a new probe run.

**Gate 2, the model.** Near-domain questions mostly pass gate 1: "What
vocabulary size does GPT-7 use?" scores 0.785, because "vocabulary size"
matches real passages. The answer prompt asks for exactly `NOT_SUPPORTED`,
and qwen3:8b did that. It sometimes explains first and appends the token,
so `_is_not_supported` looks for it anywhere in the reply (a bug the
sourdough test caught).

## 6. LLM clients

**Where:** `chat/llm.py`, D-032.

- **One interface,** `complete(system, messages, max_tokens) -> Completion`,
  for three clients:
  - `OllamaClient`: httpx on `/api/chat`, `think: false`, temperature 0;
  - `AnthropicClient`: the SDK; the key comes from the environment only;
  - `ScriptedLLM`: for tests.
- **Local by default:** qwen3:8b (5.2 GB). Free, offline, and chunks never
  leave the machine. 7–17 s per answer here; the first call also loads the
  model (~25 s).
- **Switch with** `--provider anthropic` once `ANTHROPIC_API_KEY` is in
  `.env` (`claude-haiku-5-5`).
- **The embedding model loads from the local cache** (Phase 4 fix), so a
  chat session makes no network calls on the Ollama path.

## 7. Transcripts

**Where:** `chat/transcript.py`.

One JSONL file per session in `data/chat/` (gitignored):
- **a header:** commit, config hash, model, and the sha256 of both prompts;
- **one line per turn:** raw input, rewrite, every context chunk with its
  BM25/dense/RRF scores, best corpus scores, gate outcome, raw and shown
  answer, sources, invalid and uncited counts, tokens, timings.

Phase 8 reads these. Your prompt-change experiment is traceable through
the prompt hash.

---

## What to look at

- `uv run rag chat --debug`: per turn, the rewrite, the best scores
  against the gate, the context table (BM25 and dense ranks for each
  chunk), the raw model answer before renumbering, and the
  invalid/uncited counts.
- `--show-sources`: an excerpt of each cited chunk under its source line.
  Read it against the sentence that cites it: does it really say that?
- `docs/figures/phase7-gate.png`: why one threshold can stop off-topic
  questions but not false premises.
- `docs/phase7-demo.md`: the five behaviours in one session.

## Common pitfalls (summary)

| Pitfall | Guard here |
|---|---|
| Follow-up retrieved literally ("the two") | rewrite with a worked example; fallback flagged |
| Old topics leaking into retrieval | rewrite instead of embedding the history |
| Answer drawn from an earlier (possibly wrong) answer | generator sees context + standalone question only |
| Invented or garbled references | model cites ids; code writes every source line |
| Citation to a passage not in the context | removed and reported as invalid |
| Thresholding RRF scores | gate on the best dense cosine over the whole corpus |
| Near-domain false premises answered confidently | gate 2: NOT_SUPPORTED, detected anywhere in the reply |
| A secret in config | key name in config, value only in the environment |
| Smoke runs overwriting real data | `make smoke` runs in `.smoke/` |
