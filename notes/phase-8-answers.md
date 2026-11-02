# Phase 8 teaching note: evaluating the chatbot's answers

Phases 4–6 asked "did retrieval find the right passages?". Phase 8 asks "what did the
chatbot *say*, was it supported, and did it refuse when it should?".

```
eval/gold/questions.jsonl ─▶ rag eval answers ─▶ results/<date>-answers.jsonl (+ .json meta)
                                 (chat, gate off)        │
                                                         ├─▶ rag eval label   (Sharad judges)
                                                         │      └─▶ <date>-answers.labels.jsonl
                                                         └─▶ rag eval answer-report
                                                                └─▶ <date>-answer-report.{json,md}
                                                                    docs/figures/phase8-risk-coverage.png
probes/phase-8-transcript.txt ─▶ rag eval transcript ─▶ docs/phase8-transcript.md
```

Code:
- `src/rag/eval/answers.py`: the run, claims, records;
- `src/rag/eval/labels.py`: the labeller;
- `src/rag/eval/answer_report.py`: simulated gate, sweep, failure candidates, figure;
- `src/rag/chat/script.py`: the transcript;
- `src/rag/chat/citations.py` `claim_sentences`.

Decision: D-046.

First run (2026-11-02, qwen3:8b, 28 gold questions), before labels:
- **Unanswerable:** 5 of 5 handled at the deployed threshold, all by the model saying NOT_SUPPORTED.
- **False abstention:** 1 of 23 answerable questions (C4-T2).
- **Claims:** 105 in total; 36 of the 94 the deployed gate lets through are uncited (38%).
- **Invalid citations:** 0.

---

## 1. LLM-as-judge, and why here the judge is a person

The usual way to score free-text answers is to ask another LLM. It's cheap and repeatable, but
it has known biases:
- **Self-preference:** a model rates text like its own higher. That's why PLAN asks for a judge
  from a different family than the generator.
- **Verbosity:** longer answers look more thorough.
- **Position:** in pairwise comparisons, the first answer tends to win.
- **Leniency on fluent text:** a confident, well-written unsupported claim reads as supported.

So an LLM judge needs its own audit: label a sample by hand and measure agreement. Here Sharad
chose to be the judge (D-046), which removes those biases. The cost is one annotator: there's
no inter-annotator agreement, and a rerun after a change means labelling again. With about 100
claims that's practical; with thousands it wouldn't be.

## 2. Claims, not answers: decomposing for faithfulness

One label per answer hides the interesting failure, where most of the answer is right and one
sentence isn't. FActScore-style evaluation splits an answer into atomic facts and checks each
against the evidence.

Here a claim is a **sentence**:
- `claim_sentences` splits on sentence ends and line breaks, skipping fragments of fewer than 4
  words (list headers, "Yes.").
- Each claim keeps the `[Cn]` labels it cites. `rag eval label` shows exactly those passages
  in full, so you judge the claim against what the model said supports it, not the whole context.
- An **uncited** claim is judged against the whole context. Its rate (38% now) is a grounding
  problem in itself: the prompt (rule 2) asks for a citation after every factual sentence.
- Sentences are coarser than atomic facts. **partial** is the label for a sentence where one
  part is supported and another isn't.

## 3. Abstention and the risk–coverage trade-off

The chat refuses in two places (D-036):
- **Gate 1:** the best dense cosine is below 0.70 (retrieval found nothing close enough).
- **Gate 2:** the model replied NOT_SUPPORTED.

Selective answering is a trade-off:
- **coverage:** the share of questions answered;
- **risk:** the share of answered questions that are wrong.

A stricter gate lowers both. Plotting risk against coverage over thresholds gives the curve.

**One run, every threshold.** `rag eval answers` runs with gate 1 *off*, so every question gets
an answer and a recorded cosine. `answer_report.abstains(r, t)` then reproduces the engine's
decision for any t: refuse if cosine < t or the model said NOT_SUPPORTED. The sweep needs one
LLM pass, not one per threshold. The only gap is in conversations, where a refusal would change
the next turn's history.

**What the first sweep shows:**
- Below 0.80, the threshold changes nothing. All 5 unanswerable questions were refused by the
  *model*, not the gate; U04 (cosine 0.695) is the only one the gate also catches at 0.70.
- Above 0.80, answerable questions start being refused fast: false abstention 0.17 at 0.80,
  0.61 at 0.85.
- So on this set the generator, not the gate, does the work. With 5 unanswerables the sweep
  shows the shape of the trade-off but can't choose a threshold (the report says so).

## 4. False premises: abstain, or correct?

"Why did ByT5 choose a 250,000-token vocabulary?" assumes something false: 250k is mT5's
vocabulary, while ByT5 uses 256 byte values. Two behaviours are defensible:
- **refuse:** what `prompts/answer_system.md` rule 4 asks for;
- **correct the premise:** what the gold label `expected_behavior: correct_premise` expects.

The report counts either as handled (an answer counts only if labelled correct). On this run
the model refused both false-premise questions. Which behaviour you want is a prompt decision
(the Phase 7 prompt task).

## 5. Follow-ups: rewrite quality and its knock-on effects

A follow-up ("How did they force the other direction?") is unanswerable on its own. The
rewriter has to turn it into a standalone query first (D-033), and two things can go wrong:
- **No rewrite:** C4-T2 went to retrieval unchanged. Cosine 0.633, the gate refused, and an
  answerable question looked unanswerable. The report now blames the rewrite, not the gate,
  when a follow-up arrives unrewritten.
- **A wrong rewrite:** C4-T3's rewrite names an unrelated paper, plausibly taken from C4-T2's
  abstention message, which lists "nearest papers". One stage's failure feeds the next turn.

`rag eval run --rewrite` scores retrieval for raw, oracle and system rewrites. The gap between
oracle and system is what better rewriting could recover. `rag eval label` adds a human
equivalent/different judgement per rewrite.

## 6. Attributing a failure to a stage

A wrong answer is a symptom. The error taxonomy asks which stage first went wrong, checking
upstream first:

| Stage | Evidence that points to it |
|---|---|
| rewrite | follow-up sent unrewritten, or rewrite labelled *different* from the oracle |
| retrieval | gold paper and passage not in the context (`gold_*_in_context` = 0) |
| gate | refused with the gold in reach (cosine just under the threshold) |
| generation | gold passage in the context, answer still wrong or a claim unsupported |
| citations | invalid labels, or a supported fact cited to the wrong passage |

`answer_report.failures` suggests a stage from that evidence, and prints the
`rag inspect query` command that shows the retrieval behind it. The judgement is Sharad's
(`docs/phase8-failures.md`).

## What to look at

- `results/2026-11-02-answers.jsonl`: per question, the rewrite, context, cosine, raw answer
  and claims. `jq` or any JSON viewer works.
- `results/2026-11-02-answer-report.md` and `docs/figures/phase8-risk-coverage.png`. Rerun
  `rag eval answer-report` after labelling.
- `docs/phase8-transcript.md`: a 6-turn conversation with follow-ups, an unanswerable question
  and a false premise.
- `uv run rag eval label --id C4-T1`: try the labeller on one question.

## Common pitfalls

| Pitfall | Guard here |
|---|---|
| One score per answer hides the bad sentence | claim-level labels |
| An LLM judge rewarding fluent, long or self-similar text | human judge (D-046); audit needed if an LLM judge is added |
| Rerunning the LLM for every threshold | gate off in the run; thresholds simulated exactly |
| Tuning the gate on 5 unanswerables | the sweep is reported, the threshold unchanged |
| Blaming the last stage that refused | stage suggested from upstream evidence first (rewrite → retrieval → gate → generation) |
| Counting abstention on a false premise as failure | handled if refused or corrected |
| Losing labels to a crash | labels saved after each one; a rerun resumes |
