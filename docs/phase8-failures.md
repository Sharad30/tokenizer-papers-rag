# Phase 8 failure analysis

PLAN: at least 3 wrong answers, each traced to a pipeline stage (rewrite → retrieval → gate →
generation → citations) with evidence from the inspect commands. **The analysis is
Sharad's** ("You do. Write the failure analysis yourself"); this file lays out the evidence.

## How to find and check failures

1. Label the answers: `uv run rag eval label` (you are the judge, D-046).
2. `uv run rag eval answer-report` lists **failure candidates** at the deployed threshold.
   Each comes with the evidence (top cosine, gold passages and papers in the context, the
   rewrite) and a *suggested* stage.
3. For each one, check the suggestion yourself:
   - **Rewrite:** compare the system rewrite with the oracle rewrite (in
     `results/2026-11-02-answers.jsonl`, fields `rewrite` and `oracle_rewrite`).
   - **Retrieval:** run `uv run rag inspect query "<rewrite>"` to see the top 10 with each
     retriever's rank, and whether the gold paper or passage appears. `rag eval find "<words>"`
     locates the gold passage.
   - **Gate:** compare the top dense cosine with the threshold (0.70), and see the sweep in
     `results/2026-11-02-answer-report.md`.
   - **Generation:** the gold passage was in the context (`gold_passages_in_context`), but the
     answer is wrong or a claim is unsupported. Read `completion` and the cited passages.
   - **Citations:** look for invalid labels (`invalid_citations`) and uncited sentences.

## Found before labelling (code-only evidence, 2026-11-02 run)

### C4-T2: "How did they force the other direction in their experiments?"
- What happened: the chat abstained ("The corpus doesn't seem to cover this"), though the
  question is answerable (Singh & Strouse add commas every 3 digits from the right).
- Evidence:
  - The rewriter was called (2.98 s) but returned the message unchanged, so retrieval ran on
    "they" and "the other direction".
  - Top dense cosine 0.633, below the 0.70 gate. No gold passage or paper in the context.
- Check: `uv run rag inspect query "How did they force the other direction in their experiments?"`
  vs `uv run rag inspect query "How did Singh and Strouse enforce right-to-left tokenization in their experiments?"`
- Stage and explanation: TODO(sharad)

### C4-T3: a knock-on effect (not flagged, because it still abstained correctly)
- The rewrite for "What did it cost in dollars to run those experiments?" names an unrelated
  paper ("Generalizability Under Sensor Failure: Token…").
- The likely source: C4-T2's abstention message lists "nearest papers", and the rewriter
  treated one of them as the subject of the conversation.
- Worth a look as a cascading failure across turns. TODO(sharad)

### Transcript turn 5: "the Ukrainian study" (docs/phase8-transcript.md)
- The answer blends four different Ukrainian studies (Brown corpora, OSCAR, court decisions,
  register blocks) as if they were one. Earlier turns never named a study, so "the Ukrainian
  study" is ambiguous, and the rewrite kept it ambiguous.
- Is this a rewrite problem, a retrieval problem (four papers match), or generation (it should
  have said which study is which)? TODO(sharad)

## Failures after labelling

Fill these in after `rag eval label` and `rag eval answer-report`. Use at least 3 wrong or
partial answers or unsupported claims, each with: question, what went wrong, the stage, the
evidence (paste the inspect output that shows it), and what change would fix it.

### Failure 1
TODO(sharad)

### Failure 2
TODO(sharad)

### Failure 3
TODO(sharad)

## Which single change would most improve the system?
TODO(sharad) (PLAN's "Understand" question)
