# Phase 7 demo conversation

A 5-turn session with `rag chat --show-sources`, run 2026-10-26 on the
local default model. It's a demonstration, not an evaluation: Phase 8 makes
the assessed transcript.

- Model: `qwen3:8b` via Ollama, temperature 0, thinking off
- Commit `a484d6518935` · config hash `2282d2405e2d`
- Prompts: `answer_system.md` `6078027685f9`, `rewrite_system.md` `473fb9158a51`
- Retrieval: hybrid, 6 chunks of context; retrieval gate at dense cosine 0.70

| Turn | User | Standalone query (rewrite) | Best dense cosine | Outcome | Sources | Uncited sentences | Generate (s) |
|---|---|---|---|---|---|---|---|
| 1 | What is BPE-dropout? | (not rewritten) | 0.765 | answered | 2 | 2 | 6.8 |
| 2 | How does it compare to the Unigram language model approach to subword regularization? | How does BPE-dropout compare to the Unigram language model approach to subword regularization? | 0.906 | answered | 6 | 3 | 10.5 |
| 3 | Which of the two is easier to implement? | Which of BPE-dropout and the unigram language model approach is easier to implement? | 0.795 | answered | 2 | 1 | 16.9 |
| 4 | What vocabulary size does GPT-7 use? | What vocabulary size does GPT-7 use? | 0.785 | model | 0 | 0 | 0.3 |
| 5 | How do I bake sourdough bread? | How do I bake sourdough bread? | 0.584 | retrieval_gate | 0 | 0 | 0.0 |

What the five turns show:
1. **A grounded answer.** Every source line comes from chunk and manifest
   metadata. 2 of the answer's sentences carry no citation (counted, not
   hidden).
2. **Follow-up rewriting.** "How does it compare…" became an explicit
   BPE-dropout vs Unigram query; the best cosine rose from 0.765 to 0.906.
3. **A follow-up that needed a fix.** In the first run, qwen3:8b returned
   "Which of the two is easier to implement?" unchanged, and the turn
   abstained. One worked example in `prompts/rewrite_system.md` fixed it
   (commit a484d65). The answer quotes the BPE-dropout paper's own
   comparison.
4. **A false premise (GPT-7).** It passes the retrieval gate (0.785: "vocabulary
   size" matches real passages), and the model replies NOT_SUPPORTED.
5. **Off-topic.** The best cosine is 0.584, below the gate, so no LLM call is
   made for the answer.

Duplicates are visible: turn 3's two sources are the arXiv and ACL copies of
one paper, and turn 2 cites three copies of *SubRegWeigh*. Phase 5 targets
this.

## Terminal output

```
Chat with qwen3:8b over the corpus. Commands: /reset (new conversation), /quit. Ctrl-D also quits.
Transcript: data/chat/20261026T084644Z.jsonl

you> 
BPE-dropout is a subword regularization technique designed for BPE (Byte Pair Encoding) that introduces randomness into the tokenization process. It achieves this by randomly discarding merge rules with a certain probability $ p $. This makes BPE-dropout probabilistic, allowing for multiple segmentations of a given token, unlike the deterministic nature of standard BPE [1]. The technique was proposed to improve a model's robustness to noise and has been shown to outperform SentencePiece in various languages [2]. Additionally, BPE-dropout is faster and easier to implement compared to likelihood-based approaches [2].

Sources:
  [1] Bridging the Gap between Subword and Character Segmentation in Pretrained Language Models (2023), §2.2 Subword Regularization for BPE, p.3 · acl:2023.ranlp-1.62#0010
      Subword regularization (Kudo, 2018) is a technique for improving a model's robustness to noise. To achieve this, this technique incorporates multiple segmentations of a given token into the training. BPE-dropout (Provilkov et al., 2020) is a subword regularization technique developed for BPE, which …
  [2] LCP-dropout: Compression-based Multiple Subword Segmentation for Neural Machine Translation (2022), §2.2 BPE and BPE-dropout, p.3 · arxiv:2202.13590v2#0013
      Since BPE is a deterministic algorithm, it splits a given T in one way. Thus, it is not easy to generate multiple partitions like stochastic approach (e.g., [13]). Therefore, BPE-dropout [4], ignoring the merging process with a certain probability, has been proposed. In BPE-dropout, for the current …

you> 
BPE-dropout and the unigram language model approach are both methods for subword regularization, but they differ in their implementation and objectives. The unigram language model approach, as described in Kudo (2018), samples tokenization candidates using the unigram probability of tokens to make the model more robust [1, 2, 3]. In contrast, BPE-dropout, introduced in [4, 5, 6], operates within the standard BPE framework and randomly drops some merges from the BPE merge table during training, resulting in different segmentations for the same word. BPE-dropout is shown to outperform both BPE and previous subword regularization methods on translation tasks and improves the quality of learned token embeddings and model robustness to noisy input. Thus, while both methods aim to enhance model robustness through subword regularization, BPE-dropout is a distinct technique that integrates more seamlessly with the BPE framework.

Sources:
  [1] SubRegWeigh: Effective and Efficient Annotation Weighing with Subword Regularization (2025), §2.2 Subword Regularization, p.3 · acl:2025.coling-main.130#0008
      However, there has been no research using subword regularization during inference to reduce noise in the training data.  Subword regularization using a unigram language model (Kudo, 2018) is a method to sample tokenization candidates using the unigram probability of tokens. For the cases where the m…
  [2] SubRegWeigh: Effective and Efficient Annotation Weighing with Subword Regularization (2024), §2.2 Subword Regularization, p.3 · arxiv:2409.06216v1#0010
      However, there has been no research on using subword regularization during inference to reduce noise in the training data.  Subword regularization using a unigram language model (Kudo, 2018) is a method to make the model more robust by sampling tokenization candidates using the unigram probability o…
  [3] SubRegWeigh: Effective and Efficient Annotation Weighing with Subword Regularization (2024), §2.2 Subword Regularization, p.3 · arxiv:2409.06216v2#0008
      However, there has been no research using subword regularization during inference to reduce noise in the training data.  Subword regularization using a unigram language model (Kudo, 2018) is a method to sample tokenization candidates using the unigram probability of tokens. For the cases where the m…
  [4] BPE-Dropout: Simple and Effective Subword Regularization (2019), §1 Introduction, p.2 · arxiv:1910.13267v2#0004
      Our key contributions are as follows:  • We introduce BPE-dropout -a simple and effective subword regularization method;  • We show that our method outperforms both BPE and previous subword regularization on a wide range of translation tasks;  • We analyze how training with BPE-dropout affects a mod…
  [5] BPE-Dropout: Simple and Effective Subword Regularization (2020), §1 Introduction, p.2 · acl:2020.acl-main.170#0004
      BPE-dropout is superior compared to both BPE and Kudo (2018) on a wide range of translation tasks, therefore is effective.  Our key contributions are as follows:  • We introduce BPE-dropout -a simple and effective subword regularization method;  • We show that our method outperforms both BPE and pre…
  [6] BPE-Dropout: Simple and Effective Subword Regularization (2020), §8 Conclusions, p.9 · acl:2020.acl-main.170#0037
      We introduce BPE-dropout -simple and effective subword regularization, which operates within the standard BPE framework. The only difference from BPE is how a word is segmented during model training: BPE-dropout randomly drops some merges from the BPE merge table, which results in different segmenta…

you> 
BPE-dropout is easier to implement compared to the unigram language model approach. BPE-dropout is based on and compatible with conventional BPE, requiring no additional segmentation training and using standard BPE at test time [1, 2]. In contrast, the unigram language model approach requires training a separate segmentation unigram language model using EM and Viterbi algorithms, and forbids using conventional BPE [1, 2].

Sources:
  [1] BPE-Dropout: Simple and Effective Subword Regularization (2019), §1 Introduction, p.1 · arxiv:1910.13267v2#0003
      However, the introduced approach is rather complicated: it requires training a separate segmentation unigram language model, using EM and Viterbi algorithms, and forbids using conventional BPE.  In contrast, we show that BPE itself incorporates the ability to produce multiple segmentations of the sa…
  [2] BPE-Dropout: Simple and Effective Subword Regularization (2020), §1 Introduction, p.1 · acl:2020.acl-main.170#0003
      However, the introduced approach is rather complicated: it requires training a separate segmentation unigram language model, using EM and Viterbi algorithms, and forbids using conventional BPE.  In contrast, we show that BPE itself incorporates the ability to produce multiple segmentations of the sa…

you> 
The retrieved passages don't support an answer to this.
Nearest papers (not supporting an answer): Getting the most out of your tokenizer for pre-training and domain adaptation; Training and Evaluation of a Multilingual Tokenizer for GPT-SW3; Word predictability estimates from language models are not robust to tokenizer vocabulary

you> 
The corpus doesn't seem to cover this: the closest passage scores 0.58, below the 0.70 needed to attempt an answer.
Nearest papers (not supporting an answer): fugashi, a Tool for Tokenizing Japanese in Python; Spell Once, Summon Anywhere: A Two-Level Open-Vocabulary Language Model; Beyond Token Alignment: Event Completion for Cross-Tokenizer On-Policy Distillation

you>
```
