# Study guide: theory before each phase

A short, targeted study block before each phase, so you can judge the alternatives Claude Code proposes in plan mode and write a real prediction before running anything. Pairs with `PLAN.md`.

---

## How to study for this project

**Just enough, just in time.** Study before a phase only what you need to make that phase's decisions: about 45–90 minutes. Theory sticks much better once you've seen it run, so each phase also lists "deepen after" reading for once the code exists. Don't front-load all the reading on day one.

**Two kinds of knowledge.** You need RAG and information-retrieval techniques (most phases), and you need the domain: tokenization itself. You can't draw the topic boundary, write good eval questions, or tell whether an answer is right without the domain. That's why Phase 0's study is the domain primer, and it's the most important block in this guide.

**Write a pre-note.** After studying, write 5–10 lines in your own words in `notes/phase-N-prenote.md`: what the technique does, why it should work here, and what could go wrong. After the phase, compare it with Claude Code's teaching note. The differences show you what you misunderstood.

**Predictions don't need to be right.** You won't know what to expect for techniques you're new to, and that's fine. Write "my guess is X, low confidence, because Y." The value is in having a stated expectation that the result can surprise. A wrong guess with a reason teaches you more than no guess, and the gaps between guesses and results are your best material for the journey write-up.

**Use Claude chat as a tutor, separate from Claude Code.** Before approving a phase plan, ask Claude to walk you through the decision points: "here are the three options Claude Code proposed for chunking; quiz me on the trade-offs." Ask it to question you rather than explain everything.

**Readiness check.** Each phase ends with "Ready when" checks. If you can answer them, start the phase. If not, study a little more, or start anyway and use the phase's concept probe to fill the gap.

Rough total: about 10–12 hours of study across all phases, on top of the build time.

---

## Phase 0: Domain primer on tokenization (most important)

**Why now.** Every later phase depends on knowing the domain: the topic boundary (Phase 1), the eval questions (Phase 3), and judging answers (Phase 8).

**Must (about 3–4 hours, can be split across the first day).**
- Andrej Karpathy, *Let's build the GPT Tokenizer* (video). BPE built from scratch; the clearest intuition for how tokenizers work and fail.
- Hugging Face NLP course, the tokenizers chapter: the BPE, WordPiece and Unigram sections. How the three algorithms differ in training and in segmentation.
- Mielke et al., 2021, *Between words and characters: A brief history of open-vocabulary modeling and tokenization in NLP*. Skim for the map of the field and its vocabulary.

**Then read the abstract and introduction of these anchor papers** (they become your snowball seeds in Phase 1):
- Sennrich et al., 2016, BPE for neural machine translation.
- Kudo, 2018, subword regularization (the Unigram LM tokenizer); Kudo and Richardson, 2018, SentencePiece.
- Xue et al., 2022, ByT5; Clark et al., 2022, CANINE (byte/character-level and tokenizer-free).
- Rust et al., 2021, *How Good is Your Tokenizer?* (fertility, multilingual).
- Petrov et al., 2023, *Language Model Tokenizers Introduce Unfairness Between Languages*.

**Ready when** you can explain:
- How BPE, WordPiece and Unigram differ, in a sentence each.
- What "fertility" means and why it matters for low-resource languages.
- What byte-level and tokenizer-free models trade away to avoid a fixed vocabulary (sequence length, compute).
- Three questions you'd expect this literature to answer, and one it probably can't.

---

## Phase 1: Corpus collection

**Study goal.** Understand how to build a literature corpus systematically and how to measure the quality of a relevance filter.

**Must (about 45 minutes).**
- Wohlin, 2014, *Guidelines for snowballing in systematic literature studies*. Backward and forward snowballing, and the biases each introduces.
- The arXiv API user manual and Semantic Scholar Graph API docs: query syntax, rate limits, which fields carry linked identifiers.
- Refresh: precision and recall of a classifier, and how to estimate them from a labelled sample.

**Deepen after.** How arXiv versioning and identifiers work (`2301.12345v3`), DOIs, and how Semantic Scholar links records across sources.

**Ready when** you can explain:
- Why citation snowballing finds papers keyword search misses, and what bias it adds (towards well-cited work).
- How you'd estimate the filter's recall when you don't know the true set of relevant papers.
- Prediction for the journal: roughly how many relevant papers you expect, and which source will contribute most.

---

## Phase 2: Parse and chunk

**Study goal.** Understand why PDFs are hard to parse, and the trade-offs of chunk size and chunk context.

**Must (about 60 minutes).**
- GROBID documentation, principles section: what it extracts, how it labels document zones, what TEI output looks like. Optionally Lopez, 2009, the original GROBID paper.
- Anthropic, *Introducing Contextual Retrieval* (blog post): why chunks lose meaning out of context, and contextual embeddings and contextual BM25 as fixes.
- sentence-transformers docs on `max_seq_length`: what happens to text past the limit.

**Deepen after.** Compare GROBID's TEI with the original PDF for one paper you know well; read about layout-aware parsers (Nougat, Marker) to know the alternatives you didn't use.

**Ready when** you can explain:
- Why one embedding for a long, multi-topic chunk retrieves worse than one for a short, focused chunk.
- What silent truncation is and how you'd detect it.
- Why reference lists hurt retrieval if indexed but help duplicate detection if kept.
- Prediction: which element types (tables, captions, equations) will parse worst.

---

## Phase 3: Evaluation set

**Study goal.** Understand how retrieval evaluations are designed and what makes a question set biased or weak.

**Must (about 60 minutes).**
- Manning, Raghavan and Schütze, *Introduction to Information Retrieval* (free online), chapter 8 "Evaluation in information retrieval": test collections, relevance judgments, precision/recall at k.
- Voorhees, 2002, *The Philosophy of Information Retrieval Evaluation*: the Cranfield paradigm and why reusable test collections work.
- Rajpurkar et al., 2018, *Know What You Don't Know: Unanswerable Questions for SQuAD* (SQuAD 2.0), introduction and data-collection sections: how to write unanswerable questions that are hard rather than obviously off-topic.

**Deepen after.** Pooling bias in TREC collections; how conversational QA datasets construct follow-up questions.

**Ready when** you can explain:
- What pooling bias is, and how writing questions before seeing retriever output avoids it.
- The difference between a hard unanswerable question and an easy one.
- Why one question type (for example, only lookups) would make the embedding comparison misleading.

---

## Phase 4: Baseline retrieval and eval harness

**Study goal.** Understand the two retrieval families, how to combine them, and how to tell whether a difference between systems is real.

**Must (about 90 minutes; the most theory-heavy phase).**
- *Introduction to Information Retrieval*, chapter 6 (term weighting, TF-IDF) and section 11.4.3 (Okapi BM25). For depth: Robertson and Zaragoza, 2009, *The Probabilistic Relevance Framework: BM25 and Beyond*.
- Karpukhin et al., 2020, *Dense Passage Retrieval for Open-Domain Question Answering*: bi-encoders, contrastive training with in-batch negatives.
- Cormack, Clarke and Büttcher, 2009, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*: short; the whole idea fits on a page.
- Thakur et al., 2021, *BEIR*: introduction and results. Why BM25 remains a strong baseline out of domain, and where dense models fail.

**Deepen after.** Smucker, Allan and Carterette, 2007, *A comparison of statistical significance tests for information retrieval evaluation*: bootstrap and paired tests for small evaluation sets. MRR and nDCG definitions in IIR chapter 8.

**Ready when** you can explain:
- What k1 and b do in BM25, in words.
- Why dense retrieval can miss a query containing a rare exact term like "ByT5".
- Why RRF works without normalising scores, and what its k constant changes.
- Prediction: BM25 vs dense vs hybrid Recall@5 on your eval set, and which question types each will win.

---

## Phase 5: Versions and near-duplicates

**Study goal.** Understand similarity hashing, clustering, and diversification, and the difference between identity and similarity.

**Must (about 90 minutes).**
- Leskovec, Rajaraman and Ullman, *Mining of Massive Datasets* (free online), chapter 3 "Finding Similar Items": shingling, Jaccard, MinHash, LSH banding and the S-curve. The core reading for this phase.
- Carbonell and Goldstein, 1998, *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*: short; the MMR formula and intuition.
- Refresh union-find (disjoint-set) from any algorithms reference, and think through why transitive merging is risky.

**Deepen after.** Lee et al., 2022, *Deduplicating Training Data Makes Language Models Better*: MinHash deduplication at scale in an LLM setting. Broder, 1997, *On the resemblance and containment of documents*, the original MinHash paper.

**Ready when** you can explain:
- Why the probability that two MinHash values match equals the Jaccard similarity.
- How changing the number of bands and rows moves the LSH threshold.
- Why two papers on fertility can be near-identical in embedding space but must not be merged.
- How union-find can merge two papers that aren't similar.
- Prediction: how much the duplicate rate drops at each ablation step, and whether Recall@5 rises or falls.

---

## Phase 6: Embedding comparison and hybrid search

**Study goal.** Understand how embedding models are trained, why they differ by domain, and the bi-encoder vs cross-encoder trade-off.

**Must (about 90 minutes).**
- Reimers and Gurevych, 2019, *Sentence-BERT*: siamese bi-encoders for sentence embeddings.
- Cohan et al., 2020, *SPECTER*: training on citation links for document-level embeddings. Then Singh et al., 2023, *SciRepEval* (introduction): SPECTER2 and its task-specific adapters.
- Wang et al., 2022, *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5) or Xiao et al., 2023, *C-Pack* (BGE), introduction and training sections: why these models use query/passage prefixes.
- Nogueira and Cho, 2019, *Passage Re-ranking with BERT*, plus the sentence-transformers "retrieve and re-rank" docs: cross-encoders and two-stage retrieval.

**Deepen after.** Muennighoff et al., 2022, *MTEB*: how embedding benchmarks are built, and why a leaderboard rank may not transfer to your corpus.

**Ready when** you can explain:
- Why SPECTER's training objective might not suit question-to-passage retrieval.
- What the `query:` prefix does and what happens if you omit it.
- Why a cross-encoder is more accurate but can't search the whole corpus.
- Prediction: which model wins, by how much, and whether the scientific model beats the general one.

---

## Phase 7: Grounded multi-turn chat

**Study goal.** Understand query rewriting, how LLMs use (and misuse) retrieved context, and how to make citations trustworthy.

**Must (about 75 minutes).**
- Lewis et al., 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*: introduction only, for the original framing.
- Elgohary et al., 2019, *Can You Unpack That? Learning to Rewrite Questions-in-Context* (CANARD): why follow-up questions need rewriting and what good rewrites look like.
- Liu et al., 2023, *Lost in the Middle*: how position in the context affects what the model uses.
- Gao et al., 2023, *Enabling Large Language Models to Generate Text with Citations* (ALCE): citation quality, citation recall and precision.

**Deepen after.** Gao et al., 2022, *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE), as an alternative to rewriting.

**Ready when** you can explain:
- Why embedding the whole conversation history retrieves worse than a rewritten standalone query.
- The difference between an answer being grounded and being correct.
- Why the LLM should cite chunk IDs and leave titles to code.
- Prediction: what fraction of follow-ups the rewriter resolves correctly.

---

## Phase 8: Answer evaluation and failure analysis

**Study goal.** Understand how to measure faithfulness, the limits of LLM judges, and abstention as a trade-off.

**Must (about 75 minutes).**
- Zheng et al., 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*: the section on judge biases (position, verbosity, self-enhancement).
- Min et al., 2023, *FActScore*: decomposing answers into atomic claims and scoring each.
- Es et al., 2023, *RAGAS*: faithfulness and answer relevance metrics for RAG (also to know what you chose not to use).
- Kamath, Jia and Liang, 2020, *Selective Question Answering under Domain Shift*: abstention, coverage, and risk–coverage curves.

**Deepen after.** Geifman and El-Yaniv, 2017, *Selective Classification for Deep Neural Networks*, the formal basis of risk–coverage.

**Ready when** you can explain:
- Why you audit an LLM judge against human labels, and what agreement level you'd accept.
- What a risk–coverage curve shows that a single abstention rate doesn't.
- Prediction: groundedness rate, abstention rate on unanswerable questions, and which pipeline stage will cause most failures.

---

## Phase 9: Deep dives

Study depends on which deep dives you pick. For each, before starting: search for 2–3 papers on the same question (for example, tokenization effects on dense retrieval, or claim attribution in surveys), read their introductions and results, and write down how your experiment differs. This also tells you whether your finding is new or a replication, which matters for how you present it.

---

## Phases 10–11: Write-ups

No new theory. Re-read your pre-notes alongside the teaching notes and journal: the places where your understanding changed are what the journey write-up is about.