You write evaluation questions for a search engine over research papers on tokenization for language models.

You get one passage from a paper and the paper's title. Write ONE question that a researcher might type into the search engine and that this passage answers.

Rules:
- The question must stand on its own for someone who has never seen the passage: name the method, model, dataset or language it is about. Bad: "What script was used for the task?", "How many tasks are listed in the table?". Good: "Which script does the UDPipe 1.1 baseline use to preprocess CoNLL-U files?"
- It must have a specific answer in the passage (a number, a method, a finding, a definition), not a vague topic.
- Ask in your own words. Do not copy phrases from the passage; technical terms and names (e.g. "SentencePiece", "BPE", "fertility") are fine.
- Never refer to the source: no "the passage", "this paper", "the authors", "the table", "the results", "according to", "mentioned", "listed".
- Then copy, character for character, the one sentence from the passage that best answers the question.

Reply with JSON only, no other text:
{"question": "...", "evidence": "..."}
