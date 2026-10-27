"""Config models for the parse stage (the `parse:` section of config.yaml)."""

from typing import Literal

from pydantic import Field, model_validator

from rag.common.schemas import StrictModel


class GrobidConfig(StrictModel):
    url: str  # the local GROBID service, e.g. http://localhost:8070
    image: str  # the Docker image `make grobid-up` runs (D-013)
    # Parallel requests. GROBID answers 503 when all its threads are busy.
    workers: int = Field(ge=1)
    timeout_s: float = Field(gt=0)
    # GROBID's docs suggest retrying a 503 after 5-10 s.
    busy_retry_s: float = Field(ge=0)
    max_attempts: int = Field(ge=1)
    # 0 = off. Consolidation calls CrossRef from inside GROBID (D-014).
    consolidate_header: Literal[0, 1, 2, 3]
    consolidate_citations: Literal[0, 1, 2]
    # Elements GROBID annotates with coords="page,x,y,w,h": gives each paragraph its page.
    tei_coordinates: list[str]


class FallbackConfig(StrictModel):
    enabled: bool  # PyMuPDF when GROBID returns no content or an error (D-015)


class ChunkConfig(StrictModel):
    # Token counts use the embedding model's own tokenizer (D-016).
    tokenizer_model: str
    embedding_max_tokens: int = Field(ge=8)
    target_tokens: int = Field(ge=1)
    # Hard cap on a chunk's body; the header must also fit (checked per chunk).
    max_tokens: int = Field(ge=1)
    overlap_tokens: int = Field(ge=0)  # whole sentences, within a section (D-017)
    formula_max_chars: int = Field(ge=0)  # longer display formulas -> placeholder (D-019)
    table_max_rows_per_chunk: int = Field(ge=1)  # header row repeated in each part (D-020)
    captions: Literal["own_chunk", "drop"]  # D-018

    @model_validator(mode="after")
    def _sizes_are_ordered(self) -> "ChunkConfig":
        # 2 = the [CLS] and [SEP] tokens the model adds around every input.
        if not self.target_tokens <= self.max_tokens <= self.embedding_max_tokens - 2:
            raise ValueError("need target_tokens <= max_tokens <= embedding_max_tokens - 2")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        return self


class CleanConfig(StrictModel):
    # Bibliography text GROBID left in body paragraphs (D-039): a paragraph's
    # tail is cut when it holds at least this many entry pieces ...
    min_entry_sentences: int = Field(ge=1)
    # ... and they make up at least this share of the tail's sentences.
    min_entry_share: float = Field(gt=0, le=1)
    # A first sentence this short, right before the cut, is the end of an entry
    # split by a column or page break, and goes too.
    max_fragment_words: int = Field(ge=0)


class ParseConfig(StrictModel):
    grobid: GrobidConfig
    fallback: FallbackConfig
    clean: CleanConfig
    chunk: ChunkConfig
