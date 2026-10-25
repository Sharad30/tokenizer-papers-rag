import pytest

from rag.parse.config import ChunkConfig


class WhitespaceCounter:
    """Stands in for the model tokenizer: one token per whitespace-separated word."""

    def count(self, text: str) -> int:
        return len(text.split())

    def split(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


@pytest.fixture
def counter() -> WhitespaceCounter:
    return WhitespaceCounter()


@pytest.fixture
def chunk_config() -> ChunkConfig:
    # Small numbers so hand-built papers hit every rule: header room = 40 - 2 - 30 = 8.
    return ChunkConfig(
        tokenizer_model="test/whitespace",
        embedding_max_tokens=40,
        target_tokens=20,
        max_tokens=30,
        overlap_tokens=5,
        formula_max_chars=20,
        table_max_rows_per_chunk=2,
        captions="own_chunk",
    )
