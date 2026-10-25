import pytest
from pydantic import ValidationError

from rag.parse.config import ChunkConfig

VALID = dict(
    tokenizer_model="m",
    embedding_max_tokens=512,
    target_tokens=300,
    max_tokens=448,
    overlap_tokens=40,
    formula_max_chars=120,
    table_max_rows_per_chunk=25,
    captions="own_chunk",
)


def test_valid_chunk_sizes() -> None:
    assert ChunkConfig(**VALID).max_tokens == 448


@pytest.mark.parametrize(
    "change",
    [
        {"target_tokens": 500},  # target above max
        {"max_tokens": 511},  # no room for [CLS] and [SEP]
        {"overlap_tokens": 300},  # overlap as large as a chunk
    ],
)
def test_inconsistent_chunk_sizes_are_rejected(change: dict) -> None:
    with pytest.raises(ValidationError):
        ChunkConfig(**{**VALID, **change})
