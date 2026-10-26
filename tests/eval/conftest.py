import pytest

from rag.eval.spans import ChunkTexts
from rag.parse.schemas import Chunk


def chunk(record_id: str, n: int, text: str, path: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{record_id}#{n:04d}",
        record_id=record_id,
        element_type="paragraph",
        section_path=path or ["Method"],
        header=f"Paper {record_id}",
        text=text,
        token_count=len(text.split()),
        header_token_count=2,
        parser="grobid",
    )


# A tiny synthetic corpus: p:a and its linked copy p:a-copy share a sentence;
# p:a#0001 starts with the overlap copy of p:a#0000's last sentence.
CHUNKS = [
    chunk("p:a", 0, "We learn 32k merge operations. The vocabulary is joint.", ["Intro"]),
    chunk("p:a", 1, "The vocabulary is joint. Rare words are split into subword units."),
    chunk("p:a", 2, "| Model | BLEU |\n| BPE | 24.1 |", ["Results"]),
    chunk("p:a-copy", 0, "We learn 32k merge operations on the training data."),
    chunk("p:b", 0, "Byte-level models avoid out-of-vocabulary tokens entirely."),
    chunk("p:b", 1, "Sequences become “four times” longer — a cost."),
]


@pytest.fixture
def texts() -> ChunkTexts:
    return ChunkTexts(CHUNKS)
