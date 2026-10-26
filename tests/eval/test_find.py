from rag.eval.find import find, render
from rag.eval.spans import ChunkTexts


def ids(result) -> list[str]:
    return [h.chunk.chunk_id for h in result.hits]


def test_hits_come_in_corpus_order(texts: ChunkTexts) -> None:
    result = find(texts, "the vocabulary", max_hits=10)
    assert ids(result) == ["p:a#0000", "p:a#0001"]


def test_match_ignores_case_and_quotes(texts: ChunkTexts) -> None:
    assert ids(find(texts, 'become "FOUR times"', max_hits=10)) == ["p:b#0001"]


def test_paper_filter(texts: ChunkTexts) -> None:
    assert ids(find(texts, "32k merge", max_hits=10, paper="p:a-copy")) == ["p:a-copy#0000"]


def test_all_words_matches_in_any_order(texts: ChunkTexts) -> None:
    assert find(texts, "subword rare", max_hits=10).total == 0
    assert ids(find(texts, "subword rare", max_hits=10, all_words=True)) == ["p:a#0001"]


def test_hit_cap_keeps_the_total(texts: ChunkTexts) -> None:
    result = find(texts, "e", max_hits=2)
    assert len(result.hits) == 2
    assert result.total == 6
    assert "6 matching chunks (showing the first 2" in render(result, "e")


def test_no_hits(texts: ChunkTexts) -> None:
    assert render(find(texts, "klingon", max_hits=3), "klingon") == "No chunk contains 'klingon'."
