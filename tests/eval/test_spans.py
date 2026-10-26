from rag.eval.schemas import GoldSpan
from rag.eval.spans import ChunkTexts, elsewhere, map_span, normalise, suggest


def test_normalise_folds_case_quotes_dashes_ligatures_and_space() -> None:
    assert normalise("  “The ﬁrst” – BPE\n|  ") == '"the first" - bpe'


def test_normalise_drops_table_pipes() -> None:
    assert normalise("| BPE | 24.1 |") == "bpe 24.1"


def test_span_maps_with_typed_quotes_and_case(texts: ChunkTexts) -> None:
    gold = GoldSpan(record_id="p:b", span='sequences become "FOUR times" longer - a cost')
    assert map_span(gold, texts) == ["p:b#0001"]


def test_overlap_copy_maps_to_both_chunks(texts: ChunkTexts) -> None:
    gold = GoldSpan(record_id="p:a", span="The vocabulary is joint.")
    assert map_span(gold, texts) == ["p:a#0000", "p:a#0001"]


def test_span_maps_only_within_its_paper(texts: ChunkTexts) -> None:
    gold = GoldSpan(record_id="p:b", span="We learn 32k merge operations")
    assert map_span(gold, texts) == []


def test_table_cells_match_without_pipes(texts: ChunkTexts) -> None:
    assert map_span(GoldSpan(record_id="p:a", span="BPE 24.1"), texts) == ["p:a#0002"]


def test_unknown_paper_maps_to_nothing(texts: ChunkTexts) -> None:
    assert map_span(GoldSpan(record_id="p:zzz", span="joint"), texts) == []


def test_elsewhere_finds_the_linked_copy(texts: ChunkTexts) -> None:
    assert elsewhere("we learn 32k merge operations", texts, exclude="p:a") == ["p:a-copy"]


def test_suggest_points_at_the_near_miss(texts: ChunkTexts) -> None:
    # Typed from the PDF with a different word: no exact match, but a hint.
    hint = suggest("Rare words are segmented into subword units", texts.paper("p:a"))
    assert hint is not None
    assert hint.chunk_id == "p:a#0001"
    assert "rare words are split" in hint.excerpt
    assert 0.7 < hint.similarity < 1


def test_suggest_on_empty_paper_is_none() -> None:
    assert suggest("anything", []) is None
