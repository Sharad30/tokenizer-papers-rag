from rag.retrieve.text import STOPWORDS, chunk_text, terms
from tests.retrieve.conftest import make_chunk

EN = STOPWORDS["en"]


def test_compounds_keep_whole_form_and_parts() -> None:
    assert terms("Byte-level BPE and GPT-2", EN, split_compounds=True) == [
        "byte-level",
        "byte",
        "level",
        "bpe",
        "gpt-2",
        "gpt",
        "2",
    ]


def test_without_splitting_compounds_stay_whole() -> None:
    assert terms("BPE-dropout on ByT5", EN, split_compounds=False) == ["bpe-dropout", "byt5"]


def test_stopwords_and_punctuation_dropped() -> None:
    assert terms("The vocabulary is (joint).", EN, split_compounds=True) == ["vocabulary", "joint"]


def test_no_stopwords_keeps_everything() -> None:
    assert terms("the a", STOPWORDS["none"], split_compounds=True) == ["the", "a"]


def test_non_latin_scripts_are_terms() -> None:
    assert terms("Тексты 文本", EN, split_compounds=True) == ["тексты", "文本"]


def test_chunk_text_header_flag() -> None:
    c = make_chunk("p:a", 0, "body", header="T > S")
    assert chunk_text(c, with_header=True) == "T > S\nbody"
    assert chunk_text(c, with_header=False) == "body"
