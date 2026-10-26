from rag.chat.citations import count_uncited, resolve

VALID = {"C1", "C2", "C3"}


def test_renumbers_in_order_of_first_use() -> None:
    out = resolve("BPE merges pairs [C3]. It starts from characters [C1][C3].", VALID)
    assert out.text == "BPE merges pairs [1]. It starts from characters [2][1]."
    assert out.used == ["C3", "C1"]


def test_comma_lists_and_lowercase() -> None:
    out = resolve("Both methods regularize segmentation [c2, C1; C2].", VALID)
    assert out.text == "Both methods regularize segmentation [1, 2]."
    assert out.used == ["C2", "C1"]


def test_invalid_labels_removed_and_reported() -> None:
    out = resolve("Unigram uses an EM step [C7]. Dropout skips merges [C2, C9].", VALID)
    assert out.text == "Unigram uses an EM step. Dropout skips merges [1]."
    assert out.invalid == ["C7", "C9"]
    assert out.used == ["C2"]


def test_uncited_sentences_counted() -> None:
    answer = "BPE merges frequent pairs [C1]. It was adapted from compression.\nYes."
    assert count_uncited(answer) == 1  # "Yes." is too short to count as a claim
    assert resolve(answer, VALID).uncited_sentences == 1


def test_text_without_citations_is_unchanged() -> None:
    out = resolve("NOT_SUPPORTED", VALID)
    assert out.text == "NOT_SUPPORTED" and out.used == [] and out.invalid == []
