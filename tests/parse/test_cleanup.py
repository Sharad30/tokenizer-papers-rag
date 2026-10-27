from collections import Counter

import pytest

from rag.parse.cleanup import cut_bibliography, is_entry_piece, non_heading_kind
from rag.parse.config import CleanConfig
from rag.parse.schemas import Paragraph, Sentence

CLEAN = CleanConfig(min_entry_sentences=2, min_entry_share=0.3, max_fragment_words=15)


def para(*sentences: str) -> Paragraph:
    return Paragraph(sentences=[Sentence(text=s) for s in sentences])


def texts(paragraphs: list[Paragraph]) -> list[list[str]]:
    return [[s.text for s in p.sentences] for p in paragraphs]


@pytest.mark.parametrize(
    "sentence",
    [
        "2023.",
        "Nan Zhang, Yusen Zhang, et al. 2023.",
        "Preprint, arXiv:2306.16842.",
        "arXiv preprint arXiv:1909.10649.",
        "In Proceedings of the 61st Annual Meeting of the ACL, pages 5184-5207, Toronto.",
        "Association for Computational Linguistics.",
        "Stephen Tratz and Eduard Hovy. 2010. A taxonomy of noun compounds.",
    ],
)
def test_entry_pieces(sentence: str) -> None:
    assert is_entry_piece(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "We begin by formalizing validity, following Phan et al. (2024).",
        "We train on the Tulu-3 SFT mixture (Lambert et al., 2024).",
        "The novel 1984 was written by George Orwell and published in 1949.",
        "The dataset was published by the Association for Computational Linguistics in 2021.",
    ],
)
def test_prose_citing_a_paper_is_not_an_entry_piece(sentence: str) -> None:
    assert not is_entry_piece(sentence)


def test_a_mixed_paragraph_keeps_its_prose() -> None:
    # Shape of arxiv:2507.23248v2's Limitations: prose, then entries.
    counts: Counter[str] = Counter()
    kept = cut_bibliography(
        [
            para(
                "Our analysis is aggregate.",
                "We do not measure grapheme splits.",
                "Vilém Zouhar, Clara Meister, and Ryan Cotterell.",
                "2023.",
                "Tokenization and the noiseless channel.",
                "Preprint, arXiv:2306.16842.",
            )
        ],
        CLEAN,
        counts,
    )
    assert texts(kept) == [["Our analysis is aggregate.", "We do not measure grapheme splits."]]
    assert counts == {"bibliography_sentences_dropped": 4}


def test_an_entry_split_by_a_page_break_goes_whole() -> None:
    # acl:2020.emnlp-main.203: an entry's end, then the venue line.
    counts: Counter[str] = Counter()
    kept = cut_bibliography(
        [
            para("We conclude here."),
            para(
                "with subword units.",
                "In Proceedings of the 54th Annual Meeting of the ACL, pages 1715-1725, Berlin.",
                "Association for Computational Linguistics.",
            ),
        ],
        CLEAN,
        counts,
    )
    assert texts(kept) == [["We conclude here."]]
    assert counts == {"bibliography_sentences_dropped": 3, "bibliography_paragraphs_dropped": 1}


def test_one_stray_piece_never_cuts() -> None:
    paragraph = para("Results are in Table 2.", "See pages 3-4, and the appendix.", "Done.")
    assert cut_bibliography([paragraph], CLEAN, Counter()) == [paragraph]


def test_a_tail_that_is_mostly_prose_is_kept() -> None:
    # Two pieces, but the tail from the first one is 2 of 8: under the 0.3 share.
    paragraph = para(
        "2020.",
        "Models were evaluated with BLEU.",
        "We used beam search.",
        "Scores are in the appendix, pages 10-12.",
        "We report the mean.",
        "Variance is small.",
        "Seeds are fixed.",
        "Code is public.",
    )
    assert cut_bibliography([paragraph], CLEAN, Counter()) == [paragraph]


def test_formula_paragraphs_are_left_alone() -> None:
    formula = Paragraph(sentences=[Sentence(text="2023.")], formula_id="formula_0")
    assert cut_bibliography([formula], CLEAN, Counter()) == [formula]


def test_an_author_list_alone_is_not_cut() -> None:
    paragraph = para("We thank Ann Lee, Bo Chen, and Cy Diaz.", "They helped a lot.")
    assert cut_bibliography([paragraph], CLEAN, Counter()) == [paragraph]


def test_the_cut_walks_back_over_authors_and_title() -> None:
    # Authors, title, then the venue line that is the first flagged piece.
    kept = cut_bibliography(
        [
            para(
                "We stop here.",
                "Jun Zhao, Zhihao Zhang, and Xuanjing Huang.",
                "Llama beyond english: An empirical study.",
                "arXiv preprint arXiv:2401.01055,",
                "2024.",
            )
        ],
        CLEAN,
        Counter(),
    )
    assert texts(kept) == [["We stop here."]]


@pytest.mark.parametrize(
    ("heading", "kind"),
    [
        ("Figure 4: Accuracy", "figure_label"),
        ("Table 5", "figure_label"),
        ("Fig. 2. A Motivating Example", "figure_label"),
        ("AE(w) = W E(w) SE(w)", "equation"),
        ("ϕ(w) ∝ I(w).", "equation"),
        ("out ⩾ b(i).", "equation"),
        ("A R T I C L E I N P R E S S", "letter_spaced"),
        (
            "MEGABYTE: Predicting million-byte sequences. In Proceedings of NeurIPS",
            "reference_title",
        ),
    ],
)
def test_non_headings(heading: str, kind: str) -> None:
    assert non_heading_kind(heading) == kind


@pytest.mark.parametrize(
    "heading",
    [
        "BPE",
        "NER",
        "GPT-4",
        "B.1 NER",
        "FR+ES→EN transformer",
        "A.4. Fixed l vs l → ∞",
        "Tables",
        "Information Theoretic Lower Bound (L = 1)",
        "F.2 |V| = 60000, T = 1.0",
        "Best Tokenization = Best Translation?",
    ],
)
def test_short_real_headings_stay(heading: str) -> None:
    assert non_heading_kind(heading) is None
