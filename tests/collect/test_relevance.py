import numpy as np
import pytest

from rag.collect.config import RelevanceConfig
from rag.collect.relevance import (
    DecidedBy,
    Decision,
    EmbeddingJudge,
    KeywordRules,
    Reason,
    judge_text,
)
from rag.collect.schemas import CandidateRecord, Source

CONFIG = RelevanceConfig(
    title_accept_terms=[
        "tokenizer",
        "tokenization",
        "bpe",
        "byte-pair encoding",
        "subword",
        "tokenizer-free",
    ],
    topic_terms=["fertility", "vocabulary size"],
    title_exclude_terms=["visual tokenizer", "token pruning"],
    topic_description="tokenization research",
    embedding_model="unused",
    embedding_batch_size=2,
    embedding_threshold=0.5,
)
rules = KeywordRules(CONFIG)


@pytest.mark.parametrize(
    ("title", "abstract", "decision", "reason"),
    [
        ("Tokenizer-Free Language Modelling", "", Decision.ACCEPT, Reason.TITLE_TERM),
        ("How BPE Merges Shape Models", "", Decision.ACCEPT, Reason.TITLE_TERM),
        ("Rethinking Byte-Pair Encoding", "", Decision.ACCEPT, Reason.TITLE_TERM),
        # Uses a tokenizer, doesn't study it: only the abstract mentions it.
        (
            "A Summariser for Turkish News",
            "We train with a BPE tokenizer.",
            Decision.AMBIGUOUS,
            Reason.ABSTRACT_TERM_ONLY,
        ),
        (
            "Translation Quality Across Languages",
            "Token fertility predicts quality.",
            Decision.AMBIGUOUS,
            Reason.ABSTRACT_TERM_ONLY,
        ),
        (
            "Graph Neural Networks for Molecules",
            "Message passing.",
            Decision.REJECT,
            Reason.NO_TOPIC_TERM,
        ),
        ("A Visual Tokenizer for Image Generation", "", Decision.REJECT, Reason.OFF_TOPIC_TERM),
        (
            "Token Pruning for Vision Transformers",
            "Fewer tokenization steps.",
            Decision.REJECT,
            Reason.OFF_TOPIC_TERM,
        ),
    ],
)
def test_keyword_rules(title: str, abstract: str, decision: Decision, reason: Reason) -> None:
    result = rules.decide(title, abstract)
    assert (result.decision, result.reason) == (decision, reason)
    assert result.decided_by is DecidedBy.KEYWORD


def test_matching_is_whole_word_and_case_insensitive() -> None:
    assert rules.decide("SUBWORD units", "").decision is Decision.ACCEPT
    # "bpe" inside another word is not a match.
    assert rules.decide("Abpex Systems", "").reason is Reason.NO_TOPIC_TERM
    # "tokenizers" is not "tokenizer": plural forms must be listed in config.
    assert rules.decide("Tokenizers Compared", "").reason is Reason.NO_TOPIC_TERM


def test_matched_terms_are_recorded() -> None:
    result = rules.decide("BPE and Subword Regularisation", "")
    assert result.matched_terms == ["bpe", "subword"]


def test_has_topic_term() -> None:
    assert rules.has_topic_term("Parsing", "vocabulary size matters")
    assert not rules.has_topic_term("Parsing", "syntax trees")


def fake_encoder(texts: list[str]) -> np.ndarray:
    """2-d vectors: [is about tokenization, is not]; rows L2-normalised."""
    rows = [[1.0, 0.0] if "token" in t.lower() else [0.0, 1.0] for t in texts]
    return np.array(rows)


def record(title: str, abstract: str | None) -> CandidateRecord:
    return CandidateRecord(
        record_id=f"s2:{title}", source=Source.S2, title=title, abstract=abstract
    )


def test_embedding_judge_scores_and_resolves() -> None:
    judge = EmbeddingJudge(fake_encoder, "tokenization research", threshold=0.5)
    on_topic, off_topic = record("A", "tokenizer study"), record("B", "phishing detection")
    scores = judge.scores([on_topic, off_topic])
    assert scores == [1.0, 0.0]

    ambiguous = rules.decide("A", "tokenizer study")
    accepted = judge.resolve(ambiguous, scores[0])
    rejected = judge.resolve(ambiguous, scores[1])
    assert (accepted.decision, accepted.reason) == (Decision.ACCEPT, Reason.EMBEDDING_ABOVE)
    assert (rejected.decision, rejected.reason) == (Decision.REJECT, Reason.EMBEDDING_BELOW)
    assert accepted.decided_by is DecidedBy.EMBEDDING
    assert accepted.embedding_score == 1.0
    assert accepted.matched_terms == ambiguous.matched_terms


def test_threshold_is_inclusive() -> None:
    judge = EmbeddingJudge(fake_encoder, "token", threshold=1.0)
    assert judge.resolve(rules.decide("x", "tokenizer"), 1.0).decision is Decision.ACCEPT


def test_scores_of_empty_list_skips_encoding() -> None:
    calls: list[list[str]] = []

    def counting(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return fake_encoder(texts)

    judge = EmbeddingJudge(counting, "token", threshold=0.5)
    assert judge.scores([]) == []
    assert len(calls) == 1  # only the topic description


def test_judge_text_falls_back_to_title() -> None:
    assert judge_text(record("Title", None)) == "Title"
    assert judge_text(record("Title", "Abstract.")) == "Title. Abstract."
