"""Relevance filter: keyword rules first, embedding similarity for the ambiguous rest.

The rules implement the D-005 boundary: tokenization must be a primary
contribution or main variable, not just a tool the paper uses. A topic term
in the *title* is strong evidence of that; a topic term only in the abstract
is common in papers that merely use a tokenizer, so those go to a second
judge. Every decision carries a reason code so rejects can be audited.
"""

import re
from collections.abc import Callable, Sequence

import numpy as np

from rag.collect.config import RelevanceConfig
from rag.collect.schemas import (
    CandidateRecord,
    DecidedBy,
    Decision,
    Reason,
    RelevanceDecision,
)


def _compile(terms: Sequence[str]) -> re.Pattern[str]:
    # Word boundaries so "bpe" doesn't match inside "bpeX", and case-insensitive.
    # Longest first, so "byte-pair encoding" wins over a shorter overlapping term.
    alternatives = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


class KeywordRules:
    def __init__(self, config: RelevanceConfig) -> None:
        self._title_terms = _compile(config.title_accept_terms)
        # Every title-accept term is also a topic term.
        self._topic_terms = _compile([*config.title_accept_terms, *config.topic_terms])
        self._exclude_terms = _compile(config.title_exclude_terms)

    def has_topic_term(self, title: str, abstract: str) -> bool:
        """Cheap pre-screen used by the ACL scan."""
        return bool(self._topic_terms.search(title) or self._topic_terms.search(abstract))

    def decide(self, title: str, abstract: str) -> RelevanceDecision:
        topic_matches = _found(self._topic_terms, f"{title}\n{abstract}")
        if not topic_matches:
            return _keyword(Decision.REJECT, Reason.NO_TOPIC_TERM, [])
        # Checked before acceptance: "A Visual Tokenizer for ..." has a title
        # term but is outside the boundary (D-005 covers LLM text tokenizers).
        excluded = _found(self._exclude_terms, title)
        if excluded:
            return _keyword(Decision.REJECT, Reason.OFF_TOPIC_TERM, excluded)
        title_matches = _found(self._title_terms, title)
        if title_matches:
            return _keyword(Decision.ACCEPT, Reason.TITLE_TERM, title_matches)
        return _keyword(Decision.AMBIGUOUS, Reason.ABSTRACT_TERM_ONLY, topic_matches)


Encoder = Callable[[list[str]], np.ndarray]


class EmbeddingJudge:
    """Cosine similarity of title + abstract to the written topic description.

    `encode` must return L2-normalised rows, so a dot product is the cosine.
    It is injected so tests can use a fake encoder instead of a model download.
    """

    def __init__(self, encode: Encoder, topic_description: str, threshold: float) -> None:
        self._encode = encode
        self._topic = encode([topic_description])[0]
        self.threshold = threshold

    def scores(self, records: Sequence[CandidateRecord]) -> list[float]:
        if not records:
            return []
        vectors = self._encode([judge_text(r) for r in records])
        return [float(v) for v in vectors @ self._topic]

    def resolve(self, ambiguous: RelevanceDecision, score: float) -> RelevanceDecision:
        accepted = score >= self.threshold
        return RelevanceDecision(
            decision=Decision.ACCEPT if accepted else Decision.REJECT,
            decided_by=DecidedBy.EMBEDDING,
            reason=Reason.EMBEDDING_ABOVE if accepted else Reason.EMBEDDING_BELOW,
            matched_terms=ambiguous.matched_terms,
            embedding_score=score,
        )


def sentence_transformer_encoder(model_name: str, batch_size: int) -> Encoder:
    """bge-small via sentence-transformers, on CPU.

    No query instruction: the model card reserves it for short retrieval
    queries, and this is passage-to-passage similarity (D-008).
    """
    from sentence_transformers import SentenceTransformer  # slow import; only needed here

    model = SentenceTransformer(model_name, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

    return encode


def judge_text(record: CandidateRecord) -> str:
    return f"{record.title}. {record.abstract}" if record.abstract else record.title


def _found(pattern: re.Pattern[str], text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in pattern.finditer(text)})


def _keyword(decision: Decision, reason: Reason, terms: list[str]) -> RelevanceDecision:
    return RelevanceDecision(
        decision=decision, decided_by=DecidedBy.KEYWORD, reason=reason, matched_terms=terms
    )
