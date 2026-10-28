import numpy as np
import pytest

from rag.dedup.minhash import (
    MinHasher,
    candidate_probability,
    estimate_jaccard,
    exact_jaccard,
    lsh_candidates,
    shingle_hashes,
    shingles,
)


def test_shingles_are_lowercased_word_ngrams() -> None:
    assert shingles("Byte Pair  encoding, merges pairs.", 3) == {
        "byte pair encoding",
        "pair encoding merges",
        "encoding merges pairs",
    }
    assert shingles("BPE", 3) == {"bpe"}
    assert shingles("", 3) == set()


def sig(hasher: MinHasher, items: set[str]) -> np.ndarray:
    return hasher.signature(shingle_hashes(items))


def test_identical_sets_agree_everywhere_and_disjoint_sets_almost_nowhere() -> None:
    hasher = MinHasher(num_perm=128, seed=0)
    a = {f"w{i}" for i in range(500)}
    b = {f"v{i}" for i in range(500)}
    assert estimate_jaccard(sig(hasher, a), sig(hasher, a)) == 1.0
    assert estimate_jaccard(sig(hasher, a), sig(hasher, b)) < 0.05


@pytest.mark.parametrize("overlap", [100, 300, 450])
def test_estimate_is_close_to_exact_jaccard(overlap: int) -> None:
    hasher = MinHasher(num_perm=256, seed=1)
    a = {f"w{i}" for i in range(500)}
    b = {f"w{i}" for i in range(500 - overlap, 1000 - overlap)}
    exact = exact_jaccard(a, b)
    # Standard error sqrt(J(1-J)/256) <= 0.031; allow about 3 of them.
    assert abs(estimate_jaccard(sig(hasher, a), sig(hasher, b)) - exact) < 0.1


def test_candidate_probability_is_an_s_curve() -> None:
    assert candidate_probability(0.0, 32, 4) == 0.0
    assert candidate_probability(1.0, 32, 4) == 1.0
    # Hand-computed: 1 - (1 - 0.5^4)^32 = 1 - (15/16)^32 ≈ 0.8732
    assert candidate_probability(0.5, 32, 4) == pytest.approx(0.8732, abs=1e-4)
    assert candidate_probability(0.1, 32, 4) < 0.01


def test_lsh_pairs_share_a_band() -> None:
    signatures = {
        "a": np.array([1, 2, 3, 4], dtype=np.uint64),
        "b": np.array([1, 2, 9, 9], dtype=np.uint64),  # same first band as a
        "c": np.array([7, 7, 3, 5], dtype=np.uint64),  # no whole band in common
    }
    assert lsh_candidates(signatures, bands=2, rows=2) == {("a", "b")}
