"""MinHash signatures and LSH banding, written out so the mechanism is visible.

Jaccard similarity of two shingle sets is |A ∩ B| / |A ∪ B|. Exact Jaccard
over every pair of ~1.8k papers is slow; MinHash estimates it from short
signatures:

- Apply a random hash function h to every shingle and keep the minimum. The
  chance that two sets have the same minimum is exactly their Jaccard
  similarity (the minimum of A ∪ B is equally likely to be any element, and
  the two minima agree only when that element is in A ∩ B).
- Repeat with `num_perm` independent hash functions: the fraction of positions
  where two signatures agree estimates Jaccard, with standard error about
  sqrt(J(1 - J) / num_perm).

LSH banding finds candidate pairs without comparing all of them: cut each
signature into b bands of r rows; two sets become candidates when any band is
identical. For Jaccard s that happens with probability 1 - (1 - s^r)^b, an
S-curve whose steep part sits near (1/b)^(1/r).
"""

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

# Hash functions h(x) = (a·x + b) mod P over shingle hashes x < P. A Mersenne
# prime below 2^32 keeps a·x + b under 2^63, so uint64 arithmetic never wraps.
PRIME = (1 << 31) - 1
WORD = re.compile(r"\w+")


def shingles(text: str, words: int) -> set[str]:
    """Word n-grams of lowercased text ("byte pair encoding merges", ...)."""
    tokens = WORD.findall(text.lower())
    if len(tokens) < words:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + words]) for i in range(len(tokens) - words + 1)}


def shingle_hashes(items: Iterable[str]) -> np.ndarray:
    """Stable 31-bit hashes (Python's hash() changes between runs)."""
    return np.array(
        [
            int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") % PRIME
            for s in items
        ],
        dtype=np.uint64,
    )


class MinHasher:
    def __init__(self, num_perm: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # a must be non-zero, or every shingle hashes to b.
        self.a = rng.integers(1, PRIME, size=(num_perm, 1), dtype=np.uint64)
        self.b = rng.integers(0, PRIME, size=(num_perm, 1), dtype=np.uint64)

    @property
    def num_perm(self) -> int:
        return len(self.a)

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        """Per hash function, the minimum hash over the set. Empty set: all PRIME."""
        if len(hashes) == 0:
            return np.full(self.num_perm, PRIME, dtype=np.uint64)
        return ((self.a * hashes[np.newaxis, :] + self.b) % PRIME).min(axis=1)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def exact_jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def candidate_probability(similarity: float, bands: int, rows: int) -> float:
    """Chance that a pair with this Jaccard shares at least one band."""
    return 1 - (1 - similarity**rows) ** bands


def lsh_candidates(
    signatures: Mapping[str, np.ndarray], bands: int, rows: int
) -> set[tuple[str, str]]:
    """Pairs (a < b) whose signatures are identical in at least one band."""
    pairs: set[tuple[str, str]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[str]] = defaultdict(list)
        for key, sig in signatures.items():
            buckets[sig[band * rows : (band + 1) * rows].tobytes()].append(key)
        for members in buckets.values():
            members.sort()
            pairs.update(
                (members[i], members[j])
                for i in range(len(members))
                for j in range(i + 1, len(members))
            )
    return pairs
