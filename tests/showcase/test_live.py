from rag.retrieve.schemas import Hit
from rag.showcase.live import badge, preview, run_query
from tests.retrieve.conftest import make_chunk


class Fixed:
    """Returns the same hits for any query."""

    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, k=None, mode=None):
        return self.hits[:k]


def hit(rank: int, record_id: str, n: int, text: str = "bpe merges") -> Hit:
    return Hit(chunk=make_chunk(record_id, n, text, header="T > S"), rank=rank, score=1 / rank)


def test_hits_are_marked_as_repeats_of_a_paper_above() -> None:
    hits = [hit(1, "acl:a", 0), hit(2, "acl:a", 1), hit(3, "arxiv:a", 0), hit(4, "acl:b", 0, "x")]
    work_of = {"acl:a": "work:a", "arxiv:a": "work:a", "acl:b": "work:b"}
    judge = {"acl:a": "g", "arxiv:a": "g"}
    view = run_query(Fixed(hits), "q", 4, "hybrid", judge, work_of, term="BPE")
    assert [h.same_paper_as for h in view.hits] == [None, 1, 1, None]
    assert [h.other_record for h in view.hits] == [False, False, True, False]
    assert badge(view.hits[2]).startswith("duplicate: another record")
    assert badge(view.hits[1]).startswith("duplicate: another chunk")
    assert [h.has_term for h in view.hits] == [True, True, True, False]
    assert view.dup == 0.5 and view.copy_dup == 0.25 and view.papers == 2


def test_preview_cuts_on_a_word() -> None:
    assert preview("one two three four", 9) == "one two …"
    assert preview("short", 9) == "short"
