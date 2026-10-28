from rag.retrieve.schemas import Hit
from rag.retrieve.show import render_hits
from tests.retrieve.conftest import make_chunk


def test_rows_show_both_retrievers_and_missing_ranks() -> None:
    c1 = make_chunk("p:a", 0, "merge operations", header="BPE Paper > Method")
    c2 = make_chunk("p:b", 3, "byte level", header="Bytes Paper > Intro")
    hits = [
        Hit(
            chunk=c1,
            rank=1,
            score=0.0325,
            ranks={"bm25": 1, "dense": 2},
            scores={"bm25": 7.5, "dense": 0.81},
        ),
        Hit(
            chunk=c2, rank=2, score=0.0161, ranks={"dense": 1}, scores={"bm25": 0.0, "dense": 0.85}
        ),
    ]
    out = render_hits("merges", "hybrid", hits)
    assert "mode=hybrid  k=2" in out
    assert "   1   7.50     2  0.810  p:a#0000" in out
    assert "   –            1  0.850  p:b#0003" in out
    assert "BPE Paper | " in out


def test_hidden_copies_are_counted_and_listed_briefly() -> None:
    from rag.retrieve.schemas import Hit
    from rag.retrieve.show import render_hits
    from tests.retrieve.conftest import make_chunk

    hit = Hit(
        chunk=make_chunk("acl:x", 0, "text"),
        rank=1,
        score=0.5,
        work_id="work:acl:x",
        hidden=["acl:x#0001", "acl:x#0002", "acl:x#0003", "arxiv:x#0000"],
    )
    out = render_hits("q", "hybrid", [hit])
    assert "+4 more from work:acl:x: acl:x#0001, acl:x#0002, acl:x#0003, …" in out
