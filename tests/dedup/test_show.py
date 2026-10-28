from rag.dedup.schemas import Work
from rag.dedup.show import find_work


def test_a_work_is_found_by_id_or_by_any_member() -> None:
    w = Work(
        work_id="work:acl:x",
        canonical="acl:x",
        canonical_rule="r",
        members=["acl:x", "arxiv:x"],
        aliases=["arxiv:x"],
        edges=[],
        rejected=[],
    )
    assert find_work([w], "work:acl:x") is w
    assert find_work([w], "arxiv:x") is w
    assert find_work([w], "nope") is None
