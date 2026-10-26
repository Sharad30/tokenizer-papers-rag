import pytest

from rag.retrieve.fusion import rrf


def test_plan_toy_example_k60_rewards_agreement() -> None:
    # Item 7 is 1st in one list only; item 3 is 3rd in both.
    fused = rrf({"bm25": [7, 1, 3], "dense": [2, 4, 3]}, k=60)
    assert fused[0][0] == 3
    assert fused[0][1] == pytest.approx(2 / 63)


def test_k1_lets_a_single_first_place_tie_agreement() -> None:
    fused = dict(rrf({"bm25": [7, 1, 3], "dense": [2, 4, 3]}, k=1))
    assert fused[7] == pytest.approx(1 / 2)
    assert fused[3] == pytest.approx(2 / 4)


def test_ties_broken_by_best_rank_then_id() -> None:
    # 5 and 9 each get 1/61 + 1/62 and a best rank of 1.
    fused = rrf({"a": [5, 9], "b": [9, 5]}, k=60)
    assert [item for item, _ in fused] == [5, 9]  # equal score, equal best rank: lower id
    # 4 and 8 each get 1/61 from one list.
    fused = rrf({"a": [4], "b": [8]}, k=60)
    assert [item for item, _ in fused] == [4, 8]


def test_item_in_one_list_only_still_counts() -> None:
    assert dict(rrf({"a": [1], "b": []}, k=60)) == {1: pytest.approx(1 / 61)}
