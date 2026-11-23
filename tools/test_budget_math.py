# test_budget_math.py
import pytest
from wedding_rag.agents.budget_math import compute_budget


def test_basic():
    r = compute_budget(30000, {"venue": 12000, "catering": 9000})
    assert r["allocated"] == 21000 and r["remainder"] == 9000 and not r["over_budget"]


def test_over_budget_warns():
    r = compute_budget(1000, {"a": 800, "b": 300})
    assert r["over_budget"] and r["remainder"] == -100 and r["warnings"]


def test_no_float_drift():
    assert compute_budget(100, {"a": 0.1, "b": 0.2})["allocated"] == 0.3


@pytest.mark.parametrize("total, alloc", [
    (0, {"a": 1}), (-5, {"a": 1}), (100, {}), (100, {"a": -1}),
    (float("nan"), {"a": 1}), (100, {"a": float("inf")}), (100, {"a": True}),
])
def test_rejects_bad_input(total, alloc):
    with pytest.raises(ValueError):
        compute_budget(total, alloc)