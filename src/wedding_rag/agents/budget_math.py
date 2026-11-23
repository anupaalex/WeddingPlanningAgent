# budget_math.py
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

MAX_CATEGORIES = 30


def _cents(x: float) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), ROUND_HALF_UP)


def compute_budget(total: float, allocations: dict[str, float]) -> dict:
    if not isinstance(allocations, dict) or not allocations:
        raise ValueError("allocations must be a non-empty object of category -> amount")
    if len(allocations) > MAX_CATEGORIES:
        raise ValueError(f"at most {MAX_CATEGORIES} categories")
    for v in (total, *allocations.values()):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("amounts must be finite numbers")
    if total <= 0:
        raise ValueError("total must be greater than 0")
    if any(v < 0 for v in allocations.values()):
        raise ValueError("allocations must not be negative")

    items = {name.strip()[:60]: _cents(v) for name, v in allocations.items()}
    if len(items) != len(allocations):
        raise ValueError("category names must be unique")

    t = _cents(total)
    allocated = sum(items.values(), Decimal("0"))
    remainder = t - allocated
    warnings = []
    if remainder < 0:
        warnings.append(f"allocations exceed the total by {-remainder}")
    elif remainder > t * Decimal("0.2"):
        warnings.append("more than 20% of the budget is unallocated")
    big = [n for n, v in items.items() if v > t * Decimal("0.5")]
    if big:
        warnings.append(f"single category above 50% of the total: {big}")

    return {
        "total": float(t), "allocated": float(allocated), "remainder": float(remainder),
        "over_budget": remainder < 0,
        "percent_of_total": {n: float((v / t * 100).quantize(Decimal("0.1"))) for n, v in items.items()},
        "warnings": warnings,
    }