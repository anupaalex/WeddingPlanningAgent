# router.py
from __future__ import annotations

import re

MONEY = re.compile(r"\$\s?\d|\b\d[\d,]*(?:\.\d+)?\s?(?:k|grand|dollars|aud)\b", re.I)
BUDGET_WORD = re.compile(r"\bbudget\w*", re.I)
CREATE_STRONG = re.compile(
    r"\b(?:how (?:should|do|can) (?:we|i) (?:spend|split|allocate|divide)"
    r"|(?:make|build|create|work out|put together|draw up) (?:me |us )?(?:a |an |our |my )?(?:\w+ )?budget)\b", re.I)
SPLITTY = re.compile(r"\b(?:split|allocat\w*|divide|spread|break ?down)\b", re.I)
CHECK = re.compile(
    r"\b(?:check|review|sanity[- ]check|assess)\s+(?:my|our|this|the)\s+budget\b"
    r"|\b(?:is|does|will)\s+(?:my|our|this|the)\s+budget\b|\bbudget\s+(?:work|realistic)", re.I)
PLAN_VERB = re.compile(r"\b(?:plan|planning|build|create|make|prepare|put together|work out|draw up)\b", re.I)
AFFORD = re.compile(r"\b(?:afford|enough|within budget|stretch|pull off|can (?:we|i) (?:do|have|host|throw))\b", re.I)


def route(message: str) -> str:
    """'budget' -> budget agent · 'offer_budget' -> RAG answer plus an offer · 'rag' -> plain RAG.
    Unsure means RAG: a wrong agent route costs several times more than a wrong RAG route."""
    m = (message or "").strip()
    if not m or len(m) > 2000:
        return "rag"
    has_money, has_budget = bool(MONEY.search(m)), bool(BUDGET_WORD.search(m))
    if CREATE_STRONG.search(m) or CHECK.search(m):
        return "budget"
    if SPLITTY.search(m) and (has_money or has_budget):
        return "budget"
    if has_money and has_budget and PLAN_VERB.search(m):
        return "budget"
    if has_budget or (has_money and AFFORD.search(m)):
        return "offer_budget"
    return "rag"