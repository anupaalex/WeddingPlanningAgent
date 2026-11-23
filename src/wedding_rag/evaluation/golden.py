# golden.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel

from wedding_rag.paths import DATA_DIR, GOLDEN_DIR

GOLDEN_PATH = GOLDEN_DIR / "golden.jsonl"
THREADS_PATH = DATA_DIR / "threads.jsonl"


class GoldenQ(BaseModel):
    id: str                                   # stable, e.g. "q001"; never reuse
    question: str
    type: Literal["factoid", "synthesis", "long_thread", "unanswerable", "red_team"]
    gold_thread_ids: list[str] = []
    gold_comment_ids: list[str] = []
    reference_answer: str | None = None       # None for unanswerable
    source: Literal["manual", "llm_draft", "failure_case"] = "manual"
    verified: bool = True
    notes: str = ""

    @property
    def split(self) -> str:                   # deterministic: adding questions never moves others
        h = int(hashlib.sha256(self.id.encode()).hexdigest()[:8], 16)
        return "test" if h % 10 >= 7 else "dev"


def load_golden(split: str | None = None) -> list[GoldenQ]:
    qs = [GoldenQ.model_validate_json(l) for l in GOLDEN_PATH.open(encoding="utf-8") if l.strip()]
    return [q for q in qs if split is None or q.split == split]


def load_corpus() -> dict[str, dict]:
    return {r["thread_id"]: r for r in map(json.loads, THREADS_PATH.open(encoding="utf-8"))}


def validate_golden() -> None:
    corpus = load_corpus()
    ids = set()
    for q in load_golden():
        assert q.id not in ids, f"duplicate id {q.id}"
        ids.add(q.id)
        if q.type == "unanswerable":
            assert not q.gold_thread_ids, f"{q.id}: unanswerable must have no gold threads"
        else:
            assert q.gold_thread_ids, f"{q.id}: needs gold_thread_ids"
        for tid in q.gold_thread_ids:
            assert tid in corpus, f"{q.id}: thread {tid} not in corpus"
        for cid in q.gold_comment_ids:
            assert f"[c:{cid}]" in "".join(corpus[t]["text"] for t in q.gold_thread_ids), \
                f"{q.id}: comment {cid} not found in its gold threads"
    print(f"{len(ids)} questions valid")


def find(term: str, limit: int = 15) -> None:
    """Authoring helper: which threads mention a term? Use it to find gold ids,
    and to confirm an 'unanswerable' question really is absent."""
    n = 0
    for tid, r in load_corpus().items():
        if term.lower() in r["text"].lower():
            print(tid, r["year"], r["title"][:80])
            n += 1
            if n >= limit:
                break
    print(f"{n} shown")

import statistics
import time

def evaluate(retrieve, k: int = 5, split=None, verbose: bool = True, overfetch: int = 10) -> dict:
    recalls, rrs, lat, detail = [], [], [], []          # ← add `detail`
    from collections import defaultdict
    by_type = defaultdict(list)            # before the loop
    for q in load_golden(split):
        if q.type == "unanswerable":
            continue
        t0 = time.perf_counter()
        results = retrieve(q.question, k * overfetch)
        lat.append((time.perf_counter() - t0) * 1000)
        ranked: list[str] = []
        for r in results:
            if r["thread_id"] not in ranked:
                ranked.append(r["thread_id"])
        ranked = ranked[:k]
        gold = set(q.gold_thread_ids)
        recall = len(gold & set(ranked)) / len(gold)
        rank = next((i + 1 for i, t in enumerate(ranked) if t in gold), None)
        recalls.append(recall)
        by_type[q.type].append(recall)   
        rrs.append(1 / rank if rank else 0.0)
        detail.append({"id": q.id, "type": q.type, "recall": recall, "first_hit_rank": rank,   # ← add
                       "gold": sorted(gold), "retrieved": ranked})
        if verbose:
            print(f"{q.id} [{q.type}] recall@{k}={recall:.2f} first_hit_rank={rank}")
    n = len(recalls)
    m = {"n": n,
         "recall": sum(recalls) / n if n else None,
         "mrr": sum(rrs) / n if n else None,
         "p50_ms": statistics.median(lat) if lat else None,
         "per_question": detail}                         # ← add
    if verbose:
        print({k_: v for k_, v in m.items() if k_ != "per_question"},
              "(smoke test only: n < 30)" if n < 30 else "")
        for t, v in by_type.items():
            print(f"  {t:<12} n={len(v):>2}  recall={sum(v)/len(v):.2f}")
    return m