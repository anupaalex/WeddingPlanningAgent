# run_log.py
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from wedding_rag.rag import rag_baseline as rb
from wedding_rag.evaluation.golden import GOLDEN_PATH, evaluate
from wedding_rag.settings import CFG
LOG_FILE = Path("logs/experiments.jsonl")

# near the top, after the other imports
import argparse
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--mode", default="dense")
ns, rest = ap.parse_known_args()
sys.argv = [sys.argv[0]] + rest
from wedding_rag.rag.hybrid import make_retriever

# in log_run: use it instead of rb.Retriever()

# and add this to the logged config:  "retriever": ns.mode
def golden_hash() -> str:
    return hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest()[:10]


def log_run(label: str, k: int = 5) -> dict:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    m = evaluate(make_retriever(ns.mode), k=k, split=None)   # split=None while the golden set is tiny
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "k": k,
        "n_questions": m["n"],
        "recall": m["recall"],
        "mrr": m["mrr"],
        "p50_ms": m["p50_ms"],
        "golden_hash": golden_hash(),   # runs are only comparable when this matches
        "config": CFG.model_dump(mode="json", exclude={"threads_path", "db_path", "index_root"}),
        "retriever": ns.mode
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("logged:", label)
    return record


def show(last: int = 10) -> None:
    if not LOG_FILE.exists():
        print("no runs yet")
        return
    runs = [json.loads(l) for l in LOG_FILE.open(encoding="utf-8") if l.strip()][-last:]
    fmt = lambda v, p=2: f"{v:.{p}f}" if v is not None else "-"
    print(f"{'label':<28}{'n':>3} {'recall':>7} {'mrr':>6} {'p50ms':>7}  golden")
    for r in runs:
        print(f"{r['label'][:27]:<28}{r['n_questions']:>3} {fmt(r['recall']):>7} "
              f"{fmt(r['mrr']):>6} {fmt(r['p50_ms'], 0):>7}  {r['golden_hash']}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--show"]:
        show()
    else:
        log_run(sys.argv[1] if len(sys.argv) > 1 else "unlabelled")