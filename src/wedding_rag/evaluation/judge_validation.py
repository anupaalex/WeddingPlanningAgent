# judge_validation.py   python judge_validation.py --profile bge_small_comment_notitle [--n 10]
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from wedding_rag.evaluation import judge as J
from wedding_rag.evaluation.eval_generation import OUT_DIR, new_run
from wedding_rag.llm.gemini_adapter import make_gemini
from wedding_rag.guardrails.budget import BudgetGuard, GuardError, load_guard_config

_PRICE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?!\d|\s?[kK])")


def corrupt_price(answer: str) -> str | None:
    m = _PRICE.search(answer)
    if not m:
        return None
    return answer[:m.start()] + f"${int(float(m.group(1).replace(',', '')) * 1.6):,}" + answer[m.end():]


def add_claim(answer: str) -> str:
    return answer.rstrip() + " Most photographers also include a free engagement shoot and unlimited retouching."


def main(a) -> None:
    latest = sorted(OUT_DIR.glob("*.jsonl"))[-1]
    recs = [json.loads(l) for l in latest.open(encoding="utf-8")]
    clean = [r for r in recs if r.get("judge") and r.get("cited_chunks") and r["judge"]["claims"]
             and all(c["verdict"] == "supported" for c in r["judge"]["claims"])][: a.n]
    if len(clean) < 3:
        raise SystemExit("not enough fully supported answers in the latest eval run to corrupt")
    cases = []
    for i, r in enumerate(clean):
        cases.append(("clean_rerun", r["question"], r["answer"], r["cited_chunks"]))
        p = corrupt_price(r["answer"])
        if p:
            cases.append(("price_changed", r["question"], p, r["cited_chunks"]))
        cases.append(("claim_added", r["question"], add_claim(r["answer"]), r["cited_chunks"]))
        cases.append(("wrong_evidence", r["question"], r["answer"], clean[(i + 1) % len(clean)]["cited_chunks"]))

    count, call = make_gemini()
    guard = BudgetGuard(load_guard_config(), token_counter=count)
    stats = defaultdict(lambda: {"n": 0, "judge": 0, "price_check": 0, "either": 0})
    for kind, question, answer, chunks in cases:
        try:
            out = J.judge_answer(guard, new_run(guard, "judge"), call, question, answer, chunks, None)
        except (GuardError, ValueError):
            continue
        flagged_j = any(c.verdict != "supported" for c in out.claims)
        flagged_p = bool(J.ungrounded_figures(answer, [c["text"] for c in chunks]))
        s = stats[kind]
        s["n"] += 1
        s["judge"] += flagged_j
        s["price_check"] += flagged_p
        s["either"] += flagged_j or flagged_p
    print(f"\n{'kind':<16}{'n':>3}  flagged by judge / price check / either   (clean_rerun should be LOW, the rest HIGH)")
    for kind, s in stats.items():
        n = s["n"]
        print(f"{kind:<16}{n:>3}  {s['judge'] / n:>6.2f} {s['price_check'] / n:>11.2f} {s['either'] / n:>11.2f}")
    with Path("logs/judge_validation.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": latest.name,
                            "judge": J.judge_model(), "stats": stats}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    main(ap.parse_args())