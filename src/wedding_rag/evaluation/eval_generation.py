# eval_generation.py   python eval_generation.py --profile bge_small_comment_notitle [--limit 20] [--no-judge]
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from wedding_rag.guardrails import defences as D
from wedding_rag.generation import generate as gen
from wedding_rag.evaluation import judge as J
from wedding_rag.settings import CFG
from wedding_rag.llm.gemini_adapter import make_gemini
from wedding_rag.evaluation.golden import load_golden
from wedding_rag.guardrails.budget import BudgetGuard, GuardError, load_guard_config
from wedding_rag.rag.hybrid import make_retriever
from wedding_rag.evaluation.run_log import golden_hash
from wedding_rag.observability.run_trace import RunTrace, new_run_id

OUT_DIR = Path("logs/eval")
SUMMARY = Path("logs/eval_results.jsonl")
REFUSED = ("gated", "no_context", "blocked_input", "blocked_output")


def new_run(guard, tag):
    run = guard.new_run(new_run_id())
    run.trace = RunTrace(run.run_id)
    run.trace.phase = tag
    return run


def evaluate_one(q, guard, call, retriever, use_judge) -> dict:
    run = new_run(guard, "rag")
    try:
        res = gen.answer_question(q.question, guard, run, retriever, call, return_context=True)
    except GuardError as e:
        res = {"status": "blocked", "reason": str(e)}
    ctx = res.pop("context", [])
    gold = set(q.gold_thread_ids)
    answered = res.get("status") == "ok" and res.get("answerable") is not False
    rec = {"id": q.id, "type": q.type, "notes": q.notes, "question": q.question, "gen_run": run.run_id,
           "status": res.get("status"), "answer": res.get("answer"), "citations": res.get("citations") or [],
           "retrieved": bool(gold & {c["thread_id"] for c in ctx}) if gold else None,
           "answered": answered, "refused": res.get("status") in REFUSED or res.get("answerable") is False,
           "gen_cost": round(run.usd, 6)}
    if q.type == "unanswerable" or not answered:
        return rec
    cited = [c for c in ctx if c["chunk_id"] in set(rec["citations"])]
    rec["cited_chunks"] = cited
    rec["ungrounded"] = J.ungrounded_figures(res["answer"], [c["text"] for c in cited])
    if use_judge:
        jrun = new_run(guard, "judge")
        rec["judge_run"] = jrun.run_id
        try:
            rec["judge"] = J.judge_answer(guard, jrun, call, q.question, res["answer"], cited or ctx,
                                          q.reference_answer).model_dump()
        except (GuardError, ValueError) as e:
            rec["judge_error"] = f"{type(e).__name__}: {e}"
        rec["judge_cost"] = round(jrun.usd, 6)
    return rec


def rate(num, den):
    return round(num / den, 3) if den else None


def group_metrics(recs: list[dict]) -> dict:
    ans = [r for r in recs if r["type"] != "unanswerable"]
    answered = [r for r in ans if r["answered"]]
    judged = [r for r in answered if r.get("judge")]
    claims = [c for r in judged for c in r["judge"]["claims"]]
    with_ref = [r for r in judged if r["judge"]["correctness"] != "no_reference"]
    return {
        "n": len(ans), "answered": rate(len(answered), len(ans)), "refused": rate(sum(r["refused"] for r in ans), len(ans)),
        "n_judged": len(judged), "n_claims": len(claims),
        "claim_supported": rate(sum(c["verdict"] == "supported" for c in claims), len(claims)),
        "claim_unsupported": rate(sum(c["verdict"] == "unsupported" for c in claims), len(claims)),
        "fully_supported_answers": rate(sum(all(c["verdict"] == "supported" for c in r["judge"]["claims"]) for r in judged), len(judged)),
        "no_unsupported_answers": rate(sum(not any(c["verdict"] == "unsupported" for c in r["judge"]["claims"]) for r in judged), len(judged)),
        "relevant": rate(sum(r["judge"]["relevance"] == "answers" for r in judged), len(judged)),
        "n_ref": len(with_ref),
        "agrees_with_reference": rate(sum(r["judge"]["correctness"] == "agrees" for r in with_ref), len(with_ref)),
        "prices_grounded": rate(sum(not r["ungrounded"] for r in answered), len(answered)),
    }


def main(a) -> None:
    count, call = make_gemini()
    guard = BudgetGuard(load_guard_config(), token_counter=count)
    retriever = make_retriever("hybrid")
    golden = load_golden()
    answerable = [q for q in golden if q.type != "unanswerable"]
    qs = (answerable[: a.limit] if a.limit else answerable) + [q for q in golden if q.type == "unanswerable"]
    records = []
    for q in qs:
        r = evaluate_one(q, guard, call, retriever, not a.no_judge)
        records.append(r)
        j = r.get("judge")
        verdicts = "".join({"supported": "+", "partial": "~", "unsupported": "-"}[c["verdict"]] for c in j["claims"]) if j else ""
        print(f"{r['id']:<5} {r['type']:<12} {str(r['status']):<13} retrieved={r['retrieved']}  {verdicts}")

    ans = [r for r in records if r["type"] != "unanswerable"]
    groups = {"all answerable": ans,
              "retrieved": [r for r in ans if r["retrieved"]],
              "not retrieved": [r for r in ans if r["retrieved"] is False],
              "factoid": [r for r in ans if r["type"] == "factoid"],
              "synthesis": [r for r in ans if r["type"] == "synthesis"]}
    metrics = {g: group_metrics(rs) for g, rs in groups.items()}
    un = [r for r in records if r["type"] == "unanswerable"]
    by_note = defaultdict(list)
    for r in un:
        by_note[r["notes"]].append(r["refused"])
    metrics["unanswerable"] = {"n": len(un), "refused": rate(sum(r["refused"] for r in un), len(un)),
                               "by_group": {k: f"{sum(v)}/{len(v)}" for k, v in by_note.items()}}
    gen_cost = sum(r["gen_cost"] for r in records)
    judge_cost = sum(r.get("judge_cost", 0) for r in records)
    summary = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "golden_hash": golden_hash(),
               "profile": CFG.profile, "generator": gen.MODEL, "judge": None if a.no_judge else J.judge_model(),
               "gate": gen.DENSE_MIN_SCORE, "defences": dict(D.ENABLED), "limit": a.limit,
               "metrics": metrics, "cost": {"generation": round(gen_cost, 4), "judge": round(judge_cost, 4)}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (OUT_DIR / f"{stamp}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                            encoding="utf-8")
    with SUMMARY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    for g, m in metrics.items():
        print(f"\n{g}: {json.dumps(m)}")
    print(f"\ncost: generation ${gen_cost:.4f}, judge ${judge_cost:.4f}   details: logs/eval/{stamp}.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="answerable questions to run (all unanswerable are always run)")
    ap.add_argument("--no-judge", action="store_true")
    main(ap.parse_args())