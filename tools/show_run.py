# show_run.py   python show_run.py              recent runs
#               python show_run.py RUN_ID        timeline of one run
#               python show_run.py RUN_ID --full  with full prompts, outputs, tool args and results
import json
import sys
from pathlib import Path

RUNS = Path("logs/runs")
META = ("seq", "ts", "t_ms", "run_id", "kind", "name", "phase")


def load(path):
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def totals(ev):
    mc = [e for e in ev if e["kind"] == "model_call"]
    tc = [e for e in ev if e["kind"] == "tool_call"]
    tok = lambda k: sum((e.get("tokens") or {}).get(k) or 0 for e in mc)
    return len(mc), len(tc), sum(e.get("usd") or 0 for e in mc), tok("prompt"), tok("output")


def list_runs(n=15):
    for p in sorted(RUNS.glob("*.jsonl"))[-n:]:
        ev = load(p)
        m, t, usd, _, _ = totals(ev)
        route = next((e["name"] for e in ev if e["kind"] == "router"), "-")
        final = next((e["name"] for e in reversed(ev) if e["kind"] == "final"), "no final event")
        print(f"{p.stem:<24} route={route:<13} model_calls={m} tool_calls={t} ${usd:.4f}  {final}")


def show(run_id, full=False):
    ev = load(RUNS / f"{run_id}.jsonl")
    m, t, usd, tin, tout = totals(ev)
    print(f"run {run_id}: {m} model calls, {t} tool calls, tokens in/out {tin}/{tout}, ${usd:.4f}, {ev[-1]['t_ms']} ms\n")
    for e in ev:
        head = f"#{e['seq']:<3} +{e['t_ms']:>6}ms  {e['kind']:<10} {e['name']:<26} "
        if e["kind"] == "model_call":
            tk = e.get("tokens") or {}
            extra = (f"[{e.get('phase')}] {e.get('duration_ms') or 0:.0f}ms  in {tk.get('prompt')} out {tk.get('output')}"
                     f"  ${e.get('usd') or 0:.4f}" + (f"  ERROR {e['error']}" if e.get("error") else ""))
        elif e["kind"] == "tool_call":
            extra = f"[{e.get('initiator')}] {e.get('status')} {e.get('duration_ms') or 0:.0f}ms  {json.dumps(e.get('args'))[:90]}"
        else:
            extra = json.dumps({k: v for k, v in e.items() if k not in META})[:160]
        print(head + extra)
        if full:
            payload = {k: v for k, v in e.items() if k in ("input", "output", "args", "result", "problems")}
            if payload:
                print("      " + json.dumps(payload, indent=2, ensure_ascii=False).replace("\n", "\n      "))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show(args[0], "--full" in sys.argv) if args else list_runs()