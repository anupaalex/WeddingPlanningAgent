# python tools/layers.py [logs/redteam/FILE.json]     which layer stopped each red-team case
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from wedding_rag.paths import LOGS_DIR

f = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((LOGS_DIR / "redteam").glob("*.json"))[-1]
summ = json.loads(f.read_text(encoding="utf-8"))
by_case = defaultdict(Counter)
for r in summ:
    ev = [json.loads(l) for l in (LOGS_DIR / "runs" / f"{r['run_id']}.jsonl").open(encoding="utf-8")]
    decisions = {e["name"] for e in ev if e["kind"] == "decision"}
    model_calls = sum(e["kind"] == "model_call" for e in ev)
    if "input_screen" in decisions:
        layer = "input_screen (no model call)"
    elif "leak_scan" in decisions:
        layer = "leak_scan"
    elif "chunk_screen" in decisions:
        layer = "chunk_screen, then model answered" if model_calls else "chunk_screen (no model call)"
    elif "gate" in decisions and model_calls == 0:
        layer = "score gate"
    else:
        layer = "model alone"
    by_case[r["case_id"]][layer] += 1
print(f"{f.name}")
for case, c in by_case.items():
    print(f"{case:<9} {dict(c)}")