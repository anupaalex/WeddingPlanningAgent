# run_trace.py
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

RUNS_DIR = Path("logs/runs")
MAX_FIELD_CHARS = 20000


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _clip(v, limit: int = MAX_FIELD_CHARS):
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit] + f"…[{len(v) - limit} more chars]"
    if isinstance(v, dict):
        return {k: _clip(x, limit) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clip(x, limit) for x in v]
    return v


def describe_response(resp) -> dict:
    """Best-effort summary of a model response. Never raises."""
    out: dict = {}
    try:
        calls = getattr(resp, "function_calls", None)
        if calls:
            out["function_calls"] = [{"name": c.name, "args": dict(c.args or {})} for c in calls]
    except Exception:
        pass
    try:
        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            out["parsed"] = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
    except Exception:
        pass
    try:
        text = getattr(resp, "text", None)
        if text:
            out["text"] = text
    except Exception:
        pass
    return out


class RunTrace:
    """One JSON line per event, written immediately, so a crashed run still leaves a trace."""

    def __init__(self, run_id: str, runs_dir: Path = RUNS_DIR):
        self.run_id = run_id
        self.path = runs_dir / f"{run_id}.jsonl"
        self.phase: str | None = None
        self._seq = 0
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        runs_dir.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, name: str, **fields) -> None:
        with self._lock:
            self._seq += 1
            rec = {"seq": self._seq, "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                   "t_ms": int((time.perf_counter() - self._t0) * 1000), "run_id": self.run_id,
                   "kind": kind, "name": name, "phase": self.phase, **_clip(fields)}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def model_call(self, model, text, *, response=None, usage=None, usd=None,
                   duration_ms=None, error=None, blocked=False) -> None:
        f = {"model": model, "input": text, "duration_ms": duration_ms, "usd": usd,
             "error": error, "blocked": blocked}
        if usage is not None:
            f["tokens"] = {"prompt": getattr(usage, "prompt_token_count", None),
                           "output": getattr(usage, "candidates_token_count", None),
                           "thinking": getattr(usage, "thoughts_token_count", None)}
        if response is not None:
            f["output"] = describe_response(response)
        self.log("model_call", model, **f)