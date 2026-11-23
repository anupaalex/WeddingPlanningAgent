# guard.py
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from wedding_rag.paths import CONFIG_DIR, ROOT

class GuardError(Exception): ...
class BudgetExceeded(GuardError): ...
class StepLimitExceeded(GuardError): ...
class LoopDetected(GuardError): ...
class UnknownModel(GuardError): ...


class ModelPrice(BaseModel):
    input: float    # USD per 1M tokens
    output: float

    @field_validator("input", "output")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price must be > 0; fill in the real price (zero would disable the cost cap)")
        return v


class GuardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_usd_per_run: float
    max_usd_per_day: float
    max_usd_total: float
    max_steps_per_run: int
    max_delegation_depth: int
    max_output_tokens_per_call: int
    thinking_reserve_tokens: int = 0
    input_safety_margin: float = 1.1
    repeat_limit: int = 3
    ledger_path: Path = Path("data/ledger.db")
    models: dict[str, ModelPrice]
    @field_validator("ledger_path")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return v if v.is_absolute() else ROOT / v


def load_guard_config(path: Path = CONFIG_DIR / "config.yaml") -> GuardConfig:
    with open(path, encoding="utf-8") as f:
        return GuardConfig(**yaml.safe_load(f)["guard"])


def conservative_counter(model: str, text: str) -> int:
    return len(text) // 3 + 1        # over-estimates; production passes the real count_tokens


@dataclass
class Run:
    run_id: str
    usd: float = 0.0
    reserved: float = 0.0
    steps: int = 0
    seen: dict[str, int] = field(default_factory=dict)
    trace: Any = None  

@dataclass
class Reservation:
    run: Run
    model: str
    worst: float


class BudgetGuard:
    def __init__(self, cfg: GuardConfig, token_counter: Callable[[str, str], int] | None = None):
        self.cfg = cfg
        self.count = token_counter or conservative_counter
        self._lock = threading.Lock()
        self._reserved = 0.0          # in-flight reservations across all runs in this process
        cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cfg.ledger_path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS ledger(
            ts REAL, run_id TEXT, model TEXT, usd REAL,
            in_tok INTEGER, out_tok INTEGER, think_tok INTEGER)""")
        self.db.commit()

    def new_run(self, run_id: str | None = None) -> Run:
        return Run(run_id or uuid.uuid4().hex[:8])

    def _cost(self, model: str, in_tok: int, out_tok: int) -> float:
        p = self.cfg.models[model]
        return (in_tok * p.input + out_tok * p.output) / 1e6

    def _sum(self, since: float | None = None) -> float:
        if since is None:
            row = self.db.execute("SELECT COALESCE(SUM(usd), 0) FROM ledger").fetchone()
        else:
            row = self.db.execute("SELECT COALESCE(SUM(usd), 0) FROM ledger WHERE ts > ?", (since,)).fetchone()
        return row[0]

    def reserve(self, run: Run, model: str, text: str, max_out: int, depth: int = 0) -> Reservation:
        c = self.cfg
        if model not in c.models:
            raise UnknownModel(f"{model!r} has no price in the guard config")
        if max_out > c.max_output_tokens_per_call:
            raise BudgetExceeded(f"max_out {max_out} exceeds the per-call cap {c.max_output_tokens_per_call}")
        in_tok = int(self.count(model, text) * c.input_safety_margin) + 1   # may be a network call: outside the lock
        worst = self._cost(model, in_tok, max_out + c.thinking_reserve_tokens)
        h = hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()
        with self._lock:
            if depth > c.max_delegation_depth:
                raise StepLimitExceeded(f"delegation depth {depth} > {c.max_delegation_depth}")
            if run.steps >= c.max_steps_per_run:
                raise StepLimitExceeded(f"step cap {c.max_steps_per_run} reached")
            if run.seen.get(h, 0) + 1 > c.repeat_limit:
                raise LoopDetected(f"same call repeated more than {c.repeat_limit} times")
            if run.usd + run.reserved + worst > c.max_usd_per_run:
                raise BudgetExceeded("per-run cap (booked + reserved + worst case)")
            if self._sum(time.time() - 86400) + self._reserved + worst > c.max_usd_per_day:
                raise BudgetExceeded("per-day cap (rolling 24h)")
            if self._sum() + self._reserved + worst > c.max_usd_total:
                raise BudgetExceeded("total project cap")
            run.steps += 1
            run.seen[h] = run.seen.get(h, 0) + 1
            run.reserved += worst
            self._reserved += worst
        return Reservation(run, model, worst)

    def commit(self, res: Reservation, in_tok: int, out_tok: int, think_tok: int = 0) -> float:
        usd = self._cost(res.model, in_tok, out_tok + think_tok)    # thinking is billed as output
        with self._lock:
            res.run.reserved -= res.worst
            self._reserved -= res.worst
            res.run.usd += usd
            self.db.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?)",
                            (time.time(), res.run.run_id, res.model, usd, in_tok, out_tok, think_tok))
            self.db.commit()
        return usd

    def release(self, res: Reservation) -> None:
        """Call failed: free the reservation, book nothing."""
        with self._lock:
            res.run.reserved -= res.worst
            self._reserved -= res.worst

    def summary(self) -> str:
        with self._lock:
            return (f"last 24h ${self._sum(time.time() - 86400):.4f} / ${self.cfg.max_usd_per_day:.2f}   "
                    f"total ${self._sum():.4f} / ${self.cfg.max_usd_total:.2f}")

                    # ← new: a RunTrace, or None


def gated_call(guard: BudgetGuard, run: Run, model: str, text: str,
               call_fn: Callable[[int], Any], max_out: int = 1000, depth: int = 0):
    t0 = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t0) * 1000, 1)
    try:
        res = guard.reserve(run, model, text, max_out, depth)
    except GuardError as e:
        if run.trace:
            run.trace.model_call(model, text, error=f"{type(e).__name__}: {e}", blocked=True, duration_ms=ms())
        raise
    spend0 = run.usd
    try:
        resp = call_fn(max_out)
    except Exception as e:
        guard.release(res)
        if run.trace:
            run.trace.model_call(model, text, error=f"{type(e).__name__}: {e}", duration_ms=ms())
        raise
    u = resp.usage_metadata
    guard.commit(res, u.prompt_token_count or 0, u.candidates_token_count or 0,
                 getattr(u, "thoughts_token_count", 0) or 0)
    if run.trace:
        run.trace.model_call(model, text, response=resp, usage=u, usd=run.usd - spend0, duration_ms=ms())
    return resp