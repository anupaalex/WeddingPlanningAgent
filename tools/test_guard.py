# test_guard.py
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from wedding_rag.guardrails.budget import (BudgetExceeded, BudgetGuard, GuardConfig, LoopDetected, ModelPrice,
                   StepLimitExceeded, UnknownModel, gated_call)

M = "test-model"


def make_cfg(tmp_path, **over):
    base = dict(max_usd_per_run=0.05, max_usd_per_day=100, max_usd_total=100, max_steps_per_run=50,
                max_delegation_depth=3, max_output_tokens_per_call=2000, repeat_limit=3,
                ledger_path=tmp_path / "ledger.db", models={M: ModelPrice(input=1.0, output=10.0)})
    base.update(over)
    return GuardConfig(**base)


def fake_call(max_out):      # each call books (100*1 + 1000*10)/1e6 = $0.0101
    return SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=100, candidates_token_count=1000, thoughts_token_count=0))


def test_per_run_cap_stops_the_run(tmp_path):
    g = BudgetGuard(make_cfg(tmp_path)); run = g.new_run()
    ok = 0
    with pytest.raises(BudgetExceeded):
        for i in range(10):
            gated_call(g, run, M, f"prompt {i}", fake_call); ok += 1
    assert ok == 4 and run.usd <= 0.05


def test_step_cap(tmp_path):
    g = BudgetGuard(make_cfg(tmp_path, max_steps_per_run=3, max_usd_per_run=10)); run = g.new_run()
    for i in range(3):
        gated_call(g, run, M, f"p{i}", fake_call)
    with pytest.raises(StepLimitExceeded):
        gated_call(g, run, M, "p3", fake_call)


def test_unknown_model_fails_closed(tmp_path):
    g = BudgetGuard(make_cfg(tmp_path))
    with pytest.raises(UnknownModel):
        g.reserve(g.new_run(), "nope", "hi", 100)


def test_loop_detection(tmp_path):
    g = BudgetGuard(make_cfg(tmp_path, max_usd_per_run=10)); run = g.new_run()
    for _ in range(3):
        gated_call(g, run, M, "same prompt", fake_call)
    with pytest.raises(LoopDetected):
        gated_call(g, run, M, "same prompt", fake_call)


def test_output_cap_per_call(tmp_path):
    g = BudgetGuard(make_cfg(tmp_path))
    with pytest.raises(BudgetExceeded):
        gated_call(g, g.new_run(), M, "hi", fake_call, max_out=5000)


def test_ledger_survives_restart(tmp_path):
    cfg = make_cfg(tmp_path, max_usd_per_day=0.015)
    gated_call(BudgetGuard(cfg), BudgetGuard(cfg).new_run(), M, "first", fake_call)   # books $0.0101
    g2 = BudgetGuard(cfg)                                                             # a "new process"
    with pytest.raises(BudgetExceeded):
        gated_call(g2, g2.new_run(), M, "second", fake_call)


def test_failed_call_releases_reservation(tmp_path):
    g = BudgetGuard(make_cfg(tmp_path)); run = g.new_run()

    def boom(max_out):
        raise RuntimeError("network")

    with pytest.raises(RuntimeError):
        gated_call(g, run, M, "hi", boom)
    assert run.reserved == pytest.approx(0, abs=1e-12) and run.usd == 0


def test_parallel_workers_cannot_overspend(tmp_path):
    g = BudgetGuard(make_cfg(tmp_path)); run = g.new_run()
    granted, lock = [], threading.Lock()

    def worker(i):
        try:
            g.reserve(run, M, f"t{i}", 1000)
            with lock:
                granted.append(i)
        except BudgetExceeded:
            pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(granted) == 4                       # 20 tried; exactly 4 fit in $0.05
    assert run.reserved <= 0.05


def test_zero_price_rejected():
    with pytest.raises(ValidationError):
        ModelPrice(input=0, output=1)