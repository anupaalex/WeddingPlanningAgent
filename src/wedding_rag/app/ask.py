# ask.py
from __future__ import annotations

import asyncio
import json
import sys

from wedding_rag.generation import generate as gen
from wedding_rag.agents.budget_agent import run_budget_agent
from wedding_rag.llm.gemini_adapter import make_gemini
from wedding_rag.guardrails.budget import BudgetGuard, load_guard_config
from wedding_rag.rag.hybrid import make_retriever
from wedding_rag.app.router import route
from wedding_rag.observability.run_trace import RunTrace, new_run_id


async def handle(message: str, guard: BudgetGuard, call, retriever) -> dict:
    run = guard.new_run(new_run_id())
    run.trace = RunTrace(run.run_id)
    decision = route(message)
    run.trace.log("router", decision, message=message)
    if decision == "budget":
        return await run_budget_agent(message, guard, run, call)
    run.trace.phase = "rag"
    result = await asyncio.to_thread(gen.answer_question, message, guard, run, retriever, call)
    result["run_id"] = run.run_id
    if decision == "offer_budget":
        result["offer"] = "Want me to build a budget for this?"
    run.trace.log("final", result.get("status", "?"), cost_usd=round(run.usd, 5))
    return result


if __name__ == "__main__":
    count, call = make_gemini()
    guard = BudgetGuard(load_guard_config(), token_counter=count)
    out = asyncio.run(handle(" ".join(sys.argv[1:]), guard, call, make_retriever("hybrid")))
    print(json.dumps(out, indent=2, default=str))
    print(guard.summary())
    print(f"\ntrace: python show_run.py {out.get('run_id')}")