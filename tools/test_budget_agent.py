# test_budget_agent.py
import asyncio
import json
from types import SimpleNamespace

import pytest

from wedding_rag.agents import budget_agent as ba
from wedding_rag.generation import generate as gen
from wedding_rag.agents.budget_math import compute_budget
from wedding_rag.guardrails.budget import BudgetGuard, GuardConfig, ModelPrice
from wedding_rag.observability.run_trace import RunTrace


class FakeTool:
    def __init__(self, fn):
        self.fn = fn

    async def ainvoke(self, args):
        return self.fn(args)


TOOLS = {
    "search_threads": FakeTool(lambda a: [{"chunk_id": "c_" + a["query"].split()[0], "thread_id": "t1",
                                           "year": 2025, "score": 0.5, "text": "typical price discussion"}]),
    "get_thread": FakeTool(lambda a: {"thread_id": a["thread_id"], "year": 2025, "text": "thread text"}),
        "calculate_budget": FakeTool(lambda a: compute_budget(a["total"], {x["category"]: x["amount"] for x in a["allocations"]}))
}
SCHEMA = {
    "$defs": {"BudgetLine": {"properties": {"category": {"description": "c", "title": "Category", "type": "string"},
                                            "amount": {"description": "a", "title": "Amount", "type": "number"}},
                             "required": ["category", "amount"], "title": "BudgetLine", "type": "object"}},
    "properties": {"total": {"description": "t", "title": "Total", "type": "number"},
                   "allocations": {"description": "lines", "items": {"$ref": "#/$defs/BudgetLine"},
                                   "title": "Allocations", "type": "array"},
                   "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Note"}},
    "required": ["total", "allocations"], "title": "calculate_budgetArguments", "type": "object"}


class FakeSession:
    def __init__(self, tools):
        self._tools = tools

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)


def mcp_tool(name, description="server says this", schema=None):
    return SimpleNamespace(name=name, description=description,
                           inputSchema=schema or {"type": "object", "properties": {}})


def test_clean_schema_inlines_refs_and_drops_unsupported_keywords():
    cleaned = ba.clean_schema(SCHEMA)
    text = json.dumps(cleaned)
    assert not any(k in text for k in ('"$ref"', '"$defs"', '"title"', '"default"', '"anyOf"'))
    assert cleaned["properties"]["allocations"]["items"]["properties"]["amount"]["type"] == "number"
    assert cleaned["properties"]["note"]["type"] == "string"


def test_declarations_come_from_the_server():
    session = FakeSession([mcp_tool("search_threads", "SEARCH DESC"), mcp_tool("get_thread"),
                           mcp_tool("calculate_budget", schema=SCHEMA), mcp_tool("delete_everything")])
    decls = asyncio.run(ba.discover_declarations(session))
    assert {d.name for d in decls} == ba.ALLOWED_TOOLS               # the unlisted tool is never exposed
    assert next(d for d in decls if d.name == "search_threads").description == "SEARCH DESC"


def test_missing_tool_is_an_error():
    with pytest.raises(RuntimeError):
        asyncio.run(ba.discover_declarations(FakeSession([mcp_tool("search_threads")])))

REQUEST = "We have $30,000 for 80 guests in Melbourne, split it up for venue and celebrant"


def fc(name, **args):
    return SimpleNamespace(name=name, args=args)


def resp(function_calls=None, parsed=None, text=None):
    return SimpleNamespace(function_calls=function_calls, parsed=parsed, text=text,
                           usage_metadata=SimpleNamespace(prompt_token_count=500, candidates_token_count=200,
                                                          thoughts_token_count=0))


def make_call(*responses):
    q = list(responses)

    def call(model, text, max_out, system=None, temperature=0.0, response_schema=None, tools=None):
        return q.pop(0)
    return call


def item(cat, amount, basis="evidence"):
    return ba.Item(category=cat, amount=amount, low=None, typical=None, high=None, basis=basis,
                   cited_chunk_ids=[f"c_{cat}"] if basis == "evidence" else [], note="x")


def result(items, total=30000):
    return ba.BudgetResult(mode="fit", total_budget=total, items=items, caveats=[], need_more=[])


CALL1 = resp(function_calls=[fc("search_threads", query="venue hire Melbourne", category="venue"),
                             fc("search_threads", query="celebrant fee Melbourne", category="celebrant")])
GOOD = resp(parsed=result([item("venue", 20000), item("celebrant", 1000), item("contingency", 9000, "assumption")]))
OVER = resp(parsed=result([item("venue", 25000), item("celebrant", 1000), item("contingency", 9000, "assumption")]))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "MODEL", "test-model")

    def make():
        cfg = GuardConfig(max_usd_per_run=10, max_usd_per_day=100, max_usd_total=100, max_steps_per_run=10,
                          max_delegation_depth=3, max_output_tokens_per_call=2000, repeat_limit=3,
                          ledger_path=tmp_path / "l.db", models={"test-model": ModelPrice(input=1.0, output=10.0)})
        g = BudgetGuard(cfg)
        run = g.new_run("r1")
        run.trace = RunTrace("r1", tmp_path / "runs")
        return g, run, ba.ToolRunner(TOOLS, run.trace)
    return make


def go(setup, request, *responses):
    g, run, runner = setup()
    out = asyncio.run(ba._pipeline(request, g, run, make_call(*responses), runner))
    return out, run, runner


def test_happy_path(setup):
    out, _, _ = go(setup, REQUEST, CALL1, GOOD)
    assert out["status"] == "ok" and out["model_calls"] == 2 and out["searches"] == 2


def test_over_budget_is_repaired_in_the_third_call(setup):
    out, _, _ = go(setup, REQUEST, CALL1, OVER, GOOD)
    assert out["status"] == "ok" and out["model_calls"] == 3


def test_still_over_budget_after_repair_is_flagged(setup):
    out, _, _ = go(setup, REQUEST, CALL1, OVER, OVER)
    assert out["status"] == "unverified" and out["model_calls"] == 3


def test_no_tool_calls_means_ask_the_user(setup):
    out, _, _ = go(setup, "help with my wedding", resp(text="What would you like budgeted?"))
    assert out["status"] == "needs_input" and out["model_calls"] == 1


def test_invented_total_is_caught(setup):
    bad = resp(parsed=result([item("venue", 40000), item("celebrant", 1000), item("contingency", 9000, "assumption")],
                             total=50000))
    out, _, _ = go(setup, REQUEST, CALL1, bad, bad)
    assert out["status"] == "unverified" and any("not stated" in p for p in out["problems"])


def test_tool_caps_and_duplicates(setup):
    _, _, runner = setup()

    async def go_():
        for i in range(30):
            await runner.run_one("search_threads", {"query": f"q{i} thing", "category": f"cat{i % 12}"})
        await runner.run_one("search_threads", {"query": "q0 thing", "category": "cat0"})   # duplicate
        await runner.run_one("delete_everything", {})                                       # not an allowed tool
    asyncio.run(go_())
    assert runner.n_search <= ba.MAX_SEARCHES and len(runner.categories) <= ba.MAX_CATEGORIES
    assert all(v <= ba.MAX_PER_CATEGORY for v in runner._per_cat.values())


def test_trace_records_every_model_and_tool_call(setup):
    _, run, _ = go(setup, REQUEST, CALL1, GOOD)
    events = [json.loads(l) for l in run.trace.path.open(encoding="utf-8")]
    kinds = [e["kind"] for e in events]
    assert kinds.count("model_call") == 2
    assert kinds.count("tool_call") == 3          # two searches by the model, one calculator check by the code
    phases = [e["phase"] for e in events if e["kind"] == "model_call"]
    assert phases == ["call1_decide", "call2_synthesise"]
    assert all(e["input"] for e in events if e["kind"] == "model_call")

def test_model_calculator_call_with_invented_amounts_is_dropped(setup):
    _, run, runner = setup()
    runner.request = "We have $30,000 for 80 guests"
    out = asyncio.run(runner.run_one("calculate_budget",
                                     {"total": 30000, "allocations": [{"category": "venue", "amount": 9000}]}))
    assert out["status"] == "dropped_invented_amounts"


def test_user_supplied_amounts_pass_the_check(setup):
    _, run, runner = setup()
    runner.request = "Check my budget: venue 12k, catering 9k, total 30k"
    out = asyncio.run(runner.run_one("calculate_budget", {"total": 30000, "allocations": [
        {"category": "venue", "amount": 12000}, {"category": "catering", "amount": 9000}]}))
    assert out["ok"]


def test_missing_category_is_inferred(setup):
    _, run, runner = setup()
    asyncio.run(runner.run_one("search_threads", {"query": "wedding catering cost per head Melbourne"}))
    assert runner.categories == ["catering and drinks"]