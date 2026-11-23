# Wedding Planning Assistant: RAG, MCP and guardrails

A retrieval-augmented assistant over r/AusWeddingPlanning (about 2,300 threads), built as an engineering
exercise in measured RAG, tool-calling agents and guardrails. It answers cost and planning questions with
citations, and builds a budget for requests that need one.

**What's in it**

- Resumable ingestion from a Reddit archive, thread reconstruction and a DuckDB store
- Retrieval experiments, each measured on a hand-labelled golden set: fixed and comment-aware chunking,
  dense, BM25, hybrid (reciprocal rank fusion) and cross-encoder reranking
- A rule-based router: budget requests go to the agent, everything else to plain RAG
- An MCP server (stdio) exposing retrieval and a deterministic budget calculator; the agent discovers its tools
  from the server's advertised schema
- A bounded tool-calling budget agent (Gemini function calling, at most 3 model calls)
- Guardrails: spend, step and model-call caps with a persistent ledger, input and retrieved-text screening,
  a canary leak scan, citation and total validation, a calibrated "I don't know" gate
- Observability: every model call and tool call in a run is written to a trace file
- Evaluation: retrieval metrics, an LLM-as-judge (claim-level faithfulness) with validation, and a red-team harness
- Offline tests with stubbed models and tools

## Architecture

```
question
   |
   v
rule router ----- rag / offer_budget --> input screen -> hybrid retrieval -> chunk screen -> score gate
   |                                      -> generate (schema, citations) -> validate -> leak scan -> answer
   |
   +--- budget ------------------------> input screen
                                           -> call 1: model chooses tool calls ----> MCP server (stdio)
                                                                                      search_threads
                                           -> code runs them (caps, duplicates,       get_thread
                                              chunk screen on results)                calculate_budget
                                           -> call 2: model writes the budget (schema)
                                           -> code checks (citations, stated total, calculator)
                                           -> optional call 3: repair or extra research
                                           -> leak scan -> result

Every model call goes through the guard (cost, step and call caps, ledger) and is traced.
```

## Setup

Requires Python 3.11 or later.

```powershell
pip install -e ".[dev]"
```

If a dependency has no wheel for your Python version, install the rest and register the package with
`pip install -e . --no-deps`.

1. **API key.** Copy `config/config.local.example.yaml` to `config/config.local.yaml` and add your Gemini key
   (this file is git-ignored). The `GEMINI_API_KEY` environment variable also works.
2. **Models and prices.** In `config/config.yaml`, add each model you use under `guard.models` with its price per
   million tokens, and set `judge.model`. An unpriced model is refused (the guard fails closed).
3. **Profile.** `config/config.yaml` defines retrieval profiles. Set `active:` to the one you want as the default
   (`bge_small_comment_notitle` is the one the experiments settled on); `--profile NAME` overrides it per command.

Run every command from the project root. In PowerShell, escape dollar signs inside quoted text:
`"Is `$2,000 enough for a photographer?"`.

## Data

The corpus is not in this repository. It came from a community Reddit archive (Arctic Shift) and is used for
personal, non-commercial work; keep it private. To rebuild it:

```powershell
python -m wedding_rag.ingestion.arctic_ingest smoke                    # check the API and field names
python -m wedding_rag.ingestion.arctic_ingest volume                   # row counts per year
python -m wedding_rag.ingestion.arctic_ingest crawl --kind posts       # resumable; --max-requests, --min-interval
python -m wedding_rag.ingestion.arctic_ingest crawl --kind comments
python -m wedding_rag.ingestion.arctic_ingest build                    # raw files -> DuckDB, prints stats
python -m wedding_rag.ingestion.export_threads                         # one record per thread -> data/threads.jsonl
python -m wedding_rag.rag.rag_baseline index --profile bge_small_comment_notitle   # build the vector index
```

Usernames are salted and hashed at ingest. Deleted and bot content is dropped.

## Commands

### Ask questions

```powershell
python -m wedding_rag.app.ask "how much does a celebrant cost"
python -m wedding_rag.app.ask "We have 30k for 80 guests in Melbourne, can you split it up? Food and photography matter most"
python tools/try_query.py                                  # interactive: print retrieved chunks, no model call
```

### Inspect a run

Every run writes `logs/runs/<run_id>.jsonl` (ignored by git: it contains prompts and chunk text).

```powershell
python tools/show_run.py                                   # recent runs
python tools/show_run.py RUN_ID                            # timeline of model and tool calls
python tools/show_run.py RUN_ID --full                     # full prompts, outputs, tool arguments and results
```

### MCP server

```powershell
python tools/show_tools.py                                 # the tool contract the server advertises
python tools/mcp_client_demo.py                            # raw SDK client calling each tool
python -m wedding_rag.mcp_server.server                    # the server itself (stdio: it waits for a client)
```

### Golden set

```powershell
python -c "from wedding_rag.evaluation.golden import validate_golden; validate_golden()"
python -c "from wedding_rag.evaluation.golden import find; find('celebrant')"    # which threads mention a term
python tools/make_golden.py --bucket medium --n 8          # write questions from sampled threads (--pool to review extras)
python tools/pool_review.py                                # review extra relevant threads, widen gold labels
python tools/inspect_corpus.py                             # corpus statistics
```

### Retrieval evaluation

```powershell
python -m wedding_rag.evaluation.run_log "hybrid notitle" --profile bge_small_comment_notitle --mode hybrid
python -m wedding_rag.evaluation.run_log "dense only" --profile bge_small_comment_notitle --mode dense
python -m wedding_rag.evaluation.run_log --show            # recent runs; compare only rows with the same golden hash
python tools/eval_only.py --profile bge_small_comment_notitle    # per-question output, no logging
python tools/compare_runs.py a.txt b.txt                   # per-question better/worse between two saved outputs
python tools/all_miss.py a.txt b.txt c.txt                 # questions every variant misses
```

Modes: `dense`, `bm25`, `hybrid`, `rerank`. Index profiles: `bge_small_fixed`, `bge_small_comment`,
`bge_small_comment_notitle`, `minilm_fixed`, `bge_small_fixed_c200` (build each index before using it).

### "I don't know" gate

```powershell
python tools/scores.py --profile bge_small_comment_notitle           # score distributions, answerable vs unanswerable
python tools/calibrate_gate.py --profile bge_small_comment_notitle   # threshold sweep
```

The threshold is `DENSE_MIN_SCORE` in `src/wedding_rag/generation/generate.py`. It is only valid for the model,
chunker and profile it was calibrated on.

### Generation evaluation (LLM-as-judge)

```powershell
python -m wedding_rag.evaluation.eval_generation --profile bge_small_comment_notitle --limit 5     # smoke test
python -m wedding_rag.evaluation.eval_generation --profile bge_small_comment_notitle               # all questions
python -m wedding_rag.evaluation.eval_generation --profile bge_small_comment_notitle --split dev   # dev questions only
python -m wedding_rag.evaluation.eval_generation --profile bge_small_comment_notitle --ids q006    # chosen questions
python -m wedding_rag.evaluation.eval_generation --profile bge_small_comment_notitle --no-judge    # generation only
python -m wedding_rag.evaluation.judge_validation --profile bge_small_comment_notitle --n 10       # does the judge catch damage?
```

Results are reported for all answerable questions, for questions where retrieval found the gold thread, and
for those where it did not. Summaries go to `logs/eval_results.jsonl`; per-question detail goes to
`logs/eval/` (git-ignored).

### Guard and setup checks

```powershell
python tools/guard_smoke.py                                # one real call through the guard
python tools/check_setup.py                                # do all expected paths resolve?
python tools/screen_corpus.py --profile bge_small_comment_notitle   # how many real chunks the screen would drop
```

### Tests

```powershell
pytest -q                                                  # all offline tests (no API key, no cost)
pytest tests/test_guard.py tests/test_router.py -q         # by file
```

Test files: guard, router, defences, budget agent, budget maths, generate, judge, red-team checks, thread
reconstruction and export.

## Red-team harness

The harness attacks the system with a hand-written case file (`golden/redteam_v1.jsonl`, 28 cases) and checks the
results automatically. Indirect-injection cases use the real model with fake tools or a fake retriever that
return poisoned text, so the real corpus is never modified. A canary string is planted in the system prompts;
any output containing it counts as a leak. Each case runs twice (three times if high severity) and passes only if
every run passes.

Case groups: direct injection, indirect injection (poisoned retrieved text), tool abuse, cost, budget traps,
off-topic, output manipulation, and router misrouting.

```powershell
# everything
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle
# only some
python redteam.py --profile bge_small_comment_notitle --defences on --only RT-D01,RT-D02,RT-D03,RT-D11,RT-D13,RT-I01,RT-I02,RT-I03,RT-I04,RT-B07,RT-B09
# defences on or off (off = baseline; the score gate is also disabled so it cannot stand in for the model)
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --defences on
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --defences off

# one path
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --path router    # no model calls
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --path rag
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --path budget

# the injection subset (direct, indirect and budget-agent injection): 11 cases
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --only RT-D01,RT-D02,RT-D03,RT-D11,RT-D13,RT-I01,RT-I02,RT-I03,RT-I04,RT-B07,RT-B09

# smaller subsets
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --only RT-D01,RT-D02,RT-D03,RT-D11,RT-D13   # direct injection
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --only RT-I01,RT-I02,RT-I03,RT-I04,RT-B07,RT-B09   # indirect injection
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --only RT-B01,RT-B02   # tool abuse and caps
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --only RT-B03,RT-B04,RT-B05,RT-B06   # budget traps
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --only RT-D05,RT-D06,RT-D07   # off-topic and harmful
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --only RT-D04,RT-D10   # output manipulation

# number of runs per case (default 2; high severity gets one extra)
python -m wedding_rag.evaluation.redteam --profile bge_small_comment_notitle --runs 8 --defences off --only RT-I01
```

Reading the results:

```powershell
python tools/layers.py                       # which layer stopped each case: input screen, chunk screen, leak scan, gate, model alone
python tools/layers.py logs/redteam/FILE.json
python tools/verify_redteam.py               # did the attack text and the canary actually reach the model?
python tools/show_run.py RUN_ID --full       # inspect any failing run
```

Run summaries are saved in `logs/redteam/`. To tell whether a defence helps, compare `--defences off` with
`--defences on`; a case that passes with defences off shows the model resisted on its own.

## Results

Fill these in from your own runs. Compare only results that share a golden-set hash.

| Area | Result |
|---|---|
| Retrieval (hybrid, comment-aware, no title prefix): recall@5, hit@5, MRR | [fill in] |
| "I don't know" gate: threshold, answerable kept, unanswerable refused | [fill in] |
| Judge: claim-level supported rate, fully supported answers (all answerable / retrieved / not retrieved) | [fill in] |
| Judge validation: damaged answers flagged, clean answers flagged | [fill in] |
| Red team, defences off: failed runs / total | [fill in] |
| Red team, defences on: failed runs / total | [fill in] |
| Cost: RAG answer, budget request | about $0.0003 and $0.004 in my runs on a flash-lite model |

## Project layout

```
config/          config.yaml (committed), config.local.yaml (secret, ignored)
golden/          golden.jsonl (questions, gold threads, reference answers), redteam_v1.jsonl
logs/            experiment and evaluation summaries; runs/ and eval/ are ignored
data/            ignored: corpus, indexes, DuckDB, spend ledger
src/wedding_rag/
  ingestion/     archive crawl, thread reconstruction, export
  rag/           chunking, embeddings, FAISS, BM25, hybrid retrieval
  guardrails/    budget (spend and step caps, ledger), defences (screens, canary)
  llm/           Gemini adapter
  observability/ per-run trace
  generation/    RAG answer with citations and validation
  agents/        budget agent and budget maths
  mcp_server/    FastMCP server
  app/           router and the ask entry point
  evaluation/    golden set, retrieval log, judge, generation eval, red team
tools/           scripts you run by hand
tests/           offline tests
```

## Design notes

- **Transport.** The MCP server uses stdio, so each budget request starts a server process (and loads the
  embedding model). A long-lived HTTP service would remove that cost.
- **Bounded agent.** Tool use is limited to the first of three model calls; the later calls have no tools, so the
  agent cannot loop. Search, category, thread and call caps are enforced in code.
- **Defences are layers.** Pattern-based screens catch known phrasings cheaply; the canary leak scan, citation
  validation and prompt rules back them up. A paraphrased attack can get past the screens.
- **Links and contact details are not removed from answers, and the corpus is not PII-masked.** This is a
  deliberate choice for this data set, since vendor links and names are the useful content.

## Limitations

- The corpus is a single Australian subreddit, collected from an archive, so coverage and prices reflect that
  community and its dates. Old prices carry a caveat but are not corrected.
- The golden set is small and hand-labelled; metrics at this size carry wide uncertainty, and labels were widened
  during development, so only results with the same golden hash are comparable.
- The judge is a model. It was checked against deliberately damaged answers, but its scores are estimates.
- The red-team set is hand-written and mostly in English; rule-based screens will miss paraphrases, other
  languages and obfuscation.
- The budget agent's output quality varies between runs (category choices, how much of the total it fills);
  improvements are planned: a fixed category list, code-computed amounts and a price-grounding check.
- Guard reservations are tracked within one process only.

## What is not in the repository

The corpus, indexes, spend ledger, run traces and evaluation detail files (they contain Reddit text and prompts),
and any API key.