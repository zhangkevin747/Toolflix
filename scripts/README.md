# The data pipeline

How the marketplace (`data/pool/`) and the tasks (`data/tasks/`) are built. You only
run this to (re)build the data — once it exists, training reads it directly (see
`src/toolbandit/`).

Each script prints a JSON summary and exits non-zero if a check fails. Run from the
repo root, e.g. `python scripts/build_tool_pool.py`.

## Phase A — build the marketplace (`src/tool_pool/`)

| # | Script | Does | Writes |
|---|--------|------|--------|
| 1 | `build_tool_pool.py` | Pick base tools, make reworded + broken copies, add distractors | `data/pool/listings.jsonl` (+ base_tools, variant_candidates) |
| 2 | `review_base_tools.py` | Draft known-good arguments for each base tool | `base_tool_fixtures.jsonl`, `base_tool_review.csv` |
| 3 | `validate_base_fixtures.py` | Offline check: do the fixtures fit each schema? | `base_fixture_validation.jsonl` |
| 4 | `live_validate_base_tools.py` | **Actually call** each base tool through MCP; reject failures | `live_base_validation.jsonl` |
| 5 | `live_validate_adapters.py` | **Actually call** each reworded copy through its adapter | `live_adapter_validation.jsonl` |
| 6 | `finalize_tool_pool.py` | Gather results, write the "ready" manifest | `manifest.json`, `smoke_tests.jsonl` |

Steps 4–5 need the MCP-Bench servers (`external/mcp-bench/`) and any API keys in
`.env`. Steps 1–3 and 6 are offline. Step 1 is deterministic — same seed, identical pool.

## Phase B — generate tasks (`src/task_generation/`)

| # | Script | Does | Writes |
|---|--------|------|--------|
| 1 | `generate_tasks.py` | LLM writes a fuzzy one-step task per tool, grounded in its real output | `data/tasks/tasks.jsonl` |
| 2 | `validate_tasks.py` | Re-check grounding and tool references | `data/tasks/validation.json` |
| 3 | `filter_ready_tasks.py` | Drop the tasks that failed | `data/tasks/tasks_ready.jsonl` |

Step 1 needs `OPENAI_API_KEY`.

## Why it's split this way

The slow, expensive truth is in steps 4–5: a tool can look perfect and still fail when
actually called (dead API, redirect, or an error returned as a normal response). So the
pipeline cleanly separates *proposing* metadata (cheap, offline, rebuildable) from
*proving* tools run (live MCP calls). Each step is a rerunnable checkpoint that writes a
file the next step reads.

## Shared helpers

All scripts import `tool_pool/io.py` for paths and JSONL/`.env` helpers, so each script
is just its own logic. The build/validate logic lives in `src/tool_pool/` and
`src/task_generation/` (each module has a docstring explaining its job).
