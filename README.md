# Toolflix Progress Log

Quick state of the project as of 2026-05-15.

## Goal

Toolflix is building a marketplace-style tool selection benchmark for ToolBandit: given a fuzzy user task and a marketplace query, retrieve candidate tools, show a top-5 slate to one of six calling models, execute the selected tool, observe reward, and update an online contextual bandit.

The current experiment is scoped to single-tool tasks. Multi-step agent workflows are intentionally out of scope for this pass.

## Tool Pool Construction

We decided to use MCP-Bench tools rather than ToolBench. ToolBench was archived because its folders were large and less directly aligned with our MCP marketplace setup.

The assembled pool lives in `data/pool`.

Current pool status:

- Total marketplace listings: `482`
- Base gold tools: `50`
- Valid schema variants: `75`
- Corrupted/failure variants: `150`
- Background distractors: `207`
- Fixture/static validation: `50 / 50` pass
- Live base validation: `50 / 50` pass
- Live adapter validation: `75 / 75` pass

Variant types:

- `base_gold`: original validated executable tools
- `valid_schema_variant`: wrapper variants that adapt arguments back to the base tool
- `corrupted_schema_mismatch`
- `corrupted_timeout`
- `corrupted_auth_quota`
- `corrupted_upstream_api`
- `corrupted_protocol_bug`
- `background_distractor`

Important files:

- `data/pool/listings.jsonl`: final marketplace listings
- `data/pool/manifest.json`: pool counts and validation status
- `data/pool/live_base_validation.jsonl`: cached validated live outputs for base tools
- `data/pool/live_adapter_validation.jsonl`: adapter validation results
- `src/tool_pool/`: pool models, adapters, fixtures, execution, validation, and catalog helpers
- `scripts/build_tool_pool.py`
- `scripts/finalize_tool_pool.py`
- `scripts/live_validate_base_tools.py`
- `scripts/live_validate_adapters.py`

Why we wrote the extra Python infrastructure:

- MCP-Bench server/tool formats were not uniform enough to use directly as a clean marketplace.
- We needed executable wrappers for variants, not just text descriptions.
- We needed deterministic cached execution for fast training runs.
- We needed validation gates so every base and valid variant could actually run before being used in tasks.

See also: `control-room/specs/tool_pool_assembly_log.md`.

## Task Construction

Task generation follows the MCP-Bench idea of fuzzy user tasks, but adapted to our use case: each task should ultimately require exactly one gold base tool.

Important distinction:

- `user_task`: what the agent/caller sees as the real user request.
- `marketplace_query`: the capability query used to retrieve from the marketplace.

The user task is not necessarily the marketplace query. This matters because ToolBandit is learning over marketplace discovery, not simply matching the final user-facing text.

Current task set:

- Ready tasks: `440`
- Validation errors: `0`
- File: `data/tasks/tasks_full_ready.jsonl`
- Manifest/validation: `data/tasks/full_ready_validation.json`

Task generation and validation scripts:

- `scripts/generate_tasks.py`
- `scripts/filter_ready_tasks.py`
- `scripts/validate_tasks.py`
- `src/task_generation/`

## Leakage Incident

The first full online run was invalidated.

Problem: caller prompts exposed internal listing IDs and tool names, including strings like `valid.mild`, `corrupted_timeout`, and `schema_mismatch`. That let models infer which variants were valid or broken.

Invalidated run:

- `data/runs/experiment1_online_metadata_full`

Fixes:

- Caller now sees only aliases like `tool_1`, `tool_2`, etc.
- Raw `listing_id` and raw `tool_name` are hidden.
- Descriptions and schema text are sanitized.
- The caller prompt no longer hints that ranking is based on reliability feedback.
- Judge payloads no longer include raw variant identity.
- `scripts/check_caller_payload_leakage.py` checks caller-visible payloads.

Leakage check now passes on all `440` tasks.

See: `control-room/specs/experiment1_leakage_report.md`.

## Training Runs So Far

### 1. Cheap UCB Baseline

Run:

- `data/runs/online_train_v1`

Setup:

- `440` unique tasks
- Candidate pool: top `80`
- Slate size: top `5`
- Metadata reward
- Policy selected top-ranked slate item directly

Results:

- Mean reward: `0.6159`
- Rolling last-50 reward: `0.5200`
- Clean gold recall in top-80: `0.9932`
- Clean gold selections: `257`
- Corrupted gold selections: `62`
- Wrong-family selections: `121`

### 2. Semantic-Anchored ToolBandit Hybrid

Run:

- `data/runs/toolbandit_contextual_v1`

This was our first neural contextual bandit pass. It was not the exact Toolbandit.pdf method. It used a hybrid score:

- retrieval similarity
- discounted tool success
- discounted model-tool success
- UCB exploration
- neural residual

Results:

- Mean reward: `0.7455`
- Rolling last-50 reward: `0.8400`
- Clean gold recall in top-80: `0.9932`
- Clean gold selections: `315`
- Corrupted gold selections: `34`
- Wrong-family selections: `91`

Interpretation:

This improved substantially over the cheap UCB baseline, but semantic similarity was still doing a lot of work. The result is useful, but it should be described honestly as a semantic-anchored hybrid, not pure ToolBandit.

### 3. Sanitized Live-Model Selection Run

Run:

- `data/runs/experiment1_sanitized_metadata_c4`

Setup:

- `440` tasks
- Six live calling models, round-robin:
  - `gpt-5.4-nano`
  - `x-ai/grok-4.1-fast`
  - `google/gemini-3.1-flash-lite-preview`
  - `google/gemma-4-26b-a4b-it`
  - `qwen/qwen3.5-flash-02-23`
  - `deepseek/deepseek-v3.2`
- Candidate pool: top `80`
- Slate size: top `5`
- Concurrency: `4`
- Metadata reward
- Cached validated execution outputs
- Caller selects from sanitized top-5 options

Results:

- Mean reward: `0.8500`
- Rolling last-50 reward: `0.7800`
- Total reward: `374 / 440`
- Candidate clean-gold recall in top-80: `0.9932`
- Clean gold selections: `369`
- Corrupted gold selections: `30`
- Wrong-family selections: `41`
- Selection errors: `2`

Interpretation:

The live caller helps. The LLM often picks a better option from the slate than the policy top-1. The strongest evidence of learning is reduced corrupted/background selection over time, but the overall learning curve is not dramatic because top-80 retrieval is already too forgiving.

Observed issue:

- Gold is almost always in top-80, so top-80 recall is not a hard enough retrieval metric.
- The main bottleneck shifts to top-5 slate ordering, fine-grained near-tool disambiguation, and argument generation.

## Current Policy Direction

We then patched the policy to match `Toolbandit.pdf` more closely:

```text
score(tool) =
  sigmoid(f_theta([query_embedding; model_embedding; tool_embedding]))
  + beta * sqrt(log(N_t + 1) / (n_t(tool) + 1))
```

Changes:

- Dropped wide features.
- Dropped retrieval similarity from the reranker score.
- Kept retrieval similarity only for Stage 1 candidate generation.
- Added learned model embeddings and learned tool embeddings.
- Added sliding-window replay updates.
- Added `toolbandit_score` to each caller-visible top-5 option.

Smoke run:

- `data/runs/experiment1_exact_toolbandit_smoke_5`

Caveat:

Exact ToolBandit has a much harsher cold start in our current setup. Once retrieval similarity is removed from the reranker score, early top-5 slates can become noisy until online feedback shapes the neural model. This is faithful to the paper method, but may be worse engineering for our current one-pass 440-task benchmark.

## Current Read

The best story right now is not “pure ToolBandit solved retrieval.”

The more accurate story:

1. Retrieval gets the right family into a broad candidate set.
2. A semantic-anchored bandit can improve reliability-aware slate ordering.
3. Live LLM callers add another reranking layer by choosing within the slate.
4. Exact ToolBandit needs either more repeated feedback, a better warm start, or a smaller/more controlled candidate pool to show clean learning.

Next experiments should compare:

- Retrieval-only top-5
- Exact ToolBandit
- Semantic-anchored ToolBandit hybrid
- Possibly smaller candidate pools, e.g. top-10 or top-20 instead of top-80
- Policy top-1 reward separately from LLM-selected reward

## Most Useful Artifacts

- Pool: `data/pool/manifest.json`
- Tasks: `data/tasks/tasks_full_ready.jsonl`
- Main sanitized run: `data/runs/experiment1_sanitized_metadata_c4`
- Exact ToolBandit smoke: `data/runs/experiment1_exact_toolbandit_smoke_5`
- Training log: `control-room/specs/training_run_log.md`
- Leakage report: `control-room/specs/experiment1_leakage_report.md`
- Tool pool assembly log: `control-room/specs/tool_pool_assembly_log.md`
