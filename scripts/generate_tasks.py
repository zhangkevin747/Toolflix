#!/usr/bin/env python3
"""TASK STAGE 1/3 — generate simple one-tool tasks with an LLM.

For each base tool that passed live validation, ask the generator model for a fuzzy
user task grounded in that tool's real output. Validates as it goes and writes the
raw task set.

Needs:  OPENAI_API_KEY (in .env)
Reads:  data/pool/* (listings, base_tools, fixtures, live_base_validation)
Writes: data/tasks/tasks.jsonl + data/tasks/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_pool import io
from task_generation.generator import OpenAITaskGenerator, retry_generate
from task_generation.validate import searchable_live_output, validate_tasks


def main() -> int:
    p = argparse.ArgumentParser(description="Generate simple one-tool fuzzy ToolBandit tasks.")
    p.add_argument("--pool-dir", type=Path, default=io.POOL_DIR)
    p.add_argument("--out", type=Path, default=io.TASKS_DIR / "tasks.jsonl")
    p.add_argument("--summary", type=Path, default=io.TASKS_DIR / "manifest.json")
    p.add_argument("--model", default=os.getenv("TOOLBANDIT_GENERATOR_MODEL", "gpt-4.1-mini"))
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--tasks-per-tool", type=int, default=2)
    p.add_argument("--limit-tools", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()
    io.load_dotenv()

    pool = args.pool_dir.resolve()
    listings = io.load_jsonl(pool / "listings.jsonl")
    fixtures = {r["tool_id"]: r for r in io.load_jsonl(pool / "base_tool_fixtures.jsonl")}
    live_by_tool = {r["tool_id"]: r for r in io.load_jsonl(pool / "live_base_validation.jsonl") if r.get("status") == "pass"}

    # For each base family, the "gold" listings (the base tool + its working variants).
    gold_by_base: dict[str, list[str]] = defaultdict(list)
    for listing in listings:
        if listing["variant_type"] in {"base_gold", "valid_schema_variant"}:
            gold_by_base[listing["base_tool_id"] or listing["listing_id"]].append(listing["listing_id"])

    # Only generate for base tools that have both a fixture and a passing live call.
    base_tools = [t for t in io.load_jsonl(pool / "base_tools.jsonl")
                  if t["tool_id"] in fixtures and t["tool_id"] in live_by_tool]
    if args.limit_tools is not None:
        base_tools = base_tools[: args.limit_tools]

    generator = OpenAITaskGenerator(args.model, temperature=args.temperature)
    tasks, skipped, index = [], [], args.start_index
    for tool in base_tools:
        gold_ids = sorted(gold_by_base.get(tool["tool_id"], [tool["tool_id"]]))
        try:
            generated = retry_generate(lambda: generator.generate_for_tool(
                tool=tool, fixture=fixtures[tool["tool_id"]],
                live_output_preview=live_by_tool[tool["tool_id"]]["output_preview"],
                gold_listing_ids=gold_ids, task_index=index, tasks_per_tool=args.tasks_per_tool,
            ), retries=args.retries)
        except Exception as exc:
            skipped.append({"tool_id": tool["tool_id"], "reason": str(exc)})
            continue
        if not generated:
            skipped.append({"tool_id": tool["tool_id"], "reason": "generator returned no grounded task"})
            continue
        tasks.extend(t.to_json() for t in generated)
        index += len(generated)

    errors = validate_tasks(
        tasks,
        base_tool_ids={t["tool_id"] for t in base_tools},
        listing_ids={l["listing_id"] for l in listings},
        live_outputs_by_tool={tid: searchable_live_output(r["output_preview"]) for tid, r in live_by_tool.items()},
    )
    io.write_jsonl(args.out, tasks)

    summary = {
        "status": "ready" if not errors else "needs_attention",
        "generator_model": args.model,
        "tasks_per_tool": args.tasks_per_tool,
        "base_tools_considered": len(base_tools),
        "tasks_generated": len(tasks),
        "skipped": skipped,
        "validation_errors": errors,
        "output": str(args.out.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
