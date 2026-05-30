#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from task_generation.generator import OpenAITaskGenerator, load_dotenv, retry_generate
from task_generation.validate import searchable_live_output, validate_tasks
from tool_pool.jsonl import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple one-tool fuzzy ToolBandit tasks.")
    parser.add_argument("--pool-dir", type=Path, default=ROOT / "data/pool")
    parser.add_argument("--out", type=Path, default=ROOT / "data/tasks/tasks.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/tasks/manifest.json")
    parser.add_argument("--model", default=os.getenv("TOOLBANDIT_GENERATOR_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tasks-per-tool", type=int, default=2)
    parser.add_argument("--limit-tools", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    pool_dir = args.pool_dir.resolve()
    listings = load_jsonl(pool_dir / "listings.jsonl")
    base_tools = load_jsonl(pool_dir / "base_tools.jsonl")
    fixtures = {row["tool_id"]: row for row in load_jsonl(pool_dir / "base_tool_fixtures.jsonl")}
    live_rows = load_jsonl(pool_dir / "live_base_validation.jsonl")
    live_by_tool = {row["tool_id"]: row for row in live_rows if row.get("status") == "pass"}

    gold_listing_ids_by_base: dict[str, list[str]] = defaultdict(list)
    for listing in listings:
        if listing["variant_type"] in {"base_gold", "valid_schema_variant"}:
            base_id = listing["base_tool_id"] or listing["listing_id"]
            gold_listing_ids_by_base[base_id].append(listing["listing_id"])

    base_tools = [
        tool for tool in base_tools
        if tool["tool_id"] in fixtures and tool["tool_id"] in live_by_tool
    ]
    if args.limit_tools is not None:
        base_tools = base_tools[: args.limit_tools]

    generator = OpenAITaskGenerator(args.model, temperature=args.temperature)
    tasks = []
    skipped = []
    task_index = args.start_index

    for tool in base_tools:
        fixture = fixtures[tool["tool_id"]]
        live_output = live_by_tool[tool["tool_id"]]["output_preview"]
        gold_listing_ids = sorted(gold_listing_ids_by_base.get(tool["tool_id"], [tool["tool_id"]]))
        try:
            generated = retry_generate(
                lambda: generator.generate_for_tool(
                    tool=tool,
                    fixture=fixture,
                    live_output_preview=live_output,
                    gold_listing_ids=gold_listing_ids,
                    task_index=task_index,
                    tasks_per_tool=args.tasks_per_tool,
                ),
                retries=args.retries,
            )
        except Exception as exc:
            skipped.append({"tool_id": tool["tool_id"], "reason": str(exc)})
            continue

        if not generated:
            skipped.append({"tool_id": tool["tool_id"], "reason": "generator returned no grounded task"})
            continue
        tasks.extend(task.to_json() for task in generated)
        task_index += len(generated)

    base_tool_ids = {tool["tool_id"] for tool in base_tools}
    listing_ids = {listing["listing_id"] for listing in listings}
    live_outputs_by_tool = {
        tool_id: searchable_live_output(row["output_preview"])
        for tool_id, row in live_by_tool.items()
    }
    errors = validate_tasks(
        tasks,
        base_tool_ids=base_tool_ids,
        listing_ids=listing_ids,
        live_outputs_by_tool=live_outputs_by_tool,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, tasks)

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
