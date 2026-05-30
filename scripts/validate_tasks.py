#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from task_generation.validate import load_jsonl, searchable_live_output, validate_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated ToolBandit tasks.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks/tasks.jsonl")
    parser.add_argument("--pool-dir", type=Path, default=ROOT / "data/pool")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/tasks/validation.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool_dir = args.pool_dir.resolve()
    tasks = load_jsonl(args.tasks)
    base_tools = load_jsonl(pool_dir / "base_tools.jsonl")
    listings = load_jsonl(pool_dir / "listings.jsonl")
    live_rows = load_jsonl(pool_dir / "live_base_validation.jsonl")

    errors = validate_tasks(
        tasks,
        base_tool_ids={row["tool_id"] for row in base_tools},
        listing_ids={row["listing_id"] for row in listings},
        live_outputs_by_tool={
            row["tool_id"]: searchable_live_output(row["output_preview"])
            for row in live_rows
            if row.get("status") == "pass"
        },
    )
    summary = {
        "status": "ready" if not errors else "needs_attention",
        "tasks": len(tasks),
        "validation_errors": errors,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
