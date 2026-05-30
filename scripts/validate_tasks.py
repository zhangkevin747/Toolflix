#!/usr/bin/env python3
"""TASK STAGE 2/3 — validate generated tasks against the pool.

Standalone re-check (generate_tasks also validates inline): every task points at a
real tool/listings, uses the judge reward source, and has an answer grounded in the
tool's live output.

Reads:  data/tasks/tasks.jsonl + data/pool/*
Writes: data/tasks/validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_pool import io
from task_generation.validate import searchable_live_output, validate_tasks


def main() -> int:
    p = argparse.ArgumentParser(description="Validate generated ToolBandit tasks.")
    p.add_argument("--tasks", type=Path, default=io.TASKS_DIR / "tasks.jsonl")
    p.add_argument("--pool-dir", type=Path, default=io.POOL_DIR)
    p.add_argument("--summary", type=Path, default=io.TASKS_DIR / "validation.json")
    args = p.parse_args()
    pool = args.pool_dir.resolve()

    tasks = io.load_jsonl(args.tasks)
    errors = validate_tasks(
        tasks,
        base_tool_ids={r["tool_id"] for r in io.load_jsonl(pool / "base_tools.jsonl")},
        listing_ids={r["listing_id"] for r in io.load_jsonl(pool / "listings.jsonl")},
        live_outputs_by_tool={
            r["tool_id"]: searchable_live_output(r["output_preview"])
            for r in io.load_jsonl(pool / "live_base_validation.jsonl") if r.get("status") == "pass"
        },
    )
    summary = {"status": "ready" if not errors else "needs_attention", "tasks": len(tasks), "validation_errors": errors}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
