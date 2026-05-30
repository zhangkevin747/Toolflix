#!/usr/bin/env python3
"""TASK STAGE 3/3 — keep only the tasks that passed validation.

Reads the validation errors, drops every task they name, and writes the clean
"ready" set that training actually uses.

Reads:  data/tasks/tasks.jsonl + data/tasks/validation.json
Writes: data/tasks/tasks_ready.jsonl + data/tasks/ready_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_pool import io


def main() -> int:
    p = argparse.ArgumentParser(description="Filter generated tasks to the validated ready subset.")
    p.add_argument("--tasks", type=Path, default=io.TASKS_DIR / "tasks.jsonl")
    p.add_argument("--validation", type=Path, default=io.TASKS_DIR / "validation.json")
    p.add_argument("--out", type=Path, default=io.TASKS_DIR / "tasks_ready.jsonl")
    p.add_argument("--summary", type=Path, default=io.TASKS_DIR / "ready_manifest.json")
    args = p.parse_args()

    tasks = io.load_jsonl(args.tasks)
    validation = io.read_json(args.validation)

    # Each validation error starts with the offending task_id (or names a duplicate).
    bad_ids = set()
    for error in validation.get("validation_errors", []):
        dup = re.search(r"duplicate task_id (\S+)", error)
        bad_ids.add(dup.group(1) if dup else error.split(":", 1)[0])

    ready = [t for t in tasks if t["task_id"] not in bad_ids]
    io.write_jsonl(args.out, ready)

    summary = {
        "status": "ready",
        "raw_task_count": len(tasks),
        "ready_task_count": len(ready),
        "filtered_task_count": len(tasks) - len(ready),
        "filtered_task_ids": sorted(bad_ids),
        "output": str(args.out.resolve()),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
