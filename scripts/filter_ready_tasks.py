#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter generated tasks to the currently validated ready subset.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks/tasks.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/tasks/validation.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/tasks/tasks_ready.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/tasks/ready_manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    bad_task_ids = set()
    for error in validation.get("validation_errors", []):
        duplicate = re.search(r"duplicate task_id (\S+)", error)
        if duplicate:
            bad_task_ids.add(duplicate.group(1))
        else:
            bad_task_ids.add(error.split(":", 1)[0])
    ready = [task for task in tasks if task["task_id"] not in bad_task_ids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for task in ready:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "status": "ready",
        "source_tasks": str(args.tasks.resolve()),
        "output": str(args.out.resolve()),
        "raw_task_count": len(tasks),
        "ready_task_count": len(ready),
        "filtered_task_count": len(tasks) - len(ready),
        "filtered_task_ids": sorted(bad_task_ids),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
