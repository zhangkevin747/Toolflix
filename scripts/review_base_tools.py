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

from tool_pool.fixtures import (
    review_and_fixture_tool,
    write_fixture_jsonl,
    write_review_csv,
)
from tool_pool.models import ToolRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review selected base tools and draft fixtures.")
    parser.add_argument(
        "--base-tools",
        type=Path,
        default=ROOT / "data/pool/base_tools.jsonl",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=ROOT / "data/pool/base_tool_review.csv",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "data/pool/base_tool_fixtures.jsonl",
    )
    return parser.parse_args()


def load_base_tools(path: Path) -> list[ToolRecord]:
    records: list[ToolRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(ToolRecord(**row))
    return records


def main() -> int:
    args = parse_args()
    records = load_base_tools(args.base_tools)
    rows = [review_and_fixture_tool(record) for record in records]
    write_review_csv(args.review_csv, rows)
    write_fixture_jsonl(args.fixtures, rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.review_status] = counts.get(row.review_status, 0) + 1
    print(json.dumps({
        "base_tools": len(records),
        "review_csv": str(args.review_csv),
        "fixtures": str(args.fixtures),
        "status_counts": counts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

