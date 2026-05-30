#!/usr/bin/env python3
"""POOL STAGE 2/6 — review base tools and draft known-good arguments.

Reads:  data/pool/base_tools.jsonl
Writes: data/pool/base_tool_review.csv  (human-readable review)
        data/pool/base_tool_fixtures.jsonl  (the arguments used to call each tool)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_pool import io
from tool_pool.fixtures import review_and_fixture_tool, write_fixture_jsonl, write_review_csv
from tool_pool.models import ToolRecord


def main() -> int:
    p = argparse.ArgumentParser(description="Review base tools and draft fixtures.")
    p.add_argument("--base-tools", type=Path, default=io.POOL_DIR / "base_tools.jsonl")
    p.add_argument("--review-csv", type=Path, default=io.POOL_DIR / "base_tool_review.csv")
    p.add_argument("--fixtures", type=Path, default=io.POOL_DIR / "base_tool_fixtures.jsonl")
    args = p.parse_args()

    tools = [ToolRecord(**row) for row in io.load_jsonl(args.base_tools)]
    reviews = [review_and_fixture_tool(tool) for tool in tools]
    write_review_csv(args.review_csv, reviews)
    write_fixture_jsonl(args.fixtures, reviews)

    counts: dict[str, int] = {}
    for r in reviews:
        counts[r.review_status] = counts.get(r.review_status, 0) + 1
    print(json.dumps({"base_tools": len(tools), "status_counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
