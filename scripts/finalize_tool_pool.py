#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tool_pool.jsonl import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize tool-pool validation artifacts.")
    parser.add_argument("--pool-dir", type=Path, default=ROOT / "data/pool")
    parser.add_argument("--denylist", type=Path, default=ROOT / "data/base_tool_denylist.txt")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    pool_dir = args.pool_dir.resolve()
    listings = load_jsonl(pool_dir / "listings.jsonl")
    base_tools = load_jsonl(pool_dir / "base_tools.jsonl")
    base_live = load_jsonl(pool_dir / "live_base_validation.jsonl")
    adapter_live = load_jsonl(pool_dir / "live_adapter_validation.jsonl")
    fixture_static = load_jsonl(pool_dir / "base_fixture_validation.jsonl")

    smoke_rows = []
    for row in base_live:
        smoke_rows.append({
            "listing_id": row["tool_id"],
            "base_tool_id": row["tool_id"],
            "validation_layer": "base_live_mcp",
            "status": row["status"],
            "server": row["server"],
            "tool_name": row["tool_name"],
        })
    for row in adapter_live:
        smoke_rows.append({
            "listing_id": row["listing_id"],
            "base_tool_id": row["base_tool_id"],
            "validation_layer": "adapter_live_mcp",
            "status": row["status"],
            "server": row["server"],
            "tool_name": row["tool_name"],
        })
    write_jsonl(pool_dir / "smoke_tests.jsonl", smoke_rows)

    denylisted = [
        line.strip() for line in args.denylist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ] if args.denylist.exists() else []

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pool_dir": str(pool_dir),
        "listing_count": len(listings),
        "base_tool_count": len(base_tools),
        "counts_by_variant_type": dict(sorted(Counter(row["variant_type"] for row in listings).items())),
        "counts_by_category": dict(sorted(Counter(row["category"] for row in listings).items())),
        "base_live_validation": summary(base_live),
        "adapter_live_validation": summary(adapter_live),
        "fixture_static_validation": summary(fixture_static),
        "smoke_test_count": len(smoke_rows),
        "denylisted_base_tool_count": len(denylisted),
        "denylisted_base_tools": denylisted,
        "status": "ready" if all_pass(base_live) and all_pass(adapter_live) and all_pass(fixture_static) else "needs_attention",
    }
    (pool_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "ready" else 1


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "passed": sum(1 for row in rows if row.get("status") == "pass"),
        "failed": sum(1 for row in rows if row.get("status") != "pass"),
        "by_status": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
    }


def all_pass(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(row.get("status") == "pass" for row in rows)


if __name__ == "__main__":
    raise SystemExit(main())
