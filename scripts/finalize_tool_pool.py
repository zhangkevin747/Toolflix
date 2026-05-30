#!/usr/bin/env python3
"""POOL STAGE 6/6 — write the manifest that marks the pool ready.

Gathers the validation results from the earlier stages into one smoke_tests.jsonl and
a manifest.json with counts and pass/fail status. If every check passed, status is
"ready" and the pool can be used for task generation and training.

Reads:  data/pool/*.jsonl (listings, base_tools, the three validation files) + denylist
Writes: data/pool/smoke_tests.jsonl + data/pool/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_pool import io


def load_jsonl_or_empty(path: Path) -> list[dict[str, Any]]:
    return io.load_jsonl(path) if path.exists() else []


def main() -> int:
    p = argparse.ArgumentParser(description="Finalize tool-pool validation artifacts.")
    p.add_argument("--pool-dir", type=Path, default=io.POOL_DIR)
    p.add_argument("--denylist", type=Path, default=io.DENYLIST)
    args = p.parse_args()
    pool_dir = args.pool_dir.resolve()

    listings = load_jsonl_or_empty(pool_dir / "listings.jsonl")
    base_tools = load_jsonl_or_empty(pool_dir / "base_tools.jsonl")
    base_live = load_jsonl_or_empty(pool_dir / "live_base_validation.jsonl")
    adapter_live = load_jsonl_or_empty(pool_dir / "live_adapter_validation.jsonl")
    fixture_static = load_jsonl_or_empty(pool_dir / "base_fixture_validation.jsonl")

    smoke_rows = [
        {"listing_id": r["tool_id"], "base_tool_id": r["tool_id"], "validation_layer": "base_live_mcp",
         "status": r["status"], "server": r["server"], "tool_name": r["tool_name"]}
        for r in base_live
    ] + [
        {"listing_id": r["listing_id"], "base_tool_id": r["base_tool_id"], "validation_layer": "adapter_live_mcp",
         "status": r["status"], "server": r["server"], "tool_name": r["tool_name"]}
        for r in adapter_live
    ]
    io.write_jsonl(pool_dir / "smoke_tests.jsonl", smoke_rows)

    denylisted = [
        line.strip() for line in args.denylist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ] if args.denylist.exists() else []

    ready = all_pass(base_live) and all_pass(adapter_live) and all_pass(fixture_static)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pool_dir": str(pool_dir),
        "listing_count": len(listings),
        "base_tool_count": len(base_tools),
        "counts_by_variant_type": dict(sorted(Counter(r["variant_type"] for r in listings).items())),
        "counts_by_category": dict(sorted(Counter(r["category"] for r in listings).items())),
        "base_live_validation": summarize(base_live),
        "adapter_live_validation": summarize(adapter_live),
        "fixture_static_validation": summarize(fixture_static),
        "smoke_test_count": len(smoke_rows),
        "denylisted_base_tool_count": len(denylisted),
        "denylisted_base_tools": denylisted,
        "status": "ready" if ready else "needs_attention",
    }
    (pool_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if ready else 1


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "passed": sum(1 for r in rows if r.get("status") == "pass"),
        "failed": sum(1 for r in rows if r.get("status") != "pass"),
        "by_status": dict(sorted(Counter(str(r.get("status")) for r in rows).items())),
    }


def all_pass(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(r.get("status") == "pass" for r in rows)


if __name__ == "__main__":
    raise SystemExit(main())
