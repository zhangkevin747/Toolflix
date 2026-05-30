#!/usr/bin/env python3
"""POOL STAGE 1/6 — build the marketplace.

Reads:  external/mcp-bench catalog + data/base_tool_denylist.txt
Writes: data/pool/{listings,base_tools,variant_candidates,adapter_tests,smoke_tests}.jsonl

Deterministic: same seed -> identical pool. This only assembles metadata; whether
the chosen tools actually run is checked later by the live-validation stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_pool import io
from tool_pool.builder import PoolBuildConfig, build_pool, write_pool
from tool_pool.validate import validate_listings


def main() -> int:
    p = argparse.ArgumentParser(description="Build the ToolBandit marketplace pool.")
    p.add_argument("--catalog", type=Path, default=io.CATALOG)
    p.add_argument("--output-dir", type=Path, default=io.POOL_DIR)
    p.add_argument("--exclude-tools", type=Path, default=io.DENYLIST)
    p.add_argument("--keep-tools", type=Path, default=io.ROOT / "data/base_tool_keep.txt",
                   help="Tools forced into the base-gold set (one id per line).")
    p.add_argument("--base-count", type=int, default=70)
    p.add_argument("--valid-count", type=int, default=75)
    p.add_argument("--corrupted-count", type=int, default=150)
    p.add_argument("--background-count", type=int, default=187)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--pilot", action="store_true", help="Tiny cheap pool: 5/8/15/20.")
    args = p.parse_args()

    if args.pilot:
        args.base_count, args.valid_count, args.corrupted_count, args.background_count = 5, 8, 15, 20

    keep = load_id_list(args.keep_tools)
    config = PoolBuildConfig(
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        base_count=args.base_count,
        valid_variant_count=args.valid_count,
        corrupted_variant_count=args.corrupted_count,
        background_count=args.background_count,
        seed=args.seed,
        exclude_tool_ids=load_denylist(args.exclude_tools),
        include_tool_ids=keep,
    )
    result = build_pool(config)
    validation = validate_listings(result.listings)
    write_pool(result, config.output_dir)

    print(json.dumps({
        "output_dir": str(config.output_dir),
        "total_listings": len(result.listings),
        "base_tools": len(result.base_tools),
        "counts": validation.counts,
        "warnings": validation.warnings,
        "errors": validation.errors,
    }, indent=2, sort_keys=True))
    return 0 if validation.ok else 1


def load_id_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_denylist(path: Path) -> set[str]:
    return set(load_id_list(path))


if __name__ == "__main__":
    raise SystemExit(main())
