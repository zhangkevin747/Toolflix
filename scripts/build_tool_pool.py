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

from tool_pool.builder import PoolBuildConfig, build_pool, write_pool
from tool_pool.validate import validate_listings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ToolBandit marketplace pool artifacts.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "external/mcp-bench/mcp_servers_info.json",
        help="Path to MCP-Bench mcp_servers_info.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/pool",
        help="Directory for JSONL outputs.",
    )
    parser.add_argument("--base-count", type=int, default=50)
    parser.add_argument("--valid-count", type=int, default=75)
    parser.add_argument("--corrupted-count", type=int, default=150)
    parser.add_argument("--background-count", type=int, default=207)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--exclude-tools",
        type=Path,
        default=ROOT / "data/base_tool_denylist.txt",
        help="Optional newline-delimited tool_id denylist for base tool selection.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Build a small cheap pilot pool: 5 bases, 8 valid variants, 15 corrupted variants.",
    )
    return parser.parse_args()


def load_exclude_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


def main() -> int:
    args = parse_args()
    if args.pilot:
        args.base_count = 5
        args.valid_count = 8
        args.corrupted_count = 15
        args.background_count = 20

    config = PoolBuildConfig(
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        base_count=args.base_count,
        valid_variant_count=args.valid_count,
        corrupted_variant_count=args.corrupted_count,
        background_count=args.background_count,
        seed=args.seed,
        exclude_tool_ids=load_exclude_ids(args.exclude_tools),
    )
    result = build_pool(config)
    validation = validate_listings(result.listings)
    write_pool(result, config.output_dir)

    summary = {
        "output_dir": str(config.output_dir),
        "total_listings": len(result.listings),
        "base_tools": len(result.base_tools),
        "counts": validation.counts,
        "warnings": validation.warnings,
        "errors": validation.errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
