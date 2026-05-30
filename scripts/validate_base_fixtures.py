#!/usr/bin/env python3
"""POOL STAGE 3/6 — statically check fixtures against each tool's schema.

A cheap offline check before the expensive live calls: required fields present,
argument types match. Catches obvious mistakes without touching MCP servers.

Reads:  data/pool/base_tools.jsonl + base_tool_fixtures.jsonl
Writes: data/pool/base_fixture_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_pool import io
from tool_pool.catalog import required_fields, schema_properties


def main() -> int:
    p = argparse.ArgumentParser(description="Statically validate base tool fixtures.")
    p.add_argument("--base-tools", type=Path, default=io.POOL_DIR / "base_tools.jsonl")
    p.add_argument("--fixtures", type=Path, default=io.POOL_DIR / "base_tool_fixtures.jsonl")
    p.add_argument("--out", type=Path, default=io.POOL_DIR / "base_fixture_validation.jsonl")
    args = p.parse_args()

    tools = {row["tool_id"]: row for row in io.load_jsonl(args.base_tools)}
    results = []
    for fixture in io.load_jsonl(args.fixtures):
        schema = tools[fixture["tool_id"]].get("input_schema") or {}
        errors = check_args(schema, fixture.get("fixture_args") or {})
        results.append({
            "tool_id": fixture["tool_id"],
            "status": "fail" if errors else "pass",
            "errors": errors,
            "fixture_args": fixture.get("fixture_args") or {},
        })

    io.write_jsonl(args.out, results)
    failed = sum(1 for r in results if r["status"] == "fail")
    print(json.dumps({"fixtures": len(results), "failed": failed, "out": str(args.out)}, indent=2, sort_keys=True))
    return 1 if failed else 0


def check_args(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    errors = [f"missing required field {f}" for f in required_fields(schema) if f not in args]
    props = schema_properties(schema)
    for field, value in args.items():
        expected = (props.get(field) or {}).get("type")
        if expected and not matches_type(value, expected):
            errors.append(f"{field}: expected {expected}, got {type(value).__name__}")
    return errors


def matches_type(value: Any, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(matches_type(value, item) for item in expected)
    checks = {
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    return checks.get(expected, True)


if __name__ == "__main__":
    raise SystemExit(main())
