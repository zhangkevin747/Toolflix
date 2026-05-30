#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tool_pool.catalog import required_fields, schema_properties


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statically validate base tool fixtures.")
    parser.add_argument("--base-tools", type=Path, default=ROOT / "data/pool/base_tools.jsonl")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "data/pool/base_tool_fixtures.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/pool/base_fixture_validation.jsonl")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    tools = {row["tool_id"]: row for row in load_jsonl(args.base_tools)}
    fixtures = load_jsonl(args.fixtures)
    results = []
    errors = 0
    for fixture in fixtures:
        tool = tools[fixture["tool_id"]]
        schema = tool.get("input_schema") or {}
        fixture_args = fixture.get("fixture_args") or {}
        row_errors = validate_args(schema, fixture_args)
        if row_errors:
            errors += 1
        results.append({
            "tool_id": fixture["tool_id"],
            "status": "fail" if row_errors else "pass",
            "errors": row_errors,
            "fixture_args": fixture_args,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    summary = {
        "fixtures": len(fixtures),
        "failed": errors,
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    props = schema_properties(schema)
    for field in required_fields(schema):
        if field not in args:
            errors.append(f"missing required field {field}")
    for field, value in args.items():
        spec = props.get(field)
        if not spec:
            continue
        expected = spec.get("type")
        if expected and not _matches_type(value, expected):
            errors.append(f"{field}: expected {expected}, got {type(value).__name__}")
    return errors


def _matches_type(value: Any, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(value, item) for item in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


if __name__ == "__main__":
    raise SystemExit(main())

