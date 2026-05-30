#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tool_pool.adapters import adapt_arguments, visible_arguments_from_base
from tool_pool.jsonl import write_jsonl
from tool_pool.models import AdapterSpec

import live_validate_base_tools as live_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-call valid schema variants through their adapters.")
    parser.add_argument("--listings", type=Path, default=ROOT / "data/pool/listings.jsonl")
    parser.add_argument("--base-tools", type=Path, default=ROOT / "data/pool/base_tools.jsonl")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "data/pool/base_tool_fixtures.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/pool/live_adapter_validation.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=45.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def main_async() -> int:
    args = parse_args()
    args.listings = args.listings.resolve()
    args.base_tools = args.base_tools.resolve()
    args.fixtures = args.fixtures.resolve()
    args.out = args.out.resolve()

    live_base.load_dotenv(ROOT / ".env")
    import os

    os.chdir(live_base.MCP_BENCH)
    from mcp_modules.server_manager import MultiServerManager
    from utils.local_server_config import LocalServerConfigLoader

    base_tools = {row["tool_id"]: row for row in load_jsonl(args.base_tools)}
    fixtures = {row["tool_id"]: row for row in load_jsonl(args.fixtures)}
    variants = [
        row for row in load_jsonl(args.listings)
        if row.get("variant_type") == "valid_schema_variant"
    ]
    if args.limit is not None:
        variants = variants[: args.limit]

    test_rows: list[dict[str, Any]] = []
    for variant in variants:
        base_id = variant["base_tool_id"]
        fixture = fixtures.get(base_id)
        base_tool = base_tools.get(base_id)
        if not fixture or not base_tool:
            test_rows.append(error_row(variant, fixture, "missing_base_fixture", "base fixture/tool not found"))
            continue
        adapter_payload = variant.get("adapter") or {}
        adapter = AdapterSpec(**adapter_payload) if adapter_payload else None
        visible_args = visible_arguments_from_base(fixture["fixture_args"], adapter)
        adapted_args = adapt_arguments(visible_args, adapter)
        missing = [
            key for key in (base_tool.get("input_schema", {}).get("required") or [])
            if key not in adapted_args
        ]
        test_rows.append({
            "listing_id": variant["listing_id"],
            "base_tool_id": base_id,
            "server": base_tool["server"],
            "tool_name": base_tool["tool_name"],
            "visible_args": visible_args,
            "adapted_args": adapted_args,
            "precheck_status": "missing_required" if missing else "ready",
            "precheck_error": f"missing required adapted fields: {missing}" if missing else None,
        })

    results: list[dict[str, Any]] = []
    ready_rows = [row for row in test_rows if row["precheck_status"] == "ready"]
    loader = LocalServerConfigLoader()
    server_configs = live_base.build_server_configs(loader, sorted({row["server"] for row in ready_rows}))

    for row in test_rows:
        if row["precheck_status"] != "ready":
            results.append({
                **row,
                "status": row["precheck_status"],
                "output_type": None,
                "output_preview": None,
                "error": row["precheck_error"],
            })

    for config in server_configs:
        server = config["name"]
        server_rows = [row for row in ready_rows if row["server"] == server]
        manager = MultiServerManager([config])
        connected = False
        try:
            await asyncio.wait_for(manager.connect_all_servers(), timeout=args.timeout)
            connected = True
            for row in server_rows:
                tool_key = f"{server}:{row['tool_name']}"
                try:
                    result = await asyncio.wait_for(
                        manager.call_tool(tool_key, row["adapted_args"], use_cache=False),
                        timeout=args.timeout,
                    )
                    results.append(success_row(row, result))
                except Exception as exc:
                    results.append({**row, "status": "call_failed", "output_type": None, "output_preview": None, "error": str(exc)})
        except Exception as exc:
            for row in server_rows:
                results.append({**row, "status": "server_failed", "output_type": None, "output_preview": None, "error": str(exc)})
        finally:
            if connected:
                try:
                    await manager.close_all_connections()
                except Exception:
                    pass

    write_jsonl(args.out, results)
    passed = sum(1 for row in results if row["status"] == "pass")
    summary = {
        "out": str(args.out),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "by_status": live_base.count_by(results, "status"),
        "by_server": live_base.count_by(results, "server"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed == len(results) else 1


def success_row(row: dict[str, Any], result: Any) -> dict[str, Any]:
    preview = live_base.serialize_preview(result)
    is_error = live_base.is_error_result(result, preview)
    return {
        **row,
        "status": "mcp_error_result" if is_error else "pass",
        "output_type": type(result).__name__,
        "output_preview": preview,
        "error": preview if is_error else None,
    }


def error_row(
    variant: dict[str, Any],
    fixture: dict[str, Any] | None,
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        "listing_id": variant["listing_id"],
        "base_tool_id": variant.get("base_tool_id"),
        "server": variant.get("server"),
        "tool_name": variant.get("tool_name"),
        "visible_args": {},
        "adapted_args": {},
        "precheck_status": status,
        "precheck_error": message,
        "status": status,
        "output_type": None,
        "output_preview": None,
        "error": message,
    }


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
