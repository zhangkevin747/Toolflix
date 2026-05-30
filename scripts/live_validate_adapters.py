#!/usr/bin/env python3
"""POOL STAGE 5/6 — call each reworded variant through its adapter.

Proves the reworded copies still work: take the base tool's known-good fixture,
rewrite it into the variant's argument shape, adapt it back, and live-call the base
tool. Reuses the MCP machinery from stage 4 (live_validate_base_tools).

Needs:  external/mcp-bench servers + .env
Reads:  data/pool/listings.jsonl + base_tools.jsonl + base_tool_fixtures.jsonl
Writes: data/pool/live_adapter_validation.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_pool import io
from tool_pool.adapters import adapt_arguments, visible_arguments_from_base
from tool_pool.models import AdapterSpec

import live_validate_base_tools as live_base  # MCP helpers: connect, serialize, is_error_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live-call valid schema variants through their adapters.")
    p.add_argument("--listings", type=Path, default=io.POOL_DIR / "listings.jsonl")
    p.add_argument("--base-tools", type=Path, default=io.POOL_DIR / "base_tools.jsonl")
    p.add_argument("--fixtures", type=Path, default=io.POOL_DIR / "base_tool_fixtures.jsonl")
    p.add_argument("--out", type=Path, default=io.POOL_DIR / "live_adapter_validation.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=45.0)
    return p.parse_args()


async def main_async() -> int:
    args = parse_args()
    io.load_dotenv()
    os.chdir(io.MCP_BENCH)

    from mcp_modules.server_manager import MultiServerManager
    from utils.local_server_config import LocalServerConfigLoader

    base_tools = {row["tool_id"]: row for row in io.load_jsonl(args.base_tools.resolve())}
    fixtures = {row["tool_id"]: row for row in io.load_jsonl(args.fixtures.resolve())}
    variants = [r for r in io.load_jsonl(args.listings.resolve()) if r.get("variant_type") == "valid_schema_variant"]
    if args.limit is not None:
        variants = variants[: args.limit]

    # Precompute the adapted call for each variant (offline), flag any that can't be made.
    rows: list[dict[str, Any]] = []
    for variant in variants:
        base_id = variant["base_tool_id"]
        fixture, base_tool = fixtures.get(base_id), base_tools.get(base_id)
        if not fixture or not base_tool:
            rows.append(error_row(variant, "missing_base_fixture", "base fixture/tool not found"))
            continue
        adapter = AdapterSpec(**variant["adapter"]) if variant.get("adapter") else None
        visible = visible_arguments_from_base(fixture["fixture_args"], adapter)
        adapted = adapt_arguments(visible, adapter)
        missing = [k for k in (base_tool.get("input_schema", {}).get("required") or []) if k not in adapted]
        rows.append({
            "listing_id": variant["listing_id"], "base_tool_id": base_id,
            "server": base_tool["server"], "tool_name": base_tool["tool_name"],
            "visible_args": visible, "adapted_args": adapted,
            "precheck_status": "missing_required" if missing else "ready",
            "precheck_error": f"missing required adapted fields: {missing}" if missing else None,
        })

    results = [{**r, "status": r["precheck_status"], "output_type": None, "output_preview": None, "error": r["precheck_error"]}
               for r in rows if r["precheck_status"] != "ready"]
    ready = [r for r in rows if r["precheck_status"] == "ready"]

    loader = LocalServerConfigLoader()
    server_configs = live_base.build_server_configs(loader, sorted({r["server"] for r in ready}))
    for config in server_configs:
        server = config["name"]
        server_rows = [r for r in ready if r["server"] == server]
        manager = MultiServerManager([config])
        connected = False
        try:
            await asyncio.wait_for(manager.connect_all_servers(), timeout=args.timeout)
            connected = True
            for row in server_rows:
                tool_key = f"{server}:{row['tool_name']}"
                try:
                    result = await asyncio.wait_for(
                        manager.call_tool(tool_key, row["adapted_args"], use_cache=False), timeout=args.timeout)
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

    io.write_jsonl(args.out.resolve(), results)
    passed = sum(1 for r in results if r["status"] == "pass")
    print(json.dumps({
        "out": str(args.out.resolve()), "total": len(results), "passed": passed,
        "failed": len(results) - passed,
        "by_status": live_base.count_by(results, "status"),
        "by_server": live_base.count_by(results, "server"),
    }, indent=2, sort_keys=True))
    return 0 if passed == len(results) else 1


def success_row(row: dict[str, Any], result: Any) -> dict[str, Any]:
    preview = live_base.serialize_preview(result)
    is_error = live_base.is_error_result(result, preview)
    return {**row, "status": "mcp_error_result" if is_error else "pass",
            "output_type": type(result).__name__, "output_preview": preview,
            "error": preview if is_error else None}


def error_row(variant: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    return {
        "listing_id": variant["listing_id"], "base_tool_id": variant.get("base_tool_id"),
        "server": variant.get("server"), "tool_name": variant.get("tool_name"),
        "visible_args": {}, "adapted_args": {},
        "precheck_status": status, "precheck_error": message,
        "status": status, "output_type": None, "output_preview": None, "error": message,
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
