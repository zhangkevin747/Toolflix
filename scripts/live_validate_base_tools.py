#!/usr/bin/env python3
"""POOL STAGE 4/6 — actually call each base tool through its real MCP server.

This is the step that proves a "gold" tool really works. A tool can look fine on
paper and still fail when called (dead API, redirect, or an error returned as a
normal-looking response). We connect to each server, call the tool with its fixture,
and mark pass/fail — treating disguised error payloads (see `is_error_result`) as
failures too. Tools that fail here get moved to the denylist.

Needs:  external/mcp-bench servers running locally + any API keys in .env
Reads:  data/pool/base_tools.jsonl + base_tool_fixtures.jsonl
Writes: data/pool/live_base_validation.jsonl  (output previews reused everywhere downstream)
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

sys.path.insert(0, str(io.MCP_BENCH))  # so we can import MCP-Bench's server manager


# MCP-Bench server name -> its folder under external/mcp-bench/mcp_servers/.
DIR_MAPPING = {
    "Bibliomantic": "bibliomantic-mcp-server",
    "BioMCP": "biomcp",
    "Call for Papers": "call-for-papers-mcp/call-for-papers-mcp-main",
    "Car Price Evaluator": "car-price-mcp-main",
    "Context7": "context7-mcp",
    "DEX Paprika": "dexpaprika-mcp",
    "FruityVice": "fruityvice-mcp",
    "Game Trends": "game-trends-mcp",
    "Huge Icons": "hugeicons-mcp-server",
    "Hugging Face": "huggingface-mcp-server",
    "Math MCP": "math-mcp",
    "NixOS": "mcp-nixos",
    "OSINT Intelligence": "mcp-osint-server",
    "Reddit": "mcp-reddit",
    "National Parks": "mcp-server-nationalparks",
    "Unit Converter": "unit-converter-mcp",
    "Medical Calculator": "medcalc",
    "Metropolitan Museum": "metmuseum-mcp",
    "Movie Recommender": "movie-recommender-mcp/movie-reccomender-mcp",
    "NASA Data": "nasa-mcp",
    "OKX Exchange": "okx-mcp",
    "Paper Search": "paper-search-mcp",
    "Scientific Computing": "scientific_computation_mcp",
    "Weather Data": "weather_mcp",
    "Wikipedia": "wikipedia-mcp",
    "Google Maps": "mcp-google-map",
    "OpenAPI Explorer": "openapi-mcp-server",
    "Time MCP": "time-mcp",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live-call selected base MCP tools.")
    p.add_argument("--base-tools", type=Path, default=io.POOL_DIR / "base_tools.jsonl")
    p.add_argument("--fixtures", type=Path, default=io.POOL_DIR / "base_tool_fixtures.jsonl")
    p.add_argument("--out", type=Path, default=io.POOL_DIR / "live_base_validation.jsonl")
    p.add_argument("--only-server", default=None)
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
    fixtures = io.load_jsonl(args.fixtures.resolve())
    if args.only_server:
        fixtures = [row for row in fixtures if row["server"] == args.only_server]
    if args.limit is not None:
        fixtures = fixtures[: args.limit]

    loader = LocalServerConfigLoader()
    server_configs = build_server_configs(loader, sorted({row["server"] for row in fixtures}))

    results: list[dict[str, Any]] = []
    for config in server_configs:
        server = config["name"]
        server_fixtures = [row for row in fixtures if row["server"] == server]
        manager = MultiServerManager([config])
        connected = False
        try:
            await asyncio.wait_for(manager.connect_all_servers(), timeout=args.timeout)
            connected = True
            for fixture in server_fixtures:
                tool = base_tools[fixture["tool_id"]]
                tool_key = f"{server}:{tool['tool_name']}"
                try:
                    result = await asyncio.wait_for(
                        manager.call_tool(tool_key, fixture["fixture_args"], use_cache=False),
                        timeout=args.timeout,
                    )
                    results.append(success_row(fixture, result))
                except Exception as exc:
                    results.append(error_row(fixture, "call_failed", exc))
        except Exception as exc:
            for fixture in server_fixtures:
                results.append(error_row(fixture, "server_failed", exc))
        finally:
            if connected:
                try:
                    await manager.close_all_connections()
                except Exception:
                    pass

    io.write_jsonl(args.out.resolve(), results)
    passed = sum(1 for row in results if row["status"] == "pass")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "by_status": count_by(results, "status"),
        "by_server": count_by(results, "server"),
    }, indent=2, sort_keys=True))
    return 0 if passed == len(results) else 1


def build_server_configs(loader: Any, servers: list[str]) -> list[dict[str, Any]]:
    configs = []
    for server in servers:
        raw = loader.local_commands[server]
        config = {
            "name": server,
            "command": raw.get("cmd", "").split(),
            "env": {key: os.environ[key] for key in raw.get("env", []) if key in os.environ},
            "cwd": f"mcp_servers/{DIR_MAPPING.get(server, server.lower().replace(' ', '-'))}",
            "description": "",
        }
        if raw.get("transport") == "http":
            config["transport"] = "http"
            config["port"] = raw.get("port", 3001)
            config["endpoint"] = raw.get("endpoint", "/mcp")
        configs.append(config)
    return configs


def success_row(fixture: dict[str, Any], result: Any) -> dict[str, Any]:
    preview = serialize_preview(result)
    is_error = is_error_result(result, preview)
    return {
        "tool_id": fixture["tool_id"],
        "server": fixture["server"],
        "tool_name": fixture["tool_name"],
        "status": "mcp_error_result" if is_error else "pass",
        "fixture_args": fixture["fixture_args"],
        "output_type": type(result).__name__,
        "output_preview": preview,
        "error": preview if is_error else None,
    }


def error_row(fixture: dict[str, Any], status: str, exc: BaseException) -> dict[str, Any]:
    return {
        "tool_id": fixture["tool_id"],
        "server": fixture["server"],
        "tool_name": fixture["tool_name"],
        "status": status,
        "fixture_args": fixture["fixture_args"],
        "output_type": None,
        "output_preview": None,
        "error": str(exc),
    }


def serialize_preview(result: Any, limit: int = 1200) -> str:
    try:
        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        elif hasattr(result, "dict"):
            payload = result.dict()
        else:
            payload = result
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(result)
    return text[:limit]


def is_error_result(result: Any, preview: str = "") -> bool:
    """Many MCP servers return API errors as a normal, non-error response. Treat those
    as failures by also scanning the text for error markers."""
    if getattr(result, "isError", False):
        return True
    if hasattr(result, "model_dump"):
        try:
            if result.model_dump().get("isError") is True:
                return True
        except Exception:
            pass
    if isinstance(result, dict) and result.get("isError") is True:
        return True
    markers = [
        "api error:", "http error:", "could not retrieve", "error retrieving",
        "connection error", "failed to fetch", "not found", "no matches found",
        "quota exceeded", "temporary redirect", "unauthorized", "forbidden",
    ]
    return any(marker in preview.lower() for marker in markers)


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get(key))] = counts.get(str(row.get(key)), 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
