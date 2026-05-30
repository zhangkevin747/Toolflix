"""Live execution backend: actually call MCP-Bench servers during a run.

The cached marketplace (marketplace.py) replays pre-recorded outputs. This module
instead keeps real MCP servers running and calls them with the arguments the caller
produced, so "did the tool work" reflects a real server response.

Each server is a subprocess, so connecting is expensive. We connect a server the
first time a tool on it is picked and keep it open for the rest of the run; a server
that fails to connect is remembered so we don't keep retrying it. MCP-Bench's
PersistentMultiServerManager does the stdio/http plumbing; we wrap it with a
synchronous `call()` so the (synchronous) training loop never has to touch asyncio.

The server-config logic mirrors scripts/live_validate_base_tools.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from tool_pool import io


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

# Many servers report API failures as an ordinary (non-error) response, so we also
# scan the text for these markers. Same list as the live-validation scripts.
ERROR_MARKERS = [
    "api error:", "http error:", "could not retrieve", "error retrieving",
    "connection error", "failed to fetch", "not found", "no matches found",
    "quota exceeded", "temporary redirect", "unauthorized", "forbidden",
]


@dataclass
class CallResult:
    ok: bool
    preview: str
    error: str | None


class LiveMCP:
    """Holds warm connections to MCP servers and calls tools on them synchronously."""

    def __init__(self, timeout: float = 45.0, use_cache: bool = True) -> None:
        io.load_dotenv()
        # Server `cwd`s are relative to the MCP-Bench root, so run from there.
        os.chdir(io.MCP_BENCH)
        if str(io.MCP_BENCH) not in sys.path:
            sys.path.insert(0, str(io.MCP_BENCH))
        from utils.local_server_config import LocalServerConfigLoader

        self.timeout = timeout
        self.use_cache = use_cache
        self._loader = LocalServerConfigLoader()
        self._loop = asyncio.new_event_loop()
        self._managers: dict[str, Any] = {}   # server -> PersistentMultiServerManager
        self._failed: dict[str, str] = {}      # server -> why it couldn't connect

    def call(self, server: str, tool_name: str, arguments: dict[str, Any]) -> CallResult:
        """Call `server:tool_name` with `arguments`, connecting the server if needed."""
        return self._loop.run_until_complete(self._call(server, tool_name, arguments))

    def close(self) -> None:
        for manager in self._managers.values():
            try:
                self._loop.run_until_complete(manager.close_all_connections())
            except Exception:
                pass
        self._managers.clear()
        try:
            self._loop.close()
        except Exception:
            pass

    # -- internals ---------------------------------------------------------
    async def _call(self, server: str, tool_name: str, arguments: dict[str, Any]) -> CallResult:
        try:
            manager = await self._ensure(server)
        except Exception as exc:
            return CallResult(ok=False, preview="", error=f"server_failed: {exc}")
        try:
            result = await asyncio.wait_for(
                manager.call_tool(f"{server}:{tool_name}", arguments, use_cache=self.use_cache),
                timeout=self.timeout,
            )
        except Exception as exc:
            return CallResult(ok=False, preview="", error=f"call_failed: {exc}")

        preview = _serialize_preview(result)
        if _is_error_result(result, preview):
            return CallResult(ok=False, preview=preview, error=preview)
        return CallResult(ok=True, preview=preview, error=None)

    async def _ensure(self, server: str):
        if server in self._failed:
            raise RuntimeError(self._failed[server])
        if server in self._managers:
            return self._managers[server]

        from mcp_modules.server_manager_persistent import PersistentMultiServerManager

        manager = PersistentMultiServerManager([self._server_config(server)])
        try:
            await asyncio.wait_for(manager.connect_all_servers(), timeout=self.timeout)
        except Exception as exc:
            self._failed[server] = f"connect_failed: {exc}"
            raise
        self._managers[server] = manager
        return manager

    def _server_config(self, server: str) -> dict[str, Any]:
        raw = self._loader.local_commands[server]
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
        return config


def _serialize_preview(result: Any, limit: int = 1200) -> str:
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


def _is_error_result(result: Any, preview: str = "") -> bool:
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
    return any(marker in preview.lower() for marker in ERROR_MARKERS)
