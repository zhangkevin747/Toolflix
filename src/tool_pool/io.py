"""Shared paths and tiny file helpers used across the build/task scripts.

Every script needs the same handful of things: where the repo and data live, how
to read/write JSONL, and how to load the .env file. They all live here so the
scripts stay short and say only what they actually do.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

# Repo layout. This file is <repo>/src/tool_pool/io.py, so the repo root is 3 up.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
MCP_BENCH = ROOT / "external/mcp-bench"        # the upstream tools we build the pool from
POOL_DIR = ROOT / "data/pool"                  # the assembled marketplace
TASKS_DIR = ROOT / "data/tasks"                # generated tasks
CATALOG = MCP_BENCH / "mcp_servers_info.json"  # MCP-Bench's tool listing
DENYLIST = ROOT / "data/base_tool_denylist.txt"
ENV = ROOT / ".env"


def add_src_to_path() -> None:
    """Let `python scripts/foo.py` import the `tool_pool` / `task_generation`
    packages. Call this once at the top of a script."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a .jsonl file into a list of dicts."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write dicts to a .jsonl file, one per line (keys sorted for stable diffs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dotenv(path: Path = ENV) -> None:
    """Load KEY=VALUE lines from .env into the environment (without overwriting
    anything already set)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
