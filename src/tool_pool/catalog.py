from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .jsonl import read_json
from .models import ToolRecord


LOW_STABILITY_PATTERNS = re.compile(
    r"\b(write|delete|remove|send|post|create|update|checkout|buy|order|"
    r"trade|transfer|cancel|upload|download|file|browser|nmap|whois)\b",
    re.IGNORECASE,
)

BASE_EXCLUDE_PATTERNS = re.compile(
    r"\b(previously stored|stored\b.*\b(tensor|tensors|matrix|matrices|vector|vectors|numpy)|tensor store|in-memory tensor|send |delete |checkout|"
    r"place order|upload|write file|browser automation)\b",
    re.IGNORECASE,
)


def normalize_id(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_") or "unnamed"


def infer_category(server: str, tool_name: str, description: str) -> str:
    blob = f"{server} {tool_name} {description}".lower()
    server_id = normalize_id(server)
    if "dexpaprika" in server_id or server_id == "dex_paprika":
        return "finance"
    if "medical" in blob or "medcalc" in blob:
        return "medical"
    checks = [
        ("science", ["bio", "paper", "pubmed", "arxiv", "science", "clinical"]),
        ("maps", ["map", "geo", "place", "distance", "directions"]),
        ("reference", ["wiki", "context", "overview", "search"]),
        ("finance", ["stock", "crypto", "price", "okx"]),
        ("media", ["movie", "museum", "icon", "image", "nasa"]),
        ("utility", ["unit", "time", "math", "weather", "calculator"]),
        ("events", ["conference", "papers", "call for papers"]),
        ("web", ["reddit", "openapi", "osint", "url"]),
    ]
    for category, needles in checks:
        if any(needle in blob for needle in needles):
            return category
    return normalize_id(server)


def load_mcp_tools(info_path: Path) -> list[ToolRecord]:
    data = read_json(info_path)
    records: list[ToolRecord] = []
    for server_name, server in data.get("servers", {}).items():
        for tool_name, tool in server.get("tools", {}).items():
            description = (tool.get("description") or "").strip()
            schema = tool.get("input_schema") or {}
            tool_id = f"{normalize_id(server_name)}.{normalize_id(tool_name)}"
            category = infer_category(server_name, tool_name, description)
            records.append(
                ToolRecord(
                    tool_id=tool_id,
                    server=server_name,
                    tool_name=tool_name,
                    description=description,
                    input_schema=schema,
                    category=category,
                )
            )
    return records


def schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def required_fields(schema: dict[str, Any]) -> list[str]:
    required = schema.get("required", [])
    return [field for field in required if isinstance(field, str)]


def base_tool_score(record: ToolRecord) -> tuple[int, int, int, str]:
    props = schema_properties(record.input_schema)
    required = required_fields(record.input_schema)
    description_len = len(record.description)
    has_description = int(description_len >= 40)
    safe = int(not LOW_STABILITY_PATTERNS.search(f"{record.tool_name} {record.description}"))
    schema_simple = int(0 < len(props) <= 5 and len(required) <= 4)
    return (has_description + safe + schema_simple, safe, -len(props), record.tool_id)


def select_base_tools(
    records: list[ToolRecord],
    count: int,
    exclude_tool_ids: set[str] | None = None,
) -> list[ToolRecord]:
    exclude_tool_ids = exclude_tool_ids or set()
    eligible = [
        record
        for record in records
        if record.description
        and schema_properties(record.input_schema)
        and not record.tool_id.startswith("openapi_explorer.")
        and record.tool_id not in exclude_tool_ids
        and not BASE_EXCLUDE_PATTERNS.search(f"{record.tool_name} {record.description}")
    ]
    by_category: dict[str, list[ToolRecord]] = {}
    for record in sorted(eligible, key=base_tool_score, reverse=True):
        by_category.setdefault(record.category, []).append(record)

    selected: list[ToolRecord] = []
    seen: set[str] = set()
    categories = sorted(by_category)
    while len(selected) < count:
        progressed = False
        for category in categories:
            bucket = by_category[category]
            while bucket and bucket[0].tool_id in seen:
                bucket.pop(0)
            if not bucket:
                continue
            tool = bucket.pop(0)
            selected.append(tool)
            seen.add(tool.tool_id)
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def select_background_tools(records: list[ToolRecord], base_ids: set[str], count: int) -> list[ToolRecord]:
    candidates = [record for record in records if record.tool_id not in base_ids]
    candidates.sort(key=lambda record: (record.description == "", record.category, record.tool_id))
    return candidates[:count]
