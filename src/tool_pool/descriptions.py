from __future__ import annotations

from .models import ToolRecord


PREFIXES = [
    "Use this endpoint to",
    "A utility for",
    "Provides a way to",
    "Call this tool when you need to",
    "Service for",
]

SURFACE_MODES = [
    "standard interface",
    "compact interface",
    "workflow interface",
    "typed interface",
    "compatibility interface",
    "batch-ready interface",
    "low-latency interface",
]

PROFILES = [
    "profile A",
    "profile B",
    "profile C",
    "profile D",
    "profile E",
    "profile F",
    "profile G",
    "profile H",
    "profile I",
    "profile J",
    "profile K",
]


def description_for_valid_variant(base: ToolRecord, variant_level: str, index: int) -> str:
    base_desc = _clean_sentence(base.description)
    prefix = PREFIXES[index % len(PREFIXES)]
    if variant_level == "mild":
        return f"{prefix} perform this operation: {base_desc}"
    if variant_level == "medium":
        return f"{prefix} complete the same task with a request-object style interface: {base_desc}"
    return f"{prefix} run this capability through a compact structured request: {base_desc}"


def description_for_corrupted_variant(base: ToolRecord, index: int) -> str:
    base_desc = _clean_sentence(base.description)
    prefix = PREFIXES[(index + 2) % len(PREFIXES)]
    mode = SURFACE_MODES[index % len(SURFACE_MODES)]
    profile = PROFILES[index % len(PROFILES)]
    return f"{prefix} access this {mode} ({profile}) for production workflows: {base_desc}"


def fallback_description(server: str, tool_name: str) -> str:
    readable = tool_name.replace("_", " ").replace("-", " ").strip()
    return f"Tool from {server} for the {readable} operation."


def _clean_sentence(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) > 360:
        compact = compact[:357].rstrip() + "..."
    return compact.rstrip(".")
