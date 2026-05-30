"""Stage 3: picking a tool from the slate. Two pickers, same output.

  policy_pick   used in simulated mode. No LLM: just take the bandit's #1 tool and
                fill in arguments from the task's known-good fixture. Free and fast.

  LLMPicker     used in live mode. Show the slate to a real model and let it choose
                one tool and write the arguments, like a real agent would.

Both return a `Choice`.

LEAKAGE GUARD (important): the internal listing ids and names contain words like
"corrupted_timeout" or "valid_schema_variant" that reveal a tool's quality. If the
caller model saw those it could cheat (this exact bug invalidated an early
experiment). So in live mode the caller only ever sees opaque aliases ("tool_1")
and text scrubbed of those words, and we hard-fail if any leak slips through.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from tool_pool.adapters import visible_arguments_from_base

from .bandit import ScoredTool
from .data import family_of


@dataclass
class Choice:
    listing_id: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    alias: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------
# Simulated picker: take the bandit's top tool, fill args from the fixture.
# --------------------------------------------------------------------------
def policy_pick(model_id: str, task: dict[str, Any], slate: list[ScoredTool], listings_by_id: dict[str, dict[str, Any]]) -> Choice:
    # model_id is unused here (no model chooses in simulated mode) but kept so the
    # simulated and live pickers share one signature.
    if not slate:
        return Choice(listing_id=None, error="empty_slate")
    top = slate[0]
    listing = listings_by_id[top.listing_id]
    # Only the correct family has usable fixture args; wrong picks get empty args
    # (and will fail execution, which is the right outcome).
    if family_of(listing) == task["gold_base_tool_id"]:
        args = visible_arguments_from_base(task.get("fixture_args") or {}, _adapter(listing))
    else:
        args = {}
    return Choice(listing_id=top.listing_id, arguments=args, alias="tool_1")


def _adapter(listing: dict[str, Any]):
    from tool_pool.models import AdapterSpec
    payload = listing.get("adapter")
    return AdapterSpec(**payload) if payload else None


# --------------------------------------------------------------------------
# Live picker: a real model chooses from the (sanitized) slate.
# --------------------------------------------------------------------------
class LLMPicker:
    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def __call__(self, model_id: str, task: dict[str, Any], slate: list[ScoredTool], listings_by_id: dict[str, dict[str, Any]]) -> Choice:
        if not slate:
            return Choice(listing_id=None, error="empty_slate")
        if model_id not in self._clients:
            self._clients[model_id] = client_for_model(model_id)

        # Build the opaque, sanitized options and the secret alias -> real id map.
        options, alias_to_id = [], {}
        for rank, tool in enumerate(slate, start=1):
            alias = f"tool_{rank}"
            alias_to_id[alias] = tool.listing_id
            listing = listings_by_id[tool.listing_id]
            options.append({
                "tool_alias": alias,
                "rank": rank,
                "toolbandit_score": round(tool.score, 6),
                "server": listing.get("server"),
                "description": sanitize_text(str(listing.get("description", "")))[:900],
                "input_schema": sanitize_schema(listing.get("input_schema") or {"type": "object", "properties": {}}),
            })

        leak = find_leakage(options)
        if leak:
            return Choice(listing_id=None, error=f"leakage: {leak[:5]}")

        messages = [
            {"role": "system", "content":
                "You are choosing exactly one marketplace tool for a user task. "
                "Return only JSON with keys tool_alias and arguments. "
                "tool_alias must be copied exactly from the options. "
                "arguments must match that tool's input schema. "
                "toolbandit_score is the marketplace's quality score for the option; "
                "choose the tool whose score, description, and schema best fit the task."},
            {"role": "user", "content": json.dumps({
                "user_task": task["user_task"],
                "marketplace_query": task["marketplace_query"],
                "tool_options": options,
            }, ensure_ascii=False)},
        ]
        try:
            raw = call_model(self._clients[model_id], model_id, messages, max_tokens=500)
            payload = parse_json(raw)
            alias = payload.get("tool_alias")
            if alias not in alias_to_id:
                return Choice(listing_id=None, alias=alias if isinstance(alias, str) else None, error=f"invalid_alias: {alias}")
            args = payload.get("arguments")
            return Choice(listing_id=alias_to_id[alias], arguments=args if isinstance(args, dict) else {}, alias=alias)
        except Exception as exc:
            return Choice(listing_id=None, error=str(exc))


# --------------------------------------------------------------------------
# Sanitization + leakage detection (used by both the picker and the judge).
# --------------------------------------------------------------------------
_NAME_SUFFIXES = [
    re.compile(r"_(mild|medium|aggressive)_variant_\d+$", re.I),
    re.compile(r"_valid_(mild|medium|aggressive)_\d+$", re.I),
    re.compile(r"_(schema_mismatch|timeout|auth_quota|upstream_api|protocol_bug)_\d+$", re.I),
]
_REPLACE = {
    "schema_mismatch": "schema", "auth_quota": "authorization", "upstream_api": "upstream service",
    "protocol_bug": "protocol", "corrupted": "service", "valid_schema_variant": "tool",
    "base_gold": "tool", "background_distractor": "tool",
}
_LEAK = re.compile(
    r"(^|[^a-z0-9])(corrupted|schema_mismatch|auth_quota|upstream_api|protocol_bug|"
    r"valid_schema_variant|base_gold|background_distractor|mild_variant|medium_variant|"
    r"aggressive_variant|valid[._-](mild|medium|aggressive)|timeout[._-]?[0-9]+)([^a-z0-9]|$)",
    re.I,
)


def sanitize_text(text: str) -> str:
    for pattern in _NAME_SUFFIXES:
        text = pattern.sub("", text)
    for source, target in _REPLACE.items():
        text = re.sub(source, target, text, flags=re.I)
    return re.sub(r"\b(valid|variant)\b", "tool", text, flags=re.I)


def sanitize_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key in {"title", "description"} and isinstance(value, str):
                out[key] = sanitize_text(value)          # scrub human-readable text
            elif key == "properties" and isinstance(value, dict):
                out[key] = {name: sanitize_schema(v) for name, v in value.items()}  # keep arg names (caller needs them)
            else:
                out[key] = sanitize_schema(value)
        return out
    if isinstance(schema, list):
        return [sanitize_schema(item) for item in schema]
    return schema


def find_leakage(payload: Any) -> list[str]:
    """Return any quality-revealing words found in what the caller would see."""
    text = json.dumps(payload, ensure_ascii=False, default=str)
    found: list[str] = []
    for match in _LEAK.finditer(text):
        token = match.group(2)
        if token not in found:
            found.append(token)
    return found


# --------------------------------------------------------------------------
# Thin OpenAI / OpenRouter plumbing. (Live mode only.)
# --------------------------------------------------------------------------
_OPENAI_NATIVE = {"gpt-5.4-nano", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"}


def client_for_model(model_id: str):
    from openai import OpenAI
    if model_id in _OPENAI_NATIVE:
        return OpenAI()
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))


def call_model(client, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    """One chat completion at temperature 0. Asks for JSON; retries without the
    json_object flag for models that don't support it."""
    base = {"model": model, "messages": messages, "temperature": 0, "max_completion_tokens": max_tokens}
    try:
        resp = client.chat.completions.create(**base, response_format={"type": "json_object"})
    except Exception:
        resp = client.chat.completions.create(**base)
    return resp.choices[0].message.content or ""


def parse_json(text: str) -> dict[str, Any]:
    """Lenient JSON parse: strips ``` fences and falls back to the first {...} block."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}
