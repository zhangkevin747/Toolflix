"""Loading the data and a few tiny shared helpers.

There are two data files, both one-JSON-object-per-line (JSONL):

  data/pool/listings.jsonl   -> the marketplace: ~482 tool listings.
  data/tasks/*.jsonl         -> the tasks: simple one-step user requests.

A *listing* is one tool in the marketplace. The fields we care about:
  listing_id      unique id, e.g. "weather.get_forecast.corrupted_timeout.1"
  base_tool_id    the real underlying tool this is a copy of
  variant_type    "base_gold" | "valid_schema_variant" | "corrupted_*" | "background_distractor"
  description     what the caller sees (broken tools look fine on purpose)
  input_schema    the arguments the tool expects
  fault_spec      if corrupted: how/when it fails. otherwise null.
  adapter         if a reworded variant: how to translate its args back to the base tool

A *task* is one request. The fields we care about:
  task_id            unique id
  user_task          the fuzzy request the caller model reads
  marketplace_query  the short search query used to retrieve tools (NOT the user_task)
  gold_base_tool_id  which base tool actually answers this
  fixture_args       known-good arguments for that base tool
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# The six calling models we cycle through, one per round. (Round N uses model
# N % 6.) Identities matter because the bandit learns per-model preferences.
MODELS = [
    "gpt-5.4-nano",
    "x-ai/grok-4.1-fast",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemma-4-26b-a4b-it",
    "qwen/qwen3.5-flash-02-23",
    "deepseek/deepseek-v3.2",
]

# Variant types that are *supposed* to work (not corrupted, not a distractor).
CLEAN_VARIANTS = {"base_gold", "valid_schema_variant"}

_WORD = re.compile(r"[a-z0-9]+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a .jsonl file into a list of dicts."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def words(text: str) -> list[str]:
    """Lowercase a string and split it into word tokens."""
    return _WORD.findall(text.lower())


def family_of(listing: dict[str, Any]) -> str:
    """Which base tool a listing belongs to. All copies of one tool share this."""
    return listing.get("base_tool_id") or listing["listing_id"]


def searchable_text(listing: dict[str, Any]) -> str:
    """The text the retriever indexes for a listing: name, server, category,
    description, and its argument names."""
    schema = listing.get("input_schema") or {}
    arg_names = " ".join(schema.get("properties", {}).keys())
    parts = [
        listing.get("tool_name", ""),
        listing.get("server", ""),
        listing.get("category", ""),
        listing.get("description", ""),
        arg_names,
    ]
    return " ".join(str(p) for p in parts if p)
