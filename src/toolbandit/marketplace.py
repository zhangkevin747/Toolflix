"""Stage 4: execution. Run the tool the caller picked.

We never call the real MCP servers during training (slow, flaky, costs money).
Instead, base tools were called once ahead of time and their outputs cached in
data/pool/live_base_validation.jsonl. Here we:

  - check the caller's arguments have the required fields,
  - translate them back to the base tool's argument names (for reworded variants),
  - if the listing is a corrupted variant, maybe inject its synthetic failure,
  - otherwise return the cached real output.

So "good" tools return real cached output, and "broken" tools fail exactly the
way their fault_spec says they should. The caller can't tell which is which up
front, that's the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tool_pool.adapters import adapt_arguments
from tool_pool.faults import should_fail
from tool_pool.models import AdapterSpec, FaultSpec

from .data import load_jsonl


@dataclass
class Outcome:
    ok: bool
    output: dict[str, Any]
    failure_type: str | None = None
    adapted_arguments: dict[str, Any] = field(default_factory=dict)


def load_cached_outputs(path: Path) -> dict[str, str]:
    """base_tool_id -> a preview of its real output, for tools that passed validation."""
    rows = load_jsonl(path)
    return {
        row["tool_id"]: row["output_preview"]
        for row in rows
        if row.get("status") == "pass" and row.get("output_preview") is not None
    }


class Marketplace:
    def __init__(self, listings_by_id: dict[str, dict[str, Any]], cached_outputs: dict[str, str]) -> None:
        self.listings_by_id = listings_by_id
        self.cached_outputs = cached_outputs

    def close(self) -> None:
        """No live connections to release; here so loop.py can treat both marketplaces alike."""

    def execute(self, listing_id: str, arguments: dict[str, Any], attempt: int) -> Outcome:
        listing = self.listings_by_id[listing_id]

        # 1. Required arguments present?
        missing = self._missing_required(listing.get("input_schema") or {}, arguments)
        if missing:
            return Outcome(ok=False, output={"error": "missing_required_fields", "fields": missing}, failure_type="schema_validation_failed")

        # 2. Translate reworded-variant args back to the base tool's names.
        adapter = self._adapter(listing)
        adapted = adapt_arguments(arguments, adapter)

        # 3. If this is a broken tool, maybe fire its failure. `attempt` seeds the
        #    randomness so the same call fails the same way every run.
        fault = self._fault(listing)
        if fault and should_fail(fault, attempt=attempt):
            return Outcome(ok=False, output=dict(fault.failure_payload), failure_type=fault.failure_type, adapted_arguments=adapted)

        # 4. Otherwise return the cached real output of the underlying base tool.
        base_id = listing.get("base_tool_id")
        if not base_id or base_id not in self.cached_outputs:
            return Outcome(ok=False, output={"error": "no_cached_output"}, failure_type="not_executable", adapted_arguments=adapted)
        return Outcome(ok=True, output={"output_preview": self.cached_outputs[base_id]}, adapted_arguments=adapted)

    @staticmethod
    def _missing_required(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
        return [f for f in (schema.get("required") or []) if f not in arguments]

    @staticmethod
    def _adapter(listing: dict[str, Any]) -> AdapterSpec | None:
        payload = listing.get("adapter")
        return AdapterSpec(**payload) if payload else None

    @staticmethod
    def _fault(listing: dict[str, Any]) -> FaultSpec | None:
        payload = listing.get("fault_spec")
        return FaultSpec(**payload) if payload else None
