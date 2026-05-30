"""Stage 4 (live): run the tool the caller picked against the REAL MCP server.

Drop-in alternative to marketplace.Marketplace. Same `execute(...)` signature, same
Outcome, so loop.py doesn't change. The difference is where the output comes from:

  - base_gold / valid_schema_variant  -> adapt the caller's args back to the base
                                         tool's schema and call the real server.
  - background_distractor             -> call that real tool directly (it has no base).
  - corrupted_*                       -> first roll its synthetic fault (this is the
                                         only way a "broken" tool stays broken, since
                                         the underlying real tool actually works). If
                                         the fault does NOT fire, fall through to a real
                                         call like any other tool.

So good tools return genuine live output, the judge reads genuine output, and real
server flakiness shows up on its own; corrupted tools keep their controlled failures.
"""

from __future__ import annotations

from typing import Any

from tool_pool.adapters import adapt_arguments
from tool_pool.faults import should_fail
from tool_pool.models import AdapterSpec, FaultSpec

from .live_mcp import LiveMCP
from .marketplace import Outcome


class LiveMarketplace:
    def __init__(self, listings_by_id: dict[str, dict[str, Any]], timeout: float = 45.0, use_cache: bool = True) -> None:
        self.listings_by_id = listings_by_id
        self.mcp = LiveMCP(timeout=timeout, use_cache=use_cache)

    def execute(self, listing_id: str, arguments: dict[str, Any], attempt: int) -> Outcome:
        listing = self.listings_by_id[listing_id]

        # 1. Required arguments present? (same gate as the cached marketplace.)
        missing = self._missing_required(listing.get("input_schema") or {}, arguments)
        if missing:
            return Outcome(ok=False, output={"error": "missing_required_fields", "fields": missing}, failure_type="schema_validation_failed")

        # 2. Translate reworded-variant args back to the base tool's names.
        adapter = self._adapter(listing)
        adapted = adapt_arguments(arguments, adapter)

        # 3. Corrupted variant: maybe fire its synthetic fault before any real call.
        fault = self._fault(listing)
        if fault and should_fail(fault, attempt=attempt):
            return Outcome(ok=False, output=dict(fault.failure_payload), failure_type=fault.failure_type, adapted_arguments=adapted)

        # 4. Resolve the real server + tool to call, then call it live.
        target = self._target(listing)
        if target is None:
            return Outcome(ok=False, output={"error": "no_underlying_tool"}, failure_type="not_executable", adapted_arguments=adapted)
        server, tool_name = target

        result = self.mcp.call(server, tool_name, adapted)
        if result.ok:
            return Outcome(ok=True, output={"output_preview": result.preview}, adapted_arguments=adapted)
        return Outcome(ok=False, output={"error": result.error}, failure_type="live_call_failed", adapted_arguments=adapted)

    def close(self) -> None:
        self.mcp.close()

    # -- helpers -----------------------------------------------------------
    def _target(self, listing: dict[str, Any]) -> tuple[str, str] | None:
        """The real (server, tool_name) to call. Variants route through their base
        tool; distractors are real tools in their own right."""
        if listing.get("variant_type") == "background_distractor":
            return listing.get("server"), listing.get("tool_name")
        base = self.listings_by_id.get(listing.get("base_tool_id") or "")
        if not base:
            return None
        return base.get("server"), base.get("tool_name")

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
