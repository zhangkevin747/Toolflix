from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .adapters import adapt_arguments
from .faults import should_fail
from .models import ListingRecord


BaseToolCaller = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass
class ExecutionResult:
    listing_id: str
    base_tool_id: str | None
    ok: bool
    output: dict[str, Any]
    adapted_arguments: dict[str, Any]
    failure_type: str | None = None


class MarketplaceExecutor:
    """Executes marketplace listings without modifying the underlying MCP servers."""

    def __init__(self, listings: list[ListingRecord], base_tool_caller: BaseToolCaller):
        self._listings = {listing.listing_id: listing for listing in listings}
        self._base_tool_caller = base_tool_caller

    def execute(
        self,
        listing_id: str,
        arguments: dict[str, Any],
        attempt: int = 0,
    ) -> ExecutionResult:
        listing = self._listings[listing_id]
        adapted = adapt_arguments(arguments, listing.adapter)

        if listing.fault_spec and should_fail(listing.fault_spec, attempt=attempt):
            output = self._failure_output(listing)
            return ExecutionResult(
                listing_id=listing.listing_id,
                base_tool_id=listing.base_tool_id,
                ok=False,
                output=output,
                adapted_arguments=adapted,
                failure_type=listing.fault_spec.failure_type,
            )

        if not listing.base_tool_id:
            return ExecutionResult(
                listing_id=listing.listing_id,
                base_tool_id=None,
                ok=False,
                output={"error": "background_distractor_not_executable_via_wrapper"},
                adapted_arguments=adapted,
                failure_type="background_distractor",
            )

        output = self._base_tool_caller(listing.base_tool_id, adapted)
        return ExecutionResult(
            listing_id=listing.listing_id,
            base_tool_id=listing.base_tool_id,
            ok=True,
            output=output,
            adapted_arguments=adapted,
        )

    def _failure_output(self, listing: ListingRecord) -> dict[str, Any]:
        assert listing.fault_spec is not None
        if listing.fault_spec.failure_type == "timeout":
            # Keep local tests fast while still surfacing timeout semantics.
            time.sleep(0.01)
        return dict(listing.fault_spec.failure_payload)

