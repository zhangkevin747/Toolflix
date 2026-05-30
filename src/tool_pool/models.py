"""The data shapes used throughout pool construction.

A ToolRecord is one raw MCP-Bench tool. A ListingRecord is one entry in our
marketplace (a base tool, a reworded copy, a broken copy, or a distractor).
AdapterSpec says how to translate a reworded variant's arguments back to the base
tool; FaultSpec says how a broken variant fails. These are plain dataclasses so the
rest of the code passes typed objects around instead of loose dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


VariantType = Literal[
    "base_gold",
    "valid_schema_variant",
    "corrupted_schema_mismatch",
    "corrupted_timeout",
    "corrupted_auth_quota",
    "corrupted_upstream_api",
    "corrupted_protocol_bug",
    "background_distractor",
]


@dataclass(frozen=True)
class ToolRecord:
    tool_id: str
    server: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    category: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterSpec:
    arg_map: dict[str, str] = field(default_factory=dict)
    nesting: dict[str, Any] = field(default_factory=dict)
    enum_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    type_casts: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    response_map: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value}


@dataclass
class FaultSpec:
    failure_type: str
    p_fail: float
    seed: int
    failure_payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ListingRecord:
    listing_id: str
    base_tool_id: str | None
    server: str
    category: str
    variant_type: VariantType
    description: str
    input_schema: dict[str, Any]
    tool_name: str
    adapter: AdapterSpec | None = None
    fault_spec: FaultSpec | None = None
    p_fail: float | None = None
    smoke_test_per_model_success: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["adapter"] = self.adapter.to_json() if self.adapter else None
        data["fault_spec"] = self.fault_spec.to_json() if self.fault_spec else None
        return data


@dataclass
class AdapterTestRecord:
    listing_id: str
    base_tool_id: str
    status: str
    details: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

