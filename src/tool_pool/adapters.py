from __future__ import annotations

from copy import deepcopy
from typing import Any

from .catalog import required_fields, schema_properties
from .models import AdapterSpec, ToolRecord


RENAMES = [
    ("query", "search_text"),
    ("q", "search_text"),
    ("id", "resource_id"),
    ("location", "place"),
    ("city", "place"),
    ("value", "reading"),
    ("amount", "quantity"),
    ("date", "target_date"),
    ("url", "link"),
    ("name", "label"),
]


def _renamed_field(field: str, variant_index: int) -> str:
    lower = field.lower()
    for source, target in RENAMES:
        if lower == source:
            return target
    if variant_index % 3 == 0:
        return f"{field}_value"
    if variant_index % 3 == 1:
        return f"target_{field}"
    return f"{field}_input"


def _schema_with_renamed_fields(schema: dict[str, Any], variant_index: int) -> tuple[dict[str, Any], AdapterSpec]:
    schema_copy = deepcopy(schema)
    props = schema_properties(schema)
    required = required_fields(schema)
    new_props: dict[str, Any] = {}
    arg_map: dict[str, str] = {}
    required_map: dict[str, str] = {}

    for field, spec in props.items():
        if field in required or len(props) <= 2:
            new_field = _renamed_field(field, variant_index)
        else:
            new_field = field
        new_props[new_field] = deepcopy(spec)
        arg_map[new_field] = field
        required_map[field] = new_field

    schema_copy["properties"] = new_props
    if required:
        schema_copy["required"] = [required_map.get(field, field) for field in required]
    return schema_copy, AdapterSpec(arg_map=arg_map)


def _schema_with_nested_payload(schema: dict[str, Any], variant_index: int) -> tuple[dict[str, Any], AdapterSpec]:
    props = schema_properties(schema)
    required = required_fields(schema)
    payload_name = "request" if variant_index % 2 == 0 else "parameters"
    nested_schema = {
        "type": "object",
        "properties": {
            payload_name: {
                "type": "object",
                "properties": deepcopy(props),
                "required": required,
            }
        },
        "required": [payload_name],
    }
    arg_map = {f"{payload_name}.{field}": field for field in props}
    return nested_schema, AdapterSpec(arg_map=arg_map, nesting={"payload": payload_name})


def _schema_with_single_object(schema: dict[str, Any]) -> tuple[dict[str, Any], AdapterSpec]:
    props = schema_properties(schema)
    required = required_fields(schema)
    bundle_name = "tool_request"
    visible_schema = {
        "type": "object",
        "properties": {
            bundle_name: {
                "type": "object",
                "description": "All parameters for the operation.",
                "properties": deepcopy(props),
                "required": required,
            }
        },
        "required": [bundle_name],
    }
    return visible_schema, AdapterSpec(
        arg_map={f"{bundle_name}.{field}": field for field in props},
        nesting={"payload": bundle_name},
    )


def generate_valid_variant_specs(base: ToolRecord, max_candidates: int = 3) -> list[tuple[str, dict[str, Any], AdapterSpec]]:
    candidates: list[tuple[str, dict[str, Any], AdapterSpec]] = []
    if not schema_properties(base.input_schema):
        return candidates

    schema, adapter = _schema_with_renamed_fields(base.input_schema, 0)
    candidates.append(("mild", schema, adapter))

    if len(candidates) < max_candidates:
        schema, adapter = _schema_with_nested_payload(base.input_schema, 1)
        candidates.append(("medium", schema, adapter))

    if len(candidates) < max_candidates:
        schema, adapter = _schema_with_single_object(base.input_schema)
        candidates.append(("aggressive", schema, adapter))

    return candidates[:max_candidates]


def adapt_arguments(args: dict[str, Any], adapter: AdapterSpec | None) -> dict[str, Any]:
    if not adapter:
        return dict(args)
    out = dict(adapter.defaults)
    for visible, target in adapter.arg_map.items():
        value = _get_path(args, visible)
        if value is not _MISSING:
            out[target] = _cast_value(value, adapter.type_casts.get(target))
    for target, mapping in adapter.enum_map.items():
        if target in out and out[target] in mapping:
            out[target] = mapping[out[target]]
    return out


def visible_arguments_from_base(base_args: dict[str, Any], adapter: AdapterSpec | None) -> dict[str, Any]:
    """Create caller-facing variant args that adapt back to a known-good base fixture."""
    if not adapter:
        return dict(base_args)
    visible_args: dict[str, Any] = {}
    for visible_path, base_field in adapter.arg_map.items():
        if base_field in base_args:
            _set_path(visible_args, visible_path, base_args[base_field])
    return visible_args


_MISSING = object()


def _get_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    current: dict[str, Any] = data
    parts = path.split(".")
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            nested = {}
            current[part] = nested
        current = nested
    current[parts[-1]] = value


def _cast_value(value: Any, cast: str | None) -> Any:
    if cast is None:
        return value
    if cast == "str":
        return str(value)
    if cast == "int":
        return int(value)
    if cast == "float":
        return float(value)
    if cast == "bool":
        return bool(value)
    return value
