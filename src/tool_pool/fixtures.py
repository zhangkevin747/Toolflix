"""Known-good arguments for each base tool, so we can actually call it.

A tool is useless to us if we can't produce one valid call. This module reviews each
base tool (dropping ones that look stateful or unfixturable) and builds `fixture_args`
for it — preferring hand-written values for known tools (the big `explicit_fixtures`
table) and falling back to type-based guesses. It also drafts a suggested task and
marketplace query. Output feeds the live-validation step and task generation.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .catalog import required_fields, schema_properties
from .io import write_jsonl
from .models import ToolRecord


DROP_PATTERNS = [
    "previously stored",
    "stored tensor",
    "stored tensors",
    "stored matrix",
    "stored matrices",
    "stored square matrix",
    "stored vector",
    "tensor store",
    "stored numpy",
    "in-memory tensor",
    "send ",
    "delete ",
    "checkout",
    "place order",
]

WEAK_PATTERNS = [
    "current weather",
    "live",
    "hot threads",
    "candlestick",
    "available dates",
    "current alerts",
    "quota",
]


@dataclass
class FixtureRecord:
    tool_id: str
    server: str
    tool_name: str
    category: str
    review_status: str
    review_reason: str
    fixture_args: dict[str, Any]
    fixture_confidence: str
    suggested_user_task: str
    suggested_marketplace_query: str
    expected_answer_source: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def review_and_fixture_tool(tool: ToolRecord) -> FixtureRecord:
    blob = f"{tool.tool_name} {tool.description}".lower()
    status = "keep"
    reason = "single-call candidate"
    confidence = "medium"

    if any(pattern in blob for pattern in DROP_PATTERNS):
        status = "drop"
        reason = "appears stateful, mutating, or requires prior setup"
        confidence = "low"
    elif any(pattern in blob for pattern in WEAK_PATTERNS):
        status = "weak_keep"
        reason = "works as a tool-output sufficiency task, but exact ground truth may be volatile"
        confidence = "medium"
    elif not schema_properties(tool.input_schema):
        status = "weak_keep"
        reason = "no visible schema properties; may be usable but fixture is limited"
        confidence = "low"

    args = fixture_args_for(tool)
    if required_fields(tool.input_schema) and not args:
        status = "drop"
        reason = "required fields exist but no safe automatic fixture was found"
        confidence = "low"

    return FixtureRecord(
        tool_id=tool.tool_id,
        server=tool.server,
        tool_name=tool.tool_name,
        category=tool.category,
        review_status=status,
        review_reason=reason,
        fixture_args=args,
        fixture_confidence=confidence,
        suggested_user_task=suggested_user_task(tool, args),
        suggested_marketplace_query=suggested_marketplace_query(tool),
        expected_answer_source="tool_output",
    )


def fixture_args_for(tool: ToolRecord) -> dict[str, Any]:
    tool_id = tool.tool_id
    props = schema_properties(tool.input_schema)
    args: dict[str, Any] = {}
    required = set(required_fields(tool.input_schema))

    explicit = explicit_fixtures(tool_id)
    args.update(explicit)

    for field, spec in props.items():
        if field in args:
            continue
        if explicit and field not in required:
            continue
        if required and field not in required:
            continue
        value = value_for_field(field, spec, tool)
        if value is not None:
            args[field] = value
    return args


def explicit_fixtures(tool_id: str) -> dict[str, Any]:
    fixtures: dict[str, dict[str, Any]] = {
        "bibliomantic.i_ching_divination": {"query": "Should I focus on careful planning this week?"},
        "bibliomantic.bibliomantic_consultation": {"query": "What guidance does the I Ching give for a cautious decision?"},
        "bibliomantic.get_hexagram_details": {"hexagram_number": 1},
        "dex_paprika.search": {"query": "USDC"},
        "dex_paprika.gettokendetails": {"network": "ethereum", "tokenAddress": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"},
        "fruityvice.get_fruit_nutrition": {"fruit_name": "apple"},
        "hugging_face.get_space_info": {"space_id": "openai/whisper"},
        "hugging_face.get_paper_info": {"arxiv_id": "1706.03762"},
        "hugging_face.get_model_info": {"model_id": "gpt2"},
        "hugging_face.get_dataset_info": {"dataset_id": "imdb"},
        "hugging_face.get_collection_info": {"namespace": "openai", "collection_id": "whisper"},
        "google_maps.maps_geocode": {"address": "1600 Amphitheatre Parkway, Mountain View, CA"},
        "google_maps.maps_elevation": {"locations": [{"latitude": 37.422, "longitude": -122.084}]},
        "google_maps.get_place_details": {"placeId": "ChIJ2eUgeAK6j4ARbn5u_wAGqWA"},
        "nasa_data.get_mars_rover_manifest": {"rover_name": "Curiosity"},
        "nasa_data.get_epic_imagery": {"collection": "natural"},
        "nasa_data.get_epic_dates": {"collection": "natural"},
        "nasa_data.get_geomagnetic_storm": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        "nasa_data.get_asteroid_lookup": {"asteroid_id": "2000433"},
        "medical_calculator.maintenance_fluids": {"weight_kg": 70},
        "medical_calculator.map_calculator": {"sbp": 120, "dbp": 80},
        "medical_calculator.homa_ir": {"fasting_insulin": 10, "fasting_glucose": 90},
        "medical_calculator.corrected_sodium": {"measured_sodium": 130, "serum_glucose": 400},
        "medical_calculator.steroid_conversion": {
            "from_dose_mg": 5,
            "from_steroid": "prednisone",
            "to_steroid": "hydrocortisone",
        },
        "national_parks.getparkdetails": {"parkCode": "yose"},
        "national_parks.getvisitorcenters": {"parkCode": "yose", "limit": 5},
        "national_parks.getcampgrounds": {"parkCode": "yose", "limit": 5},
        "national_parks.getalerts": {"parkCode": "yose", "limit": 5},
        "nixos.nixos_stats": {"channel": "unstable"},
        "nixos.home_manager_options_by_prefix": {"option_prefix": "programs.git"},
        "nixos.home_manager_info": {"name": "programs.git.enable"},
        "nixos.darwin_options_by_prefix": {"option_prefix": "system.defaults"},
        "wikipedia.get_sections": {"title": "George Washington"},
        "wikipedia.get_links": {"title": "George Washington"},
        "wikipedia.get_article": {"title": "George Washington"},
        "biomcp.trial_references_getter": {"nct_id": "NCT04280705"},
        "biomcp.trial_protocol_getter": {"nct_id": "NCT04280705"},
        "biomcp.trial_outcomes_getter": {"nct_id": "NCT04280705"},
        "biomcp.trial_locations_getter": {"nct_id": "NCT04280705"},
        "weather_data.get_live_temp": {"city": "San Francisco"},
        "weather_data.get_current_weather_tool": {"city": "San Francisco"},
        "weather_data.search_locations_tool": {"query": "San Francisco"},
        "reddit.fetch_reddit_hot_threads": {"subreddit": "science", "limit": 5},
        "reddit.fetch_reddit_post_content": {"post_id": "dummy"},
        "okx_exchange.get_candlesticks": {"instrument": "BTC-USDT", "bar": "1D", "limit": 5},
        "okx_exchange.get_price": {"instrument": "BTC-USDT"},
        "time_mcp.get_current_time": {"timezone": "America/Los_Angeles"},
        "unit_converter.list_supported_units": {"unit_type": "length"},
        "scientific_computing.curl": {"f_str": "[y, -x, 0]", "point": [1, 2, 3]},
        # --- Reclaimed tools (verified live 2026-05-30 with these real inputs) ---
        "biomcp.openfda_adverse_getter": {"report_id": "10003914"},
        "dex_paprika.getnetworks": {},
        "dex_paprika.getstats": {},
        "dex_paprika.getnetworkdexes": {"network": "ethereum"},
        "dex_paprika.getnetworkpools": {"network": "ethereum"},
        "dex_paprika.getdexpools": {"network": "ethereum", "dex": "uniswap_v3"},
        "dex_paprika.gettokenpools": {"network": "ethereum", "tokenAddress": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"},
        "dex_paprika.getpooldetails": {"network": "ethereum", "poolAddress": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"},
        "dex_paprika.getpooltransactions": {"network": "ethereum", "poolAddress": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"},
        "dex_paprika.getpoolohlcv": {"network": "ethereum", "poolAddress": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", "start": "2024-01-01"},
        "google_maps.maps_distance_matrix": {"origins": ["San Francisco, CA"], "destinations": ["Los Angeles, CA"]},
        "google_maps.maps_directions": {"origin": "San Francisco, CA", "destination": "Los Angeles, CA"},
        "google_maps.maps_reverse_geocode": {"latitude": 37.422, "longitude": -122.084},
        "google_maps.search_nearby": {"center": {"value": "San Francisco, CA"}, "keyword": "coffee"},
        "nixos.darwin_info": {"name": "system.defaults.dock.autohide"},
        "scientific_computing.create_tensor": {"shape": [2, 2], "values": [1, 2, 3, 4], "name": "A"},
        "scientific_computing.divergence": {"f_str": "[x, y, z]", "point": [1, 2, 3]},
    }
    return dict(fixtures.get(tool_id, {}))


def value_for_field(field: str, spec: dict[str, Any], tool: ToolRecord) -> Any:
    lower = field.lower()
    if "enum" in spec and spec["enum"]:
        return spec["enum"][0]
    if lower in {"query", "q"}:
        return "George Washington"
    if "city" in lower or "location" in lower:
        return "San Francisco"
    if "title" in lower:
        return "George Washington"
    if "name" in lower:
        return "apple"
    if "limit" in lower or lower in {"start"}:
        return 5 if "limit" in lower else 0
    if "date" in lower:
        return "2024-01-01"
    if "id" in lower:
        return "example"
    if "code" in lower:
        return "yose"
    typ = spec.get("type")
    if typ == "string":
        return "example"
    if typ == "number":
        return 1.0
    if typ == "integer":
        return 1
    if typ == "boolean":
        return True
    if typ == "array":
        return []
    if typ == "object":
        return {}
    return None


def suggested_user_task(tool: ToolRecord, args: dict[str, Any]) -> str:
    if tool.tool_id.startswith("wikipedia."):
        return f"Use the available information to answer a simple question about {args.get('title', 'the topic')}."
    if tool.category == "medical":
        return "Compute the requested medical calculator result from the provided numeric inputs."
    if tool.category == "maps":
        return "Find the requested geographic information for the provided place."
    if tool.category == "science":
        return "Retrieve the requested scientific or clinical-trial information."
    if tool.category == "finance":
        return "Retrieve the requested market or asset information."
    return f"Use the {tool.tool_name} capability to answer a simple one-step request."


def suggested_marketplace_query(tool: ToolRecord) -> str:
    words = tool.tool_name.replace("_", " ").replace("-", " ")
    return f"{tool.category} {words}".strip()


def write_review_csv(path: Path, rows: list[FixtureRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tool_id",
        "server",
        "tool_name",
        "category",
        "review_status",
        "review_reason",
        "fixture_confidence",
        "fixture_args",
        "suggested_marketplace_query",
        "suggested_user_task",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.to_json()
            data["fixture_args"] = repr(row.fixture_args)
            writer.writerow({key: data.get(key, "") for key in fieldnames})


def write_fixture_jsonl(path: Path, rows: list[FixtureRecord]) -> None:
    write_jsonl(path, (row.to_json() for row in rows))
