"""Assembles the whole marketplace by combining the other modules.

`build_pool` puts together the ~482 listings: the 50 base gold tools, a sample of
reworded variants (catalog -> adapters -> descriptions), the broken variants spread
across base tools (faults + descriptions), and the background distractors. It's pure
and deterministic given a seed, so the pool rebuilds byte-for-byte. `write_pool`
saves the JSONL files. This is the single entry point the build script calls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .adapters import generate_valid_variant_specs
from .catalog import load_mcp_tools, select_background_tools, select_base_tools
from .descriptions import (
    description_for_corrupted_variant,
    description_for_valid_variant,
    fallback_description,
)
from .faults import expanded_failure_labels, make_fault_spec
from .io import write_jsonl
from .models import AdapterTestRecord, ListingRecord, ToolRecord


@dataclass(frozen=True)
class PoolBuildConfig:
    catalog_path: Path
    output_dir: Path
    base_count: int = 50
    valid_variant_count: int = 75
    corrupted_variant_count: int = 150
    background_count: int = 207
    seed: int = 13
    max_valid_candidates_per_base: int = 3
    exclude_tool_ids: set[str] | None = None
    include_tool_ids: list[str] | None = None


@dataclass
class PoolBuildResult:
    base_tools: list[ToolRecord]
    listings: list[ListingRecord]
    variant_candidates: list[ListingRecord]
    adapter_tests: list[AdapterTestRecord]


def build_pool(config: PoolBuildConfig) -> PoolBuildResult:
    rng = random.Random(config.seed)
    tools = load_mcp_tools(config.catalog_path)
    base_tools = select_base_tools(
        tools, config.base_count, config.exclude_tool_ids, config.include_tool_ids
    )
    base_ids = {tool.tool_id for tool in base_tools}
    background = select_background_tools(tools, base_ids, config.background_count)

    listings: list[ListingRecord] = []
    variant_candidates: list[ListingRecord] = []
    adapter_tests: list[AdapterTestRecord] = []

    for base in base_tools:
        listings.append(base_listing(base))

    valid_candidates = []
    for base_index, base in enumerate(base_tools):
        for candidate_index, (level, schema, adapter) in enumerate(
            generate_valid_variant_specs(base, config.max_valid_candidates_per_base)
        ):
            listing = ListingRecord(
                listing_id=f"{base.tool_id}.valid.{level}.{candidate_index + 1}",
                base_tool_id=base.tool_id,
                server=base.server,
                category=base.category,
                variant_type="valid_schema_variant",
                description=description_for_valid_variant(base, level, base_index + candidate_index),
                input_schema=schema,
                tool_name=f"{base.tool_name}_{level}_variant_{candidate_index + 1}",
                adapter=adapter,
                metadata={"variant_level": level, "generated_by": "deterministic_scaffold"},
            )
            valid_candidates.append(listing)
            variant_candidates.append(listing)
            adapter_tests.append(
                AdapterTestRecord(
                    listing_id=listing.listing_id,
                    base_tool_id=base.tool_id,
                    status="metadata_only",
                    details="Adapter spec compiled; live fixture validation not run in scaffold build.",
                )
            )

    listings.extend(_sample_valid_variants(valid_candidates, config.valid_variant_count, rng))

    corrupted = _build_corrupted_variants(base_tools, config, rng)
    listings.extend(corrupted)
    variant_candidates.extend(corrupted)

    for tool in background:
        listings.append(background_listing(tool))

    return PoolBuildResult(
        base_tools=base_tools,
        listings=listings,
        variant_candidates=variant_candidates,
        adapter_tests=adapter_tests,
    )


def write_pool(result: PoolBuildResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "base_tools.jsonl", (tool.to_json() for tool in result.base_tools))
    write_jsonl(output_dir / "listings.jsonl", (listing.to_json() for listing in result.listings))
    write_jsonl(
        output_dir / "variant_candidates.jsonl",
        (listing.to_json() for listing in result.variant_candidates),
    )
    write_jsonl(
        output_dir / "adapter_tests.jsonl",
        (test.to_json() for test in result.adapter_tests),
    )
    write_jsonl(output_dir / "smoke_tests.jsonl", [])


def base_listing(base: ToolRecord) -> ListingRecord:
    return ListingRecord(
        listing_id=base.tool_id,
        base_tool_id=base.tool_id,
        server=base.server,
        category=base.category,
        variant_type="base_gold",
        description=base.description,
        input_schema=base.input_schema,
        tool_name=base.tool_name,
        metadata={"source": "mcp_bench"},
    )


def background_listing(tool: ToolRecord) -> ListingRecord:
    return ListingRecord(
        listing_id=tool.tool_id,
        base_tool_id=None,
        server=tool.server,
        category=tool.category,
        variant_type="background_distractor",
        description=tool.description or fallback_description(tool.server, tool.tool_name),
        input_schema=tool.input_schema,
        tool_name=tool.tool_name,
        metadata={"source": "mcp_bench"},
    )


def _sample_valid_variants(
    candidates: list[ListingRecord],
    target_count: int,
    rng: random.Random,
) -> list[ListingRecord]:
    if target_count >= len(candidates):
        return candidates

    by_base: dict[str, list[ListingRecord]] = {}
    for candidate in candidates:
        by_base.setdefault(candidate.base_tool_id or "", []).append(candidate)

    chosen: list[ListingRecord] = []
    # First pass: one variant per base when possible.
    for base_id in sorted(by_base):
        if by_base[base_id] and len(chosen) < target_count:
            chosen.append(by_base[base_id].pop(0))

    remaining = [candidate for bucket in by_base.values() for candidate in bucket]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, target_count - len(chosen))])
    return chosen[:target_count]


def _build_corrupted_variants(
    base_tools: list[ToolRecord],
    config: PoolBuildConfig,
    rng: random.Random,
) -> list[ListingRecord]:
    labels = expanded_failure_labels(config.corrupted_variant_count, config.seed)
    base_schedule = _corruption_schedule(base_tools, config.corrupted_variant_count, rng)
    listings: list[ListingRecord] = []
    label_index = 0
    per_base_counts: dict[str, int] = {}

    for base in base_schedule:
        label = labels[label_index]
        label_index += 1
        per_base_counts[base.tool_id] = per_base_counts.get(base.tool_id, 0) + 1
        ordinal = per_base_counts[base.tool_id]
        seed = config.seed * 100000 + label_index
        fault_spec = make_fault_spec(label, seed)
        listings.append(
            ListingRecord(
                listing_id=f"{base.tool_id}.{label}.{ordinal}",
                base_tool_id=base.tool_id,
                server=base.server,
                category=base.category,
                variant_type=label,  # type: ignore[arg-type]
                description=description_for_corrupted_variant(base, label_index),
                input_schema=base.input_schema,
                tool_name=f"{base.tool_name}_{label.replace('corrupted_', '')}_{ordinal}",
                fault_spec=fault_spec,
                p_fail=fault_spec.p_fail,
                metadata={"generated_by": "deterministic_fault_wrapper"},
            )
        )
    return listings


def _corruption_schedule(
    base_tools: list[ToolRecord],
    total: int,
    rng: random.Random,
) -> list[ToolRecord]:
    if not base_tools:
        return []
    shuffled = list(base_tools)
    rng.shuffle(shuffled)
    counts = [1] * min(15, len(shuffled))
    counts.extend([3] * min(20, max(0, len(shuffled) - len(counts))))
    counts.extend([5] * min(15, max(0, len(shuffled) - len(counts))))
    while len(counts) < len(shuffled):
        counts.append(1)

    schedule: list[ToolRecord] = []
    for base, count in zip(shuffled, counts):
        schedule.extend([base] * count)

    while len(schedule) < total:
        schedule.append(rng.choice(base_tools))
    rng.shuffle(schedule)
    return schedule[:total]
