from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .models import ListingRecord


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    counts: dict[str, int]


def validate_listings(
    listings: Iterable[ListingRecord],
    expected_counts: dict[str, int] | None = None,
) -> ValidationResult:
    rows = list(listings)
    errors: list[str] = []
    warnings: list[str] = []
    counts = Counter(row.variant_type for row in rows)

    listing_ids = [row.listing_id for row in rows]
    duplicate_ids = [item for item, count in Counter(listing_ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"Duplicate listing_id values: {duplicate_ids[:10]}")

    descriptions = [row.description.strip().lower() for row in rows if row.description.strip()]
    duplicate_descriptions = [item for item, count in Counter(descriptions).items() if count > 1]
    if duplicate_descriptions:
        warnings.append(f"Duplicate descriptions found: {len(duplicate_descriptions)}")

    for row in rows:
        if not row.description and row.variant_type != "background_distractor":
            errors.append(f"{row.listing_id} has no description")
        if row.variant_type == "valid_schema_variant" and row.adapter is None:
            errors.append(f"{row.listing_id} is valid variant without adapter")
        if row.variant_type.startswith("corrupted_") and row.fault_spec is None:
            errors.append(f"{row.listing_id} is corrupted variant without fault_spec")
        if row.variant_type == "base_gold" and row.base_tool_id != row.listing_id:
            errors.append(f"{row.listing_id} base_gold should point to itself")

    if expected_counts:
        for variant_type, expected in expected_counts.items():
            actual = counts.get(variant_type, 0)
            if actual != expected:
                errors.append(f"{variant_type}: expected {expected}, got {actual}")

    return ValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        counts=dict(counts),
    )

