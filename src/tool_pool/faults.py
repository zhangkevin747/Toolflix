"""Broken tool copies ("corrupted variants") and how they fail.

Real marketplaces have unreliable tools, so we manufacture them. The failure mix
follows a published MCP reliability study: mostly schema mismatches and timeouts,
some auth/quota and upstream errors, a few protocol bugs. Each broken copy gets a
FaultSpec with a failure type and a `p_fail` probability — many fail only sometimes,
like real flaky servers. `should_fail` decides, deterministically from a seed, whether
a given call fails, so runs are reproducible.
"""

from __future__ import annotations

import random
from typing import Any

from .models import FaultSpec


FAILURE_COUNTS = {
    "corrupted_schema_mismatch": 57,
    "corrupted_timeout": 36,
    "corrupted_auth_quota": 29,
    "corrupted_upstream_api": 18,
    "corrupted_protocol_bug": 10,
}

FAILURE_SHARES = {
    "corrupted_schema_mismatch": 0.38,
    "corrupted_timeout": 0.24,
    "corrupted_auth_quota": 0.19,
    "corrupted_upstream_api": 0.12,
    "corrupted_protocol_bug": 0.07,
}


P_FAIL_RANGES = {
    "corrupted_schema_mismatch": (1.0, 1.0),
    "corrupted_timeout": (0.25, 0.5),
    "corrupted_auth_quota": (0.5, 0.75),
    "corrupted_upstream_api": (0.25, 0.5),
    "corrupted_protocol_bug": (1.0, 1.0),
}


def expanded_failure_labels(total: int, seed: int) -> list[str]:
    if total == sum(FAILURE_COUNTS.values()):
        counts = dict(FAILURE_COUNTS)
    else:
        counts = {
            label: int(total * share)
            for label, share in FAILURE_SHARES.items()
        }
        shortfall = total - sum(counts.values())
        # Add remainder to the largest fractional parts.
        remainders = sorted(
            FAILURE_SHARES,
            key=lambda label: (total * FAILURE_SHARES[label]) % 1,
            reverse=True,
        )
        for label in remainders[:shortfall]:
            counts[label] += 1

    labels: list[str] = []
    for label, count in counts.items():
        labels.extend([label] * count)
    rng = random.Random(seed)
    rng.shuffle(labels)
    return labels


def make_fault_spec(label: str, seed: int) -> FaultSpec:
    low, high = P_FAIL_RANGES[label]
    rng = random.Random(seed)
    p_fail = low if low == high else round(rng.uniform(low, high), 2)
    return FaultSpec(
        failure_type=label.replace("corrupted_", ""),
        p_fail=p_fail,
        seed=seed,
        failure_payload=default_failure_payload(label),
    )


def default_failure_payload(label: str) -> dict[str, Any]:
    if label == "corrupted_schema_mismatch":
        return {
            "error": "schema_mismatch",
            "response": {"status": "ok", "data": ["truncated"]},
            "note": "Response intentionally violates declared output contract.",
        }
    if label == "corrupted_timeout":
        return {"error": "timeout", "message": "Tool call exceeded configured timeout."}
    if label == "corrupted_auth_quota":
        return {"error": "auth_or_quota", "status_code": 429, "message": "Quota exceeded."}
    if label == "corrupted_upstream_api":
        return {"error": "upstream_api", "status_code": 503, "message": "Upstream unavailable."}
    if label == "corrupted_protocol_bug":
        return {"jsonrpc": "2.0", "malformed": True}
    return {"error": "unknown"}


def should_fail(spec: FaultSpec, attempt: int = 0) -> bool:
    rng = random.Random(spec.seed + attempt)
    return rng.random() < spec.p_fail
