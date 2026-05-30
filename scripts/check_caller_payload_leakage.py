#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.live_online import ModelToolSelector, detect_payload_leakage
from training.simulated_online import TfidfRetriever, ToolBanditPolicy, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail if caller-visible ToolBandit payloads leak synthetic variant labels.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks/tasks_full_ready.jsonl")
    parser.add_argument("--listings", type=Path, default=ROOT / "data/pool/listings.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=80)
    parser.add_argument("--slate-size", type=int, default=5)
    parser.add_argument("--sample-out", type=Path, default=ROOT / "data/runs/payload_leakage_sample.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = load_jsonl(args.tasks)
    listings = load_jsonl(args.listings)
    listings_by_id = {listing["listing_id"]: listing for listing in listings}
    retriever = TfidfRetriever(listings)
    policy = ToolBanditPolicy(listings=listings)
    selector = ModelToolSelector()

    checked = 0
    failures = []
    sample = None
    for round_index, task in enumerate(tasks[: args.limit], start=1):
        candidates = retriever.search(task["marketplace_query"], limit=args.candidate_count)
        slate = policy.rerank(
            model_id="gpt-5.4-nano",
            query=task["marketplace_query"],
            candidates=candidates,
            slate_size=args.slate_size,
            round_index=round_index,
        )
        options, alias_map = selector.build_options(slate, listings_by_id)
        payload = {
            "user_task": task["user_task"],
            "marketplace_query": task["marketplace_query"],
            "tool_options": options,
            "response_schema": {"tool_alias": "one of the option tool_alias values", "arguments": {}},
        }
        leakage = detect_payload_leakage(payload)
        checked += 1
        if sample is None:
            sample = {"payload": payload, "internal_alias_map": alias_map}
        if leakage:
            failures.append({"task_id": task["task_id"], "leakage": leakage, "payload": payload})

    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    args.sample_out.write_text(json.dumps(sample or {}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "checked": checked,
        "failed": len(failures),
        "sample_out": str(args.sample_out.resolve()),
        "failures": failures[:10],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
