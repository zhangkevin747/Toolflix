"""The training loop. This is the whole experiment in one place.

One loop serves both modes. The only things that change are:
  - the picker:  how a tool is chosen from the slate (policy_pick vs LLMPicker)
  - the reward:  how we score the result (metadata_reward vs JudgeReward)

Everything else (retrieve -> rerank -> execute -> learn -> log) is identical, which
is why the old code's two 600-line copies collapse into this.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable

from .bandit import ScoredTool, ToolBandit
from .caller import Choice
from .data import CLEAN_VARIANTS, MODELS, family_of, load_jsonl
from .marketplace import Marketplace, load_cached_outputs
from .retriever import Retriever

Picker = Callable[[str, dict, list[ScoredTool], dict], Choice]
RewardFn = Callable[[dict, dict, Any], tuple[float, dict]]


def run(
    *,
    tasks_path: Path,
    listings_path: Path,
    cached_outputs_path: Path,
    out_dir: Path,
    picker: Picker,
    reward_fn: RewardFn,
    rounds: int | None = None,
    candidate_count: int = 80,
    slate_size: int = 5,
    seed: int = 13,
    execution: str = "cached",
    mcp_timeout: float = 45.0,
    mcp_cache: bool = True,
) -> dict[str, Any]:
    # ---- load everything -------------------------------------------------
    tasks = load_jsonl(tasks_path)
    listings = load_jsonl(listings_path)
    listings_by_id = {l["listing_id"]: l for l in listings}
    cached_outputs = load_cached_outputs(cached_outputs_path)

    random.Random(seed).shuffle(tasks)
    if rounds is not None:
        tasks = tasks[:rounds]

    retriever = Retriever(listings)
    bandit = ToolBandit(listings)
    # "cached" replays recorded outputs (fast, free); "live" calls the real MCP servers.
    if execution == "live":
        from .live_marketplace import LiveMarketplace
        marketplace = LiveMarketplace(listings_by_id, timeout=mcp_timeout, use_cache=mcp_cache)
    else:
        marketplace = Marketplace(listings_by_id, cached_outputs)

    out_dir.mkdir(parents=True, exist_ok=True)
    events = (out_dir / "events.jsonl").open("w", encoding="utf-8")
    curve = csv.writer((out_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8"))
    curve.writerow(["round", "reward", "mean_reward", "rolling_50_mean_reward"])

    rewards: list[float] = []
    counts = {"clean_gold": 0, "corrupted": 0, "wrong_family": 0, "selection_errors": 0}

    # ---- the loop --------------------------------------------------------
    # try/finally so live MCP subprocesses are always released, even on a crash.
    try:
        for round_index, task in enumerate(tasks, start=1):
            model_id = MODELS[(round_index - 1) % len(MODELS)]
            task["_caller_model"] = model_id  # used by the "self" LLM judge
            query = task["marketplace_query"]

            candidates = retriever.search(query, limit=candidate_count)        # stage 1
            slate = bandit.rerank(model_id, query, candidates, slate_size, round_index)  # stage 2
            choice = picker(model_id, task, slate, listings_by_id)             # stage 3

            # If a live model failed to choose, fall back to the bandit's top tool so
            # the round still produces a learning signal.
            if choice.listing_id is None and slate:
                counts["selection_errors"] += 1
                choice = Choice(listing_id=slate[0].listing_id, alias="tool_1", error=choice.error)

            reward, info, outcome, variant = 0.0, {}, None, None
            loss = None
            if choice.listing_id is not None:
                listing = listings_by_id[choice.listing_id]
                variant = listing["variant_type"]
                outcome = marketplace.execute(choice.listing_id, choice.arguments, attempt=round_index)  # stage 4
                reward, info = reward_fn(task, listing, outcome)               # stage 5
                loss = bandit.learn(model_id, choice.listing_id, query, reward, round_index)  # stage 6
                _tally(counts, listing, task, variant)

            rewards.append(reward)
            window = rewards[-50:]
            event = {
                "round": round_index,
                "task_id": task["task_id"],
                "model_id": model_id,
                "gold_base_tool_id": task["gold_base_tool_id"],
                "marketplace_query": query,
                "slate": [s.listing_id for s in slate],
                "selected_listing_id": choice.listing_id,
                "selected_variant_type": variant,
                "selection_error": choice.error,
                "execution_ok": outcome.ok if outcome else None,
                "failure_type": outcome.failure_type if outcome else None,
                "reward": reward,
                "reward_info": info,
                "policy_loss": round(loss, 6) if loss is not None else None,
                "mean_reward": round(sum(rewards) / len(rewards), 6),
                "rolling_50_mean_reward": round(sum(window) / len(window), 6),
            }
            events.write(json.dumps(event, ensure_ascii=False) + "\n")
            events.flush()
            curve.writerow([round_index, reward, event["mean_reward"], event["rolling_50_mean_reward"]])
            print(f"[{round_index}/{len(tasks)}] model={model_id} reward={reward:.0f} "
                  f"mean={event['mean_reward']:.3f} picked={choice.listing_id}", flush=True)
    finally:
        events.close()
        marketplace.close()

    # ---- summary ---------------------------------------------------------
    summary = {
        "rounds": len(rewards),
        "seed": seed,
        "execution": execution,
        "candidate_count": candidate_count,
        "slate_size": slate_size,
        "models": MODELS,
        "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "rolling_50_mean_reward": round(sum(rewards[-50:]) / min(len(rewards), 50), 6) if rewards else 0.0,
        **counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _tally(counts: dict[str, int], listing: dict, task: dict, variant: str) -> None:
    """Bookkeeping: was the pick the right tool, a broken copy, or the wrong tool?"""
    if family_of(listing) != task["gold_base_tool_id"]:
        counts["wrong_family"] += 1
    elif variant in CLEAN_VARIANTS:
        counts["clean_gold"] += 1
    elif str(variant).startswith("corrupted_"):
        counts["corrupted"] += 1
