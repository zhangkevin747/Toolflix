#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.simulated_online import ToolBanditPolicy, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a ToolBandit event log and inspect learned weights.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--listings", type=Path, default=ROOT / "data/pool/listings.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--discount-gamma", type=float, default=0.985)
    parser.add_argument("--exploration-weight", type=float, default=0.45)
    parser.add_argument("--neural-weight", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = load_jsonl(args.events)
    listings = load_jsonl(args.listings)
    policy = ToolBanditPolicy(
        listings=listings,
        discount_gamma=args.discount_gamma,
        exploration_weight=args.exploration_weight,
        neural_weight=args.neural_weight,
        learning_rate=args.learning_rate,
    )

    losses: list[float] = []
    for event in events:
        listing_id = event.get("selected_listing_id")
        if not listing_id:
            continue
        selected = next((row for row in event.get("slate") or [] if row["listing_id"] == listing_id), None)
        retrieval_score = float(selected["retrieval_score"]) if selected else 0.0
        loss = policy.update(
            model_id=event["model_id"],
            listing_id=listing_id,
            query=event["marketplace_query"],
            retrieval_score=retrieval_score,
            reward=float(event["reward"]),
            round_index=int(event["round"]),
        )
        if loss is not None:
            losses.append(loss)

    snapshot = snapshot_policy(policy, events, losses)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


def snapshot_policy(policy: ToolBanditPolicy, events: list[dict[str, Any]], losses: list[float]) -> dict[str, Any]:
    net = policy.net
    model_embeddings = net.model_embeddings.weight.detach()
    tool_norms = torch.linalg.vector_norm(net.tool_embeddings.weight.detach(), dim=1)

    tool_stats = [
        {
            "listing_id": listing_id,
            "discounted_pulls": round(stats.pulls, 6),
            "discounted_reward": round(stats.reward, 6),
            "posterior_mean": round(stats.mean, 6),
        }
        for listing_id, stats in policy.listing_stats.items()
        if stats.pulls > 0
    ]
    tool_stats.sort(key=lambda row: row["discounted_pulls"], reverse=True)

    model_tool_stats = [
        {
            "model_id": model_id,
            "listing_id": listing_id,
            "discounted_pulls": round(stats.pulls, 6),
            "discounted_reward": round(stats.reward, 6),
            "posterior_mean": round(stats.mean, 6),
        }
        for (model_id, listing_id), stats in policy.model_listing_stats.items()
        if stats.pulls > 0
    ]
    model_tool_stats.sort(key=lambda row: (row["posterior_mean"], -row["discounted_pulls"]))

    return {
        "events_replayed": len(events),
        "updates": len(losses),
        "last_loss": round(losses[-1], 6) if losses else None,
        "toolbandit_method": {
            "score": "sigmoid(f_theta([query_embedding; model_embedding; tool_embedding])) + beta * sqrt(log(N_t + 1) / (n_t(tool) + 1))",
            "retrieval_similarity": "stage_1_only",
            "ucb_beta": policy.exploration_weight,
            "discount_gamma": policy.discount_gamma,
            "replay_window": policy.replay_window,
            "replay_batch_size": policy.replay_batch_size,
        },
        "mlp_layer_norms": [
            round(float(torch.linalg.vector_norm(layer.weight.detach()).item()), 6)
            for layer in net.mlp
            if hasattr(layer, "weight")
        ],
        "mlp_layer_biases": [
            round(float(torch.linalg.vector_norm(layer.bias.detach()).item()), 6)
            for layer in net.mlp
            if hasattr(layer, "bias")
        ],
        "tool_embedding_norm_mean": round(float(tool_norms.mean().item()), 6),
        "tool_embedding_norm_max": round(float(tool_norms.max().item()), 6),
        "model_embedding_norms": {
            model_id: round(float(torch.linalg.vector_norm(model_embeddings[index]).item()), 6)
            for model_id, index in policy.model_to_idx.items()
        },
        "model_embedding_means": {
            model_id: round(float(model_embeddings[index].mean().item()), 6)
            for model_id, index in policy.model_to_idx.items()
        },
        "top_discounted_tool_stats": tool_stats[:20],
        "lowest_model_tool_means": model_tool_stats[:20],
    }


if __name__ == "__main__":
    raise SystemExit(main())
