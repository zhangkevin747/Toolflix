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

from training.live_online import load_dotenv, run_live_online_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1: online ToolBandit training with live LLM model selection.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks/tasks_full_ready.jsonl")
    parser.add_argument("--listings", type=Path, default=ROOT / "data/pool/listings.jsonl")
    parser.add_argument("--live-outputs", type=Path, default=ROOT / "data/pool/live_base_validation.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/runs/experiment1_online")
    parser.add_argument("--rounds", type=int, default=None, help="Maximum unique tasks to consume. Defaults to all tasks.")
    parser.add_argument("--candidate-count", type=int, default=80)
    parser.add_argument("--slate-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--judge-model", default="self", help="Use 'self' for each caller to judge itself, or name one judge model.")
    parser.add_argument("--reward-mode", choices=["llm", "metadata"], default="llm")
    parser.add_argument("--discount-gamma", type=float, default=0.985)
    parser.add_argument("--exploration-weight", type=float, default=0.45)
    parser.add_argument("--neural-weight", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--verbose-rollouts", action="store_true", help="Log caller-visible payloads, raw model responses, and alias maps for inspection.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of model-selection calls to run concurrently.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    summary = run_live_online_training(
        tasks_path=args.tasks.resolve(),
        listings_path=args.listings.resolve(),
        live_outputs_path=args.live_outputs.resolve(),
        out_dir=args.out_dir.resolve(),
        rounds=args.rounds,
        candidate_count=args.candidate_count,
        slate_size=args.slate_size,
        seed=args.seed,
        judge_model=args.judge_model,
        discount_gamma=args.discount_gamma,
        exploration_weight=args.exploration_weight,
        neural_weight=args.neural_weight,
        learning_rate=args.learning_rate,
        sleep_seconds=args.sleep_seconds,
        reward_mode=args.reward_mode,
        verbose_rollouts=args.verbose_rollouts,
        concurrency=args.concurrency,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
