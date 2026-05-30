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

from training.simulated_online import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cheap one-pass simulated online ToolBandit training loop.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks/tasks_full_ready.jsonl")
    parser.add_argument("--listings", type=Path, default=ROOT / "data/pool/listings.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/runs/online_train_v1")
    parser.add_argument("--rounds", type=int, default=None, help="Maximum unique tasks to consume. Defaults to all tasks.")
    parser.add_argument("--candidate-count", type=int, default=80)
    parser.add_argument("--slate-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--discount-gamma", type=float, default=0.985)
    parser.add_argument("--exploration-weight", type=float, default=0.45)
    parser.add_argument("--neural-weight", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_training(
        tasks_path=args.tasks.resolve(),
        listings_path=args.listings.resolve(),
        out_dir=args.out_dir.resolve(),
        rounds=args.rounds,
        candidate_count=args.candidate_count,
        slate_size=args.slate_size,
        seed=args.seed,
        discount_gamma=args.discount_gamma,
        exploration_weight=args.exploration_weight,
        neural_weight=args.neural_weight,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
