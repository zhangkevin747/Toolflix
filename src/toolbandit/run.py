"""Command-line entry point.

  Simulated (free, no API keys, good for reading/learning):
      python -m toolbandit.run sim

  Live (real models choose; needs OPENAI_API_KEY / OPENROUTER_API_KEY):
      python -m toolbandit.run live --reward metadata     # real picks, cheap reward
      python -m toolbandit.run live --reward judge        # real picks, LLM-judged reward

Execution backend (where tool output comes from):
      --execute cached   replay recorded outputs (fast, free; default for sim)
      --execute live     actually call the real MCP servers (default for live mode;
                         needs external/mcp-bench servers + .env)

  Tip: `sim --execute live` exercises the real MCP calls with no LLM/API cost.

Run with --help to see all options. To read the code instead, start at loop.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from toolbandit.caller import LLMPicker, policy_pick
from toolbandit.loop import run
from toolbandit.reward import JudgeReward, metadata_reward


def main() -> int:
    p = argparse.ArgumentParser(description="Train the ToolBandit on the tool marketplace.")
    p.add_argument("mode", choices=["sim", "live"], help="sim = no LLM; live = real models pick tools")
    p.add_argument("--reward", choices=["metadata", "judge"], default="metadata",
                   help="live only: metadata (cheap) or judge (LLM decides). sim always uses metadata.")
    p.add_argument("--judge-model", default="self", help="judge mode: model that scores, or 'self'")
    p.add_argument("--tasks", type=Path, default=ROOT / "data/tasks/tasks_ready.jsonl")
    p.add_argument("--listings", type=Path, default=ROOT / "data/pool/listings.jsonl")
    p.add_argument("--cached-outputs", type=Path, default=ROOT / "data/pool/live_base_validation.jsonl")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/runs/toolbandit")
    p.add_argument("--rounds", type=int, default=None, help="cap on tasks (default: all)")
    p.add_argument("--candidate-count", type=int, default=80)
    p.add_argument("--slate-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--execute", choices=["cached", "live"], default=None,
                   help="where tool output comes from (default: live mode -> live, sim mode -> cached)")
    p.add_argument("--mcp-timeout", type=float, default=45.0, help="live only: per-call/connect timeout (s)")
    p.add_argument("--no-mcp-cache", action="store_true", help="live only: bypass MCP-Bench's tool cache")
    args = p.parse_args()

    # Pick the two pluggable pieces based on the mode.
    if args.mode == "sim":
        picker, reward_fn = policy_pick, metadata_reward
    else:
        picker = LLMPicker()
        reward_fn = metadata_reward if args.reward == "metadata" else JudgeReward(args.judge_model)

    execution = args.execute or ("live" if args.mode == "live" else "cached")

    summary = run(
        tasks_path=args.tasks.resolve(),
        listings_path=args.listings.resolve(),
        cached_outputs_path=args.cached_outputs.resolve(),
        out_dir=args.out_dir.resolve(),
        picker=picker,
        reward_fn=reward_fn,
        rounds=args.rounds,
        candidate_count=args.candidate_count,
        slate_size=args.slate_size,
        seed=args.seed,
        execution=execution,
        mcp_timeout=args.mcp_timeout,
        mcp_cache=not args.no_mcp_cache,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
