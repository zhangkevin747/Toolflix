"""
Replay feedback.jsonl through the reranker's online training loop in the
exact same sequence as the live pipeline, and save the resulting weights.

This reproduces the in-memory state the live pipeline had at the end of a
run, without needing to re-execute any MCP/LLM calls.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from reranker import Reranker

FEEDBACK = REPO / "data/feedback.jsonl"
OUT = REPO / "data/models/reranker.pt"


def main():
    # Construct with non-existent feedback path so replay buffer starts empty.
    # Wide aggregates also start empty and accumulate via online_update.
    rr = Reranker(
        embeddings_path=str(REPO / "data/embeddings.json"),
        feedback_path="/nonexistent/feedback.jsonl",
        model_path=None,
        model_name="gpt-5.4-nano",
    )

    # Double-check the reranker starts clean.
    assert len(rr._replay_buffer) == 0, f"buffer should be empty, got {len(rr._replay_buffer)}"
    assert rr._trained_up_to == 0

    rows = [json.loads(l) for l in open(FEEDBACK)]
    print(f"Replaying {len(rows)} rollouts...")

    n_deep_trains = 0
    for i, feedback in enumerate(rows, 1):
        if "selected" not in feedback or "rating" not in feedback:
            continue
        rr.online_update(feedback)
        trained = rr.maybe_batch_train(batch_size=50)
        if trained:
            n_deep_trains += 1
        if i % 100 == 0:
            print(f"  replayed {i}/{len(rows)}  deep_trains={n_deep_trains}")

    print(f"Total deep batch trains: {n_deep_trains}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(OUT))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
