"""Plot rank-over-time for the three exp2 sub-experiments."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS = [
    ("2a_cold_start",  "Cold start (hidden → introduced)"),
    ("2b_good_to_bad", "Good → bad (canary breaks at step 500)"),
    ("2c_bad_to_good", "Bad → good (canary fixed at step 500)"),
]

EVENT_STEP = 500


def load(name):
    rows = [json.loads(l) for l in open(f"data/exp2_{name}.jsonl")]
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    steps = sorted(by_step)
    median_rank, iqr_low, iqr_high = [], [], []
    hidden = []
    for s in steps:
        ranks = [r["canary_rank"] for r in by_step[s]
                 if r["canary_rank"] is not None]
        if not ranks:
            hidden.append(s)
            median_rank.append(None)
            iqr_low.append(None); iqr_high.append(None)
        else:
            median_rank.append(statistics.median(ranks))
            iqr_low.append(min(ranks))
            iqr_high.append(max(ranks))
    return steps, median_rank, iqr_low, iqr_high, hidden, by_step


def plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (name, title) in zip(axes, EXPERIMENTS):
        steps, med, lo, hi, hidden, by_step = load(name)

        # Fill between min/max
        vis_steps = [s for s, m in zip(steps, med) if m is not None]
        vis_med = [m for m in med if m is not None]
        vis_lo = [v for v in lo if v is not None]
        vis_hi = [v for v in hi if v is not None]
        ax.fill_between(vis_steps, vis_lo, vis_hi, alpha=0.25)
        ax.plot(vis_steps, vis_med, marker="o", markersize=3, linewidth=1.5,
                label="canary rank")

        # Horizontal reference lines
        ax.axhline(1, color="#2ca02c", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.axhline(5, color="#d62728", linestyle="--", alpha=0.3, linewidth=0.8,
                   label="top-5 threshold")

        # Vertical event line
        ax.axvline(EVENT_STEP, color="black", linestyle=":", alpha=0.6,
                   label=f"state change @ {EVENT_STEP}")

        # Shade hidden interval for 2a
        if hidden:
            ax.axvspan(0, max(hidden), color="grey", alpha=0.10,
                       label="canary hidden")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.invert_yaxis()
        ax.set_yscale("symlog")
        ax.set_yticks([1, 5, 10, 25, 50, 100])
        ax.set_yticklabels([1, 5, 10, 25, 50, 100])
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("canary rank (lower = better, log scale)")

    out = Path("data/exp2_cold_start.png")
    fig.suptitle("Experiment 2: reranker dynamics under tool state transitions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    plot()
