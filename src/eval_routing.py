"""
Tool-routing evaluation.

Ground truth is the tool category. No string matching on tool output,
no end-to-end LLM judging — purely: did the retriever/reranker put a
tool of the correct category at top-1? Were synthetics avoided?

Compares three regimes per task:
  1. retriever only (cosine top-1)
  2. reranker on top of retriever (wide+deep scored top-1)
  3. reranker with model round-robin (tests per-model routing)

Usage:
  python src/eval_routing.py                     # all 3 regimes
  python src/eval_routing.py --no-decompose      # skip LLM decompose step
  python src/eval_routing.py --limit 20          # first N tasks
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from retriever import Retriever
from reranker import Reranker, MODEL_REGISTRY
from agent import Agent


SYNTHETIC_MARKERS = ("-truncated-", "-hollow-", "-flaky-", "-stale-")


def is_synthetic(server_id: str) -> bool:
    return any(m in server_id for m in SYNTHETIC_MARKERS)


def load_tasks(path: Path, limit: int | None = None):
    tasks = json.loads(path.read_text())
    if limit:
        tasks = tasks[:limit]
    return tasks


def score_candidates(candidates: list[dict], expected_category: str) -> dict:
    """Compute routing metrics for a candidate list."""
    if not candidates:
        return {
            "top1_correct": False,
            "top1_synthetic": False,
            "top5_has_correct": False,
            "top1_server": None,
            "top1_category": None,
        }
    top1 = candidates[0]
    top5 = candidates[:5]
    top1_correct = top1.get("category") == expected_category
    top1_synthetic = is_synthetic(top1["server_id"])
    return {
        "top1_correct": top1_correct,
        "top1_synthetic": top1_synthetic,
        "top1_working_and_correct": top1_correct and not top1_synthetic,
        "top5_has_correct": any(c.get("category") == expected_category for c in top5),
        "top5_has_working_correct": any(
            c.get("category") == expected_category and not is_synthetic(c["server_id"])
            for c in top5
        ),
        "top1_server": top1["server_id"],
        "top1_tool": top1["tool_name"],
        "top1_category": top1.get("category"),
    }


def run_eval(tasks, retriever, reranker, agent, use_decompose: bool,
             round_robin: bool):
    results = []
    for i, t in enumerate(tasks):
        task_text = t["task"]
        expected = t["expected_category"]

        if use_decompose:
            try:
                query = agent.call_1_decompose(task_text)
            except Exception as e:
                print(f"  decompose failed: {e}")
                query = task_text
        else:
            query = task_text

        candidates = retriever.retrieve(query, top_k=100)

        retriever_metrics = score_candidates(candidates, expected)

        if round_robin:
            from reranker import MODEL_TO_IDX
            model_name = MODEL_REGISTRY[i % len(MODEL_REGISTRY)]
            reranker.model_name = model_name
            reranker._model_idx = MODEL_TO_IDX[model_name]
        reranked = reranker.rerank(candidates, query, top_k=100, explore=False)
        reranker_metrics = score_candidates(reranked, expected)

        results.append({
            "id": t["id"],
            "task": task_text,
            "expected_category": expected,
            "query": query,
            "retriever": retriever_metrics,
            "reranker": reranker_metrics,
        })

        r_ok = "✓" if retriever_metrics["top1_correct"] else "✗"
        rr_ok = "✓" if reranker_metrics["top1_correct"] else "✗"
        print(f"  [{i+1:3d}/{len(tasks)}] {t['id']:<18}  "
              f"retr={r_ok} ({retriever_metrics['top1_category']}/{retriever_metrics['top1_server'][:30]})  "
              f"rerank={rr_ok} ({reranker_metrics['top1_category']}/{reranker_metrics['top1_server'][:30]})")

    return results


def summarize(results, label: str):
    n = len(results)
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["expected_category"]].append(r)

    keys = ["top1_correct", "top1_working_and_correct", "top5_has_correct",
            "top5_has_working_correct", "top1_synthetic"]
    metrics = {
        side: {k: sum(r[side][k] for r in results) for k in keys}
        for side in ("retriever", "reranker")
    }

    print(f"\n=== {label} — {n} tasks ===")
    print(f"{'Metric':<28}{'Retriever':>14}{'Reranker':>14}")
    for metric in keys:
        r = metrics["retriever"][metric]
        rr = metrics["reranker"][metric]
        print(f"  {metric:<26}{r:>4}/{n} ({100*r/n:>4.1f}%){rr:>6}/{n} ({100*rr/n:>4.1f}%)")

    print("\nPer-category top-1 correct:")
    print(f"  {'category':<14}{'Retriever':>14}{'Reranker':>14}")
    for cat in sorted(by_cat.keys()):
        rs = by_cat[cat]
        r_hits = sum(x["retriever"]["top1_correct"] for x in rs)
        rr_hits = sum(x["reranker"]["top1_correct"] for x in rs)
        print(f"  {cat:<14}{r_hits}/{len(rs)} ({100*r_hits/len(rs):>4.0f}%){rr_hits:>5}/{len(rs)} ({100*rr_hits/len(rs):>4.0f}%)")

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/gaia_routing.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-decompose", action="store_true",
                    help="skip LLM decompose — use task text as-is for retrieval")
    ap.add_argument("--round-robin", action="store_true",
                    help="cycle through MODEL_REGISTRY for per-model routing")
    ap.add_argument("--out", default="data/routing_eval_results.json")
    args = ap.parse_args()

    random.seed(0)

    tasks = load_tasks(Path(args.tasks), args.limit)
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    retriever = Retriever(
        embeddings_path="data/embeddings.json",
        tools_path="data/tools.json",
    )
    model_path = Path("data/models/reranker.pt")
    reranker = Reranker(
        embeddings_path="data/embeddings.json",
        feedback_path="data/feedback.jsonl",
        model_path=str(model_path) if model_path.exists() else None,
        model_name=MODEL_REGISTRY[0],
    )
    agent = Agent(model=MODEL_REGISTRY[0]) if not args.no_decompose else None

    results = run_eval(
        tasks, retriever, reranker, agent,
        use_decompose=not args.no_decompose,
        round_robin=args.round_robin,
    )

    label = (
        "Tool routing"
        + (" (no decompose)" if args.no_decompose else " (with decompose)")
        + (" + round-robin" if args.round_robin else "")
    )
    metrics = summarize(results, label)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "label": label,
        "n": len(results),
        "metrics": metrics,
        "results": results,
    }, indent=2))
    print(f"\nWrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
