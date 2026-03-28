"""
Evaluate the pairwise reranker against baselines using existing feedback.

For each feedback record we know:
  - The 10 retriever candidates (with similarity scores)
  - Which tool the agent selected
  - Whether it succeeded (ground truth)

We train the new reranker on a train split, then on the test split we ask:
  "If the agent always picked rank 1, how often would it succeed?"

Baselines:
  1. Retriever only (similarity ranking)
  2. Random (shuffle candidates)
  3. Oracle (success-rate prior — rank by historical success rate)

New model:
  4. Pairwise reranker (trained on train split)
"""
import json
import random
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from reranker import Reranker


def load_feedback(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ("retriever_candidates" in r and "selected" in r
                    and "rating" in r and "error" not in r):
                records.append(r)
    return records


def retriever_rank1_key(record: dict) -> str:
    """Return the key of the retriever's #1 candidate."""
    c = record["retriever_candidates"][0]
    return f"{c['server_id']}/{c['tool_name']}"


def success_rate_ranking(records_for_prior: list[dict], record: dict) -> list[dict]:
    """Rank candidates by historical success rate (oracle-ish baseline)."""
    tool_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in records_for_prior:
        sel = r["selected"]
        key = f"{sel['server_id']}/{sel['tool_name']}"
        tool_stats[key]["total"] += 1
        if r["rating"].get("success"):
            tool_stats[key]["success"] += 1

    candidates = record["retriever_candidates"]
    scored = []
    for c in candidates:
        key = f"{c['server_id']}/{c['tool_name']}"
        stats = tool_stats[key]
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0.5
        scored.append({**c, "score": rate})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def compute_auc_and_pairwise(test_records: list[dict], rank_fn):
    """Compute AUC and pairwise accuracy.

    For each record, the ranking function scores all 10 candidates.
    The selected tool has a known label (success=1 or 0).

    AUC: across all (record, candidate) pairs, treat the selected-and-succeeded
    tool as positive and selected-and-failed tool as negative.
    Score = whatever rank/score the model gave it.

    Pairwise accuracy: within each record's candidate list, for every pair
    (selected_tool, other_tool), count how often the model orders them correctly:
      - selected succeeded → selected should be ranked above others
      - selected failed → selected should be ranked below others
    """
    scores_pos = []  # rerank scores for selected tools that succeeded
    scores_neg = []  # rerank scores for selected tools that failed
    pair_correct = 0
    pair_total = 0

    for r in test_records:
        ranked = rank_fn(r)
        if not ranked:
            continue

        sel_key = f"{r['selected']['server_id']}/{r['selected']['tool_name']}"
        success = r["rating"].get("success", False)

        # Get score for each candidate
        candidate_scores = {}
        for c in ranked:
            key = f"{c['server_id']}/{c['tool_name']}"
            # Use position-based score (higher rank = higher score) for baselines
            # that don't have explicit scores
            candidate_scores[key] = c.get("rerank_score", c.get("score", c.get("similarity", 0)))

        sel_score = candidate_scores.get(sel_key)
        if sel_score is None:
            continue

        if success:
            scores_pos.append(sel_score)
        else:
            scores_neg.append(sel_score)

        # Pairwise accuracy within this record
        other_scores = [s for k, s in candidate_scores.items() if k != sel_key]
        for other_s in other_scores:
            pair_total += 1
            if success and sel_score > other_s:
                pair_correct += 1
            elif not success and sel_score < other_s:
                pair_correct += 1
            elif sel_score == other_s:
                pair_correct += 0.5  # tie

    # Compute AUC: P(score_pos > score_neg) across all pos/neg pairs
    auc_correct = 0
    auc_total = 0
    for sp in scores_pos:
        for sn in scores_neg:
            auc_total += 1
            if sp > sn:
                auc_correct += 1
            elif sp == sn:
                auc_correct += 0.5

    auc = auc_correct / auc_total if auc_total else 0.0
    pairwise_acc = pair_correct / pair_total if pair_total else 0.0

    return auc, pairwise_acc, len(scores_pos), len(scores_neg)


def evaluate_ranking(test_records: list[dict], rank_fn, top_k: int = 5, label: str = ""):
    """Evaluate a ranking function.

    rank_fn(record) -> list of candidates sorted by score descending.
    Each candidate dict should have a 'rerank_score', 'score', or 'similarity' key.
    """
    selected_in_top1 = 0
    selected_in_topk = 0
    correct_direction = 0
    cat_stats = defaultdict(lambda: {"total": 0, "selected_top1": 0, "selected_topk": 0})
    success_ranks = []
    fail_ranks = []

    for r in test_records:
        ranked = rank_fn(r)
        if not ranked:
            continue

        sel_key = f"{r['selected']['server_id']}/{r['selected']['tool_name']}"
        success = r["rating"].get("success", False)
        category = r["category"]

        sel_rank = None
        for i, c in enumerate(ranked):
            key = f"{c['server_id']}/{c['tool_name']}"
            if key == sel_key:
                sel_rank = i + 1
                break

        if sel_rank is None:
            continue

        if sel_rank == 1:
            selected_in_top1 += 1
            cat_stats[category]["selected_top1"] += 1
        if sel_rank <= top_k:
            selected_in_topk += 1
            cat_stats[category]["selected_topk"] += 1
        cat_stats[category]["total"] += 1

        if success:
            success_ranks.append(sel_rank)
        else:
            fail_ranks.append(sel_rank)

        if success and sel_rank <= top_k:
            correct_direction += 1
        elif not success and sel_rank > top_k:
            correct_direction += 1

    total = sum(cs["total"] for cs in cat_stats.values())

    # AUC and pairwise accuracy
    auc, pairwise_acc, n_pos, n_neg = compute_auc_and_pairwise(test_records, rank_fn)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if total == 0:
        print("  No records matched — skipping metrics.")
        return
    print(f"  Selected tool at rank 1: {selected_in_top1}/{total} ({100*selected_in_top1/total:.1f}%)")
    print(f"  Selected tool in top {top_k}:  {selected_in_topk}/{total} ({100*selected_in_topk/total:.1f}%)")
    print(f"  Correct direction:       {correct_direction}/{total} ({100*correct_direction/total:.1f}%)")
    if success_ranks:
        print(f"  Avg rank of succeeded:   {np.mean(success_ranks):.2f}  (lower=better, ideal=1)")
    if fail_ranks:
        print(f"  Avg rank of failed:      {np.mean(fail_ranks):.2f}  (higher=better)")
    print(f"  AUC:                     {auc:.4f}  ({n_pos} pos, {n_neg} neg)")
    print(f"  Pairwise accuracy:       {pairwise_acc:.4f}")

    print(f"\n  Per-category (selected in top {top_k}):")
    for cat in ["fetch", "pdf", "search", "filesystem"]:
        cs = cat_stats[cat]
        if cs["total"] > 0:
            print(f"    {cat:12s}: {cs['selected_topk']}/{cs['total']} ({100*cs['selected_topk']/cs['total']:.1f}%)")

    return {
        "top1": selected_in_top1 / total if total else 0,
        "topk": selected_in_topk / total if total else 0,
        "correct_dir": correct_direction / total if total else 0,
        "avg_success_rank": np.mean(success_ranks) if success_ranks else 0,
        "avg_fail_rank": np.mean(fail_ranks) if fail_ranks else 0,
        "auc": auc,
        "pairwise_acc": pairwise_acc,
    }


def main():
    data_dir = Path(__file__).parent.parent / "data"
    feedback_path = str(data_dir / "feedback.jsonl")
    embeddings_path = str(data_dir / "embeddings.json")

    print("Loading feedback...")
    all_records = load_feedback(feedback_path)
    print(f"Loaded {len(all_records)} records")

    # Train/test split: 80/20 by time (first 80% for training, last 20% for eval)
    split = int(len(all_records) * 0.8)
    train_records = all_records[:split]
    test_records = all_records[split:]
    print(f"Train: {len(train_records)}, Test: {len(test_records)}")

    # Write a temporary train-only feedback file for the reranker
    train_feedback_path = str(data_dir / "_train_feedback.jsonl")
    with open(train_feedback_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    # --- Baseline 1: Retriever only ---
    def retriever_ranking(record):
        return sorted(record["retriever_candidates"],
                      key=lambda c: c.get("similarity", 0), reverse=True)

    evaluate_ranking(test_records, retriever_ranking, label="Baseline: Retriever (similarity)")

    # --- Baseline 2: Random ---
    def random_ranking(record):
        cands = list(record["retriever_candidates"])
        random.shuffle(cands)
        return cands

    random.seed(42)
    evaluate_ranking(test_records, random_ranking, label="Baseline: Random")

    # --- Baseline 3: Success-rate prior ---
    def success_prior_ranking(record):
        return success_rate_ranking(train_records, record)

    evaluate_ranking(test_records, success_prior_ranking, label="Baseline: Success-rate prior (wide-only proxy)")

    # --- New: Two-head reranker ---
    print("\n\nTraining two-head reranker on train split...")
    reranker = Reranker(
        embeddings_path=embeddings_path,
        feedback_path=train_feedback_path,
        model_path=None,  # Train fresh
    )
    reranker.train_on_feedback(train_feedback_path, epochs=50)

    def _make_ranking_fn(head: str):
        def ranking_fn(record):
            query = record.get("query", record.get("task", ""))
            category = record.get("category", "")
            candidates = record["retriever_candidates"]
            enriched = []
            for c in candidates:
                key = f"{c['server_id']}/{c['tool_name']}"
                idx = reranker.endpoint_to_idx.get(key)
                if idx is not None:
                    ep = reranker.endpoints[idx]
                    enriched.append({**c, "install": ep.get("install", "")})
                else:
                    enriched.append({**c, "install": ""})
            return reranker.rerank(enriched, query, query_category=category,
                                   top_k=10, explore=False, head=head)
        return ranking_fn

    evaluate_ranking(test_records, _make_ranking_fn("relevance"),
                     label="NEW: Relevance head only (deep, no feedback)")
    evaluate_ranking(test_records, _make_ranking_fn("quality"),
                     label="NEW: Quality head only (wide + deep, feedback-trained)")
    evaluate_ranking(test_records, _make_ranking_fn("combined"),
                     label="NEW: Combined (alpha-blended relevance + quality)")

    # Cleanup
    Path(train_feedback_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
