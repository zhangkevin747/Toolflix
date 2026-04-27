"""
Measure the cosine-similarity floor for 'correct' tool picks on the GT
benchmark, so we can set a retrieval-time threshold τ empirically.

For each GT task, the agent produces a decomposed query. We compute the
similarity between that query and the selected tool. We compare two pools:

  - correct picks:   eval answer_found = True
  - incorrect picks: eval answer_found = False

If the two distributions are separable, τ lives between them. If not, a
hard threshold will hurt recall on valid picks.
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_picks(path):
    d = json.load(open(path))
    rows = []
    for r in d.get("reranker", []):
        rows.append({
            "query": r["query"],
            "picked_tool_key": f"{r['selected']['server_id']}/{r['selected']['tool_name']}",
            "answer_found": r["answer_found"],
            "regime": "reranker",
            "expected_category": r["expected_category"],
        })
    for r in d.get("retriever", []):
        rows.append({
            "query": r["query"],
            "picked_tool_key": f"{r['selected']['server_id']}/{r['selected']['tool_name']}",
            "answer_found": r["answer_found"],
            "regime": "retriever",
            "expected_category": r["expected_category"],
        })
    return rows


def main():
    endpoints = json.load(open("data/embeddings.json"))
    ep_embs = np.array([e["embedding"] for e in endpoints], dtype=np.float32)
    ep_norms = np.linalg.norm(ep_embs, axis=1) + 1e-9
    ep_keys = [f"{e['server_id']}/{e['tool_name']}" for e in endpoints]
    key_to_idx = {k: i for i, k in enumerate(ep_keys)}

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Use the most recent GT evals — both the retriever-only-trained and the on-policy
    paths = [
        "data/gt_eval_rejudged.json",      # retriever-only-trained reranker, 65.1%
        "data/gt_eval_onpolicy475.json",   # on-policy-trained reranker, 59.6%
    ]
    rows = []
    for p in paths:
        if Path(p).exists():
            rows.extend(load_picks(p))
    print(f"loaded {len(rows)} (query, picked tool) pairs\n")

    # Encode all unique queries
    uq = sorted(set(r["query"] for r in rows))
    q_embs = encoder.encode(uq, show_progress_bar=False)
    q_norms = np.linalg.norm(q_embs, axis=1) + 1e-9
    q_idx = {q: i for i, q in enumerate(uq)}

    # Compute cosine for each row
    sims = []
    for r in rows:
        qi = q_idx[r["query"]]
        ti = key_to_idx.get(r["picked_tool_key"])
        if ti is None:
            r["similarity"] = None
            continue
        sim = float((q_embs[qi] @ ep_embs[ti]) / (q_norms[qi] * ep_norms[ti]))
        r["similarity"] = sim
        sims.append(sim)

    # Split distributions
    def stats(label, vals):
        if not vals:
            print(f"  {label}: no data"); return
        a = np.array(vals)
        q = [np.min(a), np.percentile(a, 10), np.percentile(a, 25),
             np.median(a), np.percentile(a, 75), np.max(a)]
        print(f"  {label:<30}  n={len(vals):>3}  "
              f"min={q[0]:.2f}  p10={q[1]:.2f}  p25={q[2]:.2f}  "
              f"med={q[3]:.2f}  p75={q[4]:.2f}  max={q[5]:.2f}")

    print("Similarity distribution by (regime, correctness):")
    for regime in ["retriever", "reranker"]:
        correct = [r["similarity"] for r in rows
                   if r["regime"] == regime and r["answer_found"] and r.get("similarity") is not None]
        wrong = [r["similarity"] for r in rows
                 if r["regime"] == regime and not r["answer_found"] and r.get("similarity") is not None]
        stats(f"{regime} / answer_found=T", correct)
        stats(f"{regime} / answer_found=F", wrong)

    # Within-category break-out: how many CORRECT picks are below common τ candidates
    print("\nCorrect-pick counts at various τ thresholds:")
    print("  τ      | retr ✓ kept  | retr ✓ lost  | rerk ✓ kept  | rerk ✓ lost")
    for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        counts = {}
        for regime in ["retriever", "reranker"]:
            correct = [r for r in rows
                       if r["regime"] == regime and r["answer_found"] and r.get("similarity") is not None]
            kept = sum(1 for r in correct if r["similarity"] >= tau)
            lost = sum(1 for r in correct if r["similarity"] < tau)
            counts[regime] = (kept, lost)
        retk, retl = counts["retriever"]
        rerk, rerl = counts["reranker"]
        print(f"  {tau:.2f}    |    {retk:>3}       |    {retl:>3}       |    {rerk:>3}       |    {rerl:>3}")

    print("\nWrong-pick counts at various τ thresholds (high = τ filters out the bad picks):")
    print("  τ      | retr ✗ kept  | retr ✗ filtered | rerk ✗ kept | rerk ✗ filtered")
    for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        counts = {}
        for regime in ["retriever", "reranker"]:
            wrong = [r for r in rows
                     if r["regime"] == regime and not r["answer_found"] and r.get("similarity") is not None]
            kept = sum(1 for r in wrong if r["similarity"] >= tau)
            filtered = sum(1 for r in wrong if r["similarity"] < tau)
            counts[regime] = (kept, filtered)
        retk, retf = counts["retriever"]
        rerk, rerf = counts["reranker"]
        print(f"  {tau:.2f}    |    {retk:>3}       |    {retf:>3}        |    {rerk:>3}       |    {rerf:>3}")

    # Per-category minimum similarity of correct picks (reranker regime)
    print("\nPer-category cosine floor of correct reranker picks:")
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rows:
        if r["regime"] != "reranker" or not r["answer_found"]:
            continue
        by_cat[r["expected_category"]].append(r["similarity"])
    for cat, sims in sorted(by_cat.items()):
        if sims:
            print(f"  {cat:<12}  n={len(sims):>2}  min={min(sims):.2f}  "
                  f"p10={np.percentile(sims, 10):.2f}  median={np.median(sims):.2f}")


if __name__ == "__main__":
    main()
