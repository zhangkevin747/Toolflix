"""
Experiment 1: Identifying corrupted tools.

For each query, score every in-category tool with:
  (a) Retriever: cosine similarity (semantic only)
  (b) Reranker: wide+deep score, batched in one forward pass per query

Within each query's ground-truth category, partition tools into
{real, truncated, hollow, flaky, stale} and measure:
  - Mean within-category rank per group
  - AUC: P(real > synthetic | same query, same category)
  - Precision@1: top in-category tool is real
  - AUC by corruption type (real vs each of the 4 types)

This directly loads the trained reranker weights — no feedback replay,
no re-encoding of tool embeddings, no per-tool Python loop.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from reranker import WideAndDeepModel, MODEL_REGISTRY, MODEL_TO_IDX, WideFeatures


SYNTH_TYPES = ("truncated", "hollow", "flaky", "stale")


def classify(server_id: str) -> str:
    for t in SYNTH_TYPES:
        if f"-synth-{t}-" in server_id:
            return t
    if "-synth-" in server_id:
        return "other-synth"
    return "real"


def load_queries():
    """Use each GT benchmark task's decomposed query (from cached eval).
    Falls back to the raw task text if no decomposed query is cached."""
    gt = {t["id"]: t for t in json.loads(Path("data/gaia_gt.json").read_text())}
    cache = {r["id"]: r for r in json.loads(Path("data/gt_eval_full.json").read_text())["retriever"]}
    out = []
    for tid, task in gt.items():
        q = cache.get(tid, {}).get("query") or task["task"]
        out.append((q, task["expected_category"]))
    return out


def main():
    print("Loading encoder & tool embeddings ...", flush=True)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    with open("data/embeddings.json") as f:
        endpoints = json.load(f)
    with open("data/tools.json") as f:
        tools_meta = {s["id"]: s for s in json.load(f)}

    ep_embs = np.array([e["embedding"] for e in endpoints], dtype=np.float32)
    ep_norms = np.linalg.norm(ep_embs, axis=1)
    ep_servers = [e["server_id"] for e in endpoints]
    ep_tools = [e["tool_name"] for e in endpoints]
    ep_cats = [tools_meta[s].get("category", "?") for s in ep_servers]
    ep_groups = [classify(s) for s in ep_servers]
    keys = [f"{s}/{t}" for s, t in zip(ep_servers, ep_tools)]
    N = len(keys)
    print(f"  {N} endpoints", flush=True)

    print("Loading reranker weights ...", flush=True)
    state = torch.load("data/models/reranker.pt", map_location="cpu", weights_only=False)
    model = WideAndDeepModel()
    model.load_state_dict(state)
    model.eval()

    print("Rebuilding wide features from feedback.jsonl ...", flush=True)
    wide = WideFeatures()
    with open("data/feedback.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sel = r.get("selected")
            if not sel:
                continue
            key = f"{sel['server_id']}/{sel['tool_name']}"
            ok = bool(r.get("rating", {}).get("success"))
            wide.update(key, r.get("model", MODEL_REGISTRY[0]), ok)
    wide.snapshot_norms()
    print(f"  wide usage rows: {len(wide.usage_count)}", flush=True)

    queries = load_queries()
    print(f"Scoring {len(queries)} queries ...", flush=True)

    q_embs = encoder.encode([q for q, _ in queries], show_progress_bar=False)
    q_embs = np.asarray(q_embs, dtype=np.float32)

    # Pre-compute per-endpoint wide fields that don't depend on the query
    # (3 sim-independent: usage, sr, mt_sr). Sim-dependent slots filled per
    # query below, yielding the full 6-dim wide vector.
    model_name = MODEL_REGISTRY[0]
    max_usage = max(wide.usage_count.values(), default=1) or 1
    base_wide = np.zeros((N, 3), dtype=np.float32)  # [usage, sr, mtsr]
    for i, k in enumerate(keys):
        usage = wide.usage_count[k]
        norm_u = usage / max_usage if max_usage else 0.0
        sr = wide.success_count[k] / usage if usage > 0 else 0.0
        mt_key = f"{model_name}||{k}"
        mtu = wide.model_tool_usage[mt_key]
        mtsr = wide.model_tool_success[mt_key] / mtu if mtu > 0 else 0.0
        base_wide[i] = [norm_u, sr, mtsr]

    ep_embs_t = torch.tensor(ep_embs, dtype=torch.float32)
    model_idx_t = torch.full((N,), MODEL_TO_IDX[model_name], dtype=torch.long)

    rows = []
    for qi, (query, gt_cat) in enumerate(queries):
        q_emb = q_embs[qi]
        q_norm = np.linalg.norm(q_emb) + 1e-9

        sims = (ep_embs @ q_emb) / (ep_norms * q_norm)  # [N]

        # Full wide vector (6-dim): [usage, sr, mtsr, sim, sim*sr, sim*mtsr]
        sr_col = base_wide[:, 1:2]
        mtsr_col = base_wide[:, 2:3]
        sim_col = sims.reshape(-1, 1).astype(np.float32)
        wide_np = np.concatenate(
            [base_wide, sim_col, sim_col * sr_col, sim_col * mtsr_col], axis=1
        )
        wide_t = torch.tensor(wide_np, dtype=torch.float32)
        q_t = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0).repeat(N, 1)

        with torch.no_grad():
            rer_scores = model(wide_t, model_idx_t, q_t, ep_embs_t).squeeze(-1).numpy()

        for i, (s, t, c, g) in enumerate(zip(ep_servers, ep_tools, ep_cats, ep_groups)):
            if c != gt_cat:
                continue
            rows.append({
                "qi": qi, "query": query, "gt_cat": gt_cat,
                "server_id": s, "tool_name": t, "group": g,
                "retriever": float(sims[i]), "reranker": float(rer_scores[i]),
            })

    print(f"  produced {len(rows)} (query, tool) pairs\n", flush=True)

    # Within-category rank
    def rank_rows(scorer):
        by_q = defaultdict(list)
        for r in rows:
            by_q[r["qi"]].append(r)
        out = []
        for qi, items in by_q.items():
            items_sorted = sorted(items, key=lambda x: x[scorer], reverse=True)
            for rank, r in enumerate(items_sorted, 1):
                out.append({**r, "rank": rank})
        return out

    def mean_rank_by_group(ranked):
        buckets = defaultdict(list)
        for r in ranked:
            buckets[(r["gt_cat"], r["group"])].append(r["rank"])
        return {k: (float(np.mean(v)), len(v)) for k, v in buckets.items()}

    def auc(rows_, scorer, per_cat=False):
        by_q = defaultdict(list)
        for r in rows_:
            by_q[r["qi"]].append(r)
        wins = ties = total = 0
        per_cat_buckets = defaultdict(lambda: [0, 0, 0])
        for qi, items in by_q.items():
            reals = [r[scorer] for r in items if r["group"] == "real"]
            synths = [r[scorer] for r in items if r["group"] in SYNTH_TYPES]
            cat = items[0]["gt_cat"]
            for rs in reals:
                for ss in synths:
                    total += 1
                    per_cat_buckets[cat][2] += 1
                    if rs > ss:
                        wins += 1; per_cat_buckets[cat][0] += 1
                    elif rs == ss:
                        ties += 1; per_cat_buckets[cat][1] += 1
        a = (wins + 0.5 * ties) / total if total else float("nan")
        if per_cat:
            return a, total, {c: (w + 0.5 * t) / max(n, 1) for c, (w, t, n) in per_cat_buckets.items()}
        return a, total

    def p_at_1(rows_, scorer, per_cat=False):
        by_q = defaultdict(list)
        for r in rows_:
            by_q[r["qi"]].append(r)
        hits = total = 0
        buckets = defaultdict(lambda: [0, 0])
        for qi, items in by_q.items():
            top = max(items, key=lambda x: x[scorer])
            cat = top["gt_cat"]
            total += 1; buckets[cat][1] += 1
            if top["group"] == "real":
                hits += 1; buckets[cat][0] += 1
        p = hits / total if total else float("nan")
        if per_cat:
            return p, {c: h/t for c, (h, t) in buckets.items()}
        return p

    retr_rank = mean_rank_by_group(rank_rows("retriever"))
    rerk_rank = mean_rank_by_group(rank_rows("reranker"))

    cats = sorted({k[0] for k in retr_rank})
    print("Mean within-category rank (lower = better)")
    groups = ("real",) + SYNTH_TYPES
    print(f"  {'category':<12} {'scorer':<10}" + "".join(f"{g:>10}" for g in groups))
    for cat in cats:
        for scorer, stats in [("retriever", retr_rank), ("reranker", rerk_rank)]:
            line = f"  {cat:<12} {scorer:<10}"
            for g in groups:
                val, n = stats.get((cat, g), (float("nan"), 0))
                line += f"  {val:>4.1f}(n={n})"
            print(line)

    print()
    retr_auc, _, retr_per_cat = auc(rows, "retriever", per_cat=True)
    rerk_auc, _, rerk_per_cat = auc(rows, "reranker", per_cat=True)
    print("AUC: P(real_score > synthetic_score | same query, same category)")
    print(f"  overall         retriever={retr_auc:.3f}   reranker={rerk_auc:.3f}")
    for cat in cats:
        rt = retr_per_cat.get(cat, float("nan"))
        rr = rerk_per_cat.get(cat, float("nan"))
        print(f"  {cat:<14}  retriever={rt:.3f}   reranker={rr:.3f}")

    print()
    retr_p1, retr_p1_cats = p_at_1(rows, "retriever", per_cat=True)
    rerk_p1, rerk_p1_cats = p_at_1(rows, "reranker", per_cat=True)
    print("Precision@1 within GT category")
    print(f"  overall         retriever={retr_p1:.3f}   reranker={rerk_p1:.3f}")
    for cat in cats:
        rt = retr_p1_cats.get(cat, float("nan"))
        rr = rerk_p1_cats.get(cat, float("nan"))
        print(f"  {cat:<14}  retriever={rt:.3f}   reranker={rr:.3f}")

    print()
    print("AUC by corruption type (real vs each synthetic type)")
    by_type = {}
    for synth in SYNTH_TYPES:
        typed = [r for r in rows if r["group"] in ("real", synth)]
        rt, _ = auc(typed, "retriever")
        rr, _ = auc(typed, "reranker")
        by_type[synth] = {"retriever": rt, "reranker": rr}
        print(f"  vs {synth:<10}  retriever={rt:.3f}   reranker={rr:.3f}")

    Path("data/exp1_corruption.json").write_text(json.dumps({
        "n_queries": len(queries),
        "n_pairs": len(rows),
        "mean_rank": {
            "retriever": {f"{c}|{g}": v[0] for (c, g), v in retr_rank.items()},
            "reranker":  {f"{c}|{g}": v[0] for (c, g), v in rerk_rank.items()},
        },
        "auc_overall": {"retriever": retr_auc, "reranker": rerk_auc},
        "auc_per_cat": {"retriever": retr_per_cat, "reranker": rerk_per_cat},
        "p_at_1": {
            "retriever": retr_p1, "reranker": rerk_p1,
            "per_cat": {"retriever": retr_p1_cats, "reranker": rerk_p1_cats},
        },
        "auc_by_corruption_type": by_type,
    }, indent=2))
    print("\nSaved data/exp1_corruption.json")


if __name__ == "__main__":
    main()
