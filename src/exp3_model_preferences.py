"""
Experiment 3: Model-specific tool preferences.

Two sub-questions:

  3a — Does tool success rate actually vary by calling LLM?
       Measures per-(model, tool) SR from the training feedback and quantifies
       the spread across the 6 models in MODEL_REGISTRY.

  3b — Does the trained reranker pick up on it through emb_model?
       For a fixed query, runs the reranker 6 times with different model
       indices and measures how much the top-5 ranking changes.

If 3a shows variation AND 3b shows the reranker's rankings shift with the
model, then the emb_model channel is doing its job.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from reranker import WideAndDeepModel, WideFeatures, MODEL_REGISTRY, MODEL_TO_IDX


def per_model_tool_sr(path: str = "data/feedback.jsonl", min_calls: int = 5):
    """Return {(model, tool_key): (success, n)} for pairs with n >= min_calls."""
    counts = defaultdict(lambda: [0, 0])
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "selected" not in r:
                continue
            key = f"{r['selected']['server_id']}/{r['selected']['tool_name']}"
            m = r.get("model", MODEL_REGISTRY[0])
            counts[(m, key)][1] += 1
            if r.get("rating", {}).get("success"):
                counts[(m, key)][0] += 1
    return {k: v for k, v in counts.items() if v[1] >= min_calls}


def main():
    # ----------- 3a: measured model-conditional SR spread -----------
    print("=" * 70)
    print("3a — per-model SR variation from training feedback")
    print("=" * 70)

    stats = per_model_tool_sr(min_calls=3)
    by_tool = defaultdict(dict)  # tool -> {model: (success, n)}
    for (m, tool), (s, n) in stats.items():
        by_tool[tool][m] = (s, n)

    # Pick tools observed under >=3 different models
    multi_model = {t: d for t, d in by_tool.items() if len(d) >= 3}
    print(f"{len(multi_model)} tools observed under ≥3 different LLMs (min 3 calls each)")

    # For each multi-model tool, compute SR range and std
    spreads = []
    rows = []
    for tool, d in multi_model.items():
        srs = [s / n for (s, n) in d.values()]
        sp = max(srs) - min(srs)
        spreads.append(sp)
        rows.append((tool, sp, np.std(srs), d))
    rows.sort(key=lambda x: -x[1])

    print(f"\nmean SR range across models:  {np.mean(spreads):.2f}")
    print(f"median SR range across models: {np.median(spreads):.2f}")
    print(f"fraction of tools with range >= 0.3:  "
          f"{np.mean([s >= 0.3 for s in spreads]):.2f}")
    print(f"fraction of tools with range >= 0.5:  "
          f"{np.mean([s >= 0.5 for s in spreads]):.2f}")

    print(f"\nTop 10 tools by SR spread across models (showing SR per model):")
    for tool, spread, _, d in rows[:10]:
        per_m = "  ".join(f"{m.split('/')[-1][:12]:<12}={s}/{n}" for m, (s, n) in d.items())
        print(f"  spread={spread:.2f}   {tool[:50]}")
        print(f"    {per_m}")

    # ----------- 3b: reranker ranking shifts across models -----------
    print("\n" + "=" * 70)
    print("3b — does reranker output shift with model_idx?")
    print("=" * 70)

    with open("data/embeddings.json") as f:
        endpoints = json.load(f)
    ep_embs = np.array([e["embedding"] for e in endpoints], dtype=np.float32)
    ep_keys = [f"{e['server_id']}/{e['tool_name']}" for e in endpoints]
    N = len(endpoints)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    model = WideAndDeepModel()
    model.load_state_dict(torch.load("data/models/reranker.pt", map_location="cpu",
                                     weights_only=False))
    model.eval()

    # Build wide features (shared across models — sim-cross slots depend on model
    # only through mt_sr, which is per-(model, tool))
    wide = WideFeatures()
    wide.load_from_feedback("data/feedback.jsonl")
    wide.snapshot_norms()

    # Probe queries, one per category
    probe_queries = [
        ("excel",      "Read all cells from a spreadsheet xlsx file"),
        ("pdf",        "Extract text from a PDF document"),
        ("filesystem", "Read a local text file"),
        ("wikipedia",  "Get Wikipedia article content"),
        ("search",     "Search the web for a topic"),
        ("fetch",      "Fetch the contents of a URL"),
        ("arxiv",      "Read an arxiv paper"),
    ]

    ep_embs_t = torch.tensor(ep_embs, dtype=torch.float32)

    print(f"\n{'query category':<14}  {'top-5 Jaccard across 6 models':>28}  "
          f"{'top-1 changed?':>15}  {'rank spread of pool-top-1'}")

    for cat, q in probe_queries:
        q_emb = encoder.encode([q], show_progress_bar=False)[0]
        q_t = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0).repeat(N, 1)

        # Cosine for similarity features (retriever score)
        q_norm = np.linalg.norm(q_emb) + 1e-9
        ep_norms = np.linalg.norm(ep_embs, axis=1) + 1e-9
        sims = (ep_embs @ q_emb) / (ep_norms * q_norm)

        # Per-model rankings
        top5s = []
        top1s = []
        rank_of_top = None  # rank of the FIRST model's top-1 across all models
        per_model_rank_of_fixed_tool = []
        for m_idx, m_name in enumerate(MODEL_REGISTRY):
            wide_feats = np.zeros((N, 6), dtype=np.float32)
            for i, k in enumerate(ep_keys):
                usage = wide.usage_count[k]
                max_u = max(wide.usage_count.values(), default=1) or 1
                norm_u = usage / max_u if max_u else 0.0
                sr = wide.success_count[k] / usage if usage > 0 else 0.0
                mt_key = f"{m_name}||{k}"
                mtu = wide.model_tool_usage[mt_key]
                mtsr = wide.model_tool_success[mt_key] / mtu if mtu > 0 else 0.0
                s = sims[i]
                wide_feats[i] = [norm_u, sr, mtsr, s, s * sr, s * mtsr]
            wide_t = torch.tensor(wide_feats, dtype=torch.float32)
            model_idx_t = torch.full((N,), m_idx, dtype=torch.long)
            with torch.no_grad():
                logits = model(wide_t, model_idx_t, q_t, ep_embs_t).squeeze(-1).numpy()
            order = np.argsort(-logits)
            top5 = [ep_keys[i] for i in order[:5]]
            top5s.append(set(top5))
            top1s.append(top5[0])
            if m_idx == 0:
                fixed = order[0]
            per_model_rank_of_fixed_tool.append(int(np.where(order == fixed)[0][0]))

        # Jaccard across all 6 top-5 sets
        union = set().union(*top5s)
        inter = set.intersection(*top5s)
        jaccard = len(inter) / len(union) if union else 0.0
        top1_changes = len(set(top1s))  # 1 = same across all 6 models
        print(f"{cat:<14}  {jaccard:>28.2f}  {top1_changes:>15}  "
              f"{min(per_model_rank_of_fixed_tool)}–{max(per_model_rank_of_fixed_tool)}")

    # Save
    out = {"per_model_sr_spread": {"mean": float(np.mean(spreads)),
                                    "median": float(np.median(spreads)),
                                    "frac_gte_0.3": float(np.mean([s >= 0.3 for s in spreads])),
                                    "frac_gte_0.5": float(np.mean([s >= 0.5 for s in spreads])),
                                    "n_tools": len(multi_model)}}
    Path("data/exp3_model_preferences.json").write_text(json.dumps(out, indent=2))
    print("\nSaved data/exp3_model_preferences.json")


if __name__ == "__main__":
    main()
