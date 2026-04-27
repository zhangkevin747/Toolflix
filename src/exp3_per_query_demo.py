"""
Experiment 3 concrete demo: for a specific query, show that the set of
tools that actually work depends on the calling LLM.

Strategy:
  1. Pick one query.
  2. For each (model, tool) observed in feedback for that task family,
     show per-model SR from the training data.
  3. Run the reranker 6 times on the query, one per model, and show the
     top-5 rankings. The ranking shift is the reranker's learned response
     to the per-model SR differences.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from reranker import WideAndDeepModel, WideFeatures, MODEL_REGISTRY


# Zoom in on "fetch" — exp3a showed the biggest per-model spread there
# (fetch-synth-truncated-001: gpt-5.4-nano 9% vs gemini 100%).
QUERY = "Fetch the HTML content of a web page from a given URL"
QUERY_CATEGORY = "fetch"

# Tools of interest (observed with per-model spread in exp3a output)
TOOLS_OF_INTEREST = [
    "fetch-mcp/fetch_url",
    "fetch-synth-truncated-001/fetch_url",
    "fetch-synth-hollow-002/fetch_url",
    "fetch-synth-flaky-003/fetch_url",
    "tokenizin-mcp-npx-fetch/fetch_html",
    "mcp-server-fetch/fetch",
    "fetcher-mcp/fetch_url",
    "zcaceres-fetch-mcp/fetch_readable",
    "tavily-mcp/tavily_extract",
]


def per_model_sr_for_tools(path="data/feedback.jsonl",
                           tools=TOOLS_OF_INTEREST,
                           task_category=QUERY_CATEGORY):
    """Per-model success rate for each tool, across fetch tasks."""
    cnt = defaultdict(lambda: [0, 0])
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("category") != task_category:
                continue
            sel = r.get("selected", {})
            key = f"{sel.get('server_id','?')}/{sel.get('tool_name','?')}"
            if key not in tools:
                continue
            m = r.get("model", MODEL_REGISTRY[0])
            cnt[(m, key)][1] += 1
            if r.get("rating", {}).get("success"):
                cnt[(m, key)][0] += 1
    return cnt


def main():
    print(f'Query: "{QUERY}"   (category: {QUERY_CATEGORY})\n')

    # ---------- per-model SR table ----------
    cnt = per_model_sr_for_tools()
    print("Observed per-model SR on fetch tasks in training:")
    print(f"  {'tool':<50}  " + "  ".join(f"{m.split('/')[-1][:10]:>10}" for m in MODEL_REGISTRY))
    for tool in TOOLS_OF_INTEREST:
        row = f"  {tool:<50}  "
        for m in MODEL_REGISTRY:
            s, n = cnt.get((m, tool), (0, 0))
            row += f"  {'---':>8}" if n == 0 else f"  {s}/{n:<4}={100*s/n:>3.0f}%"
        print(row)

    # ---------- reranker per-model top-5 for this query ----------
    with open("data/embeddings.json") as f:
        endpoints = json.load(f)
    ep_embs = np.array([e["embedding"] for e in endpoints], dtype=np.float32)
    ep_keys = [f"{e['server_id']}/{e['tool_name']}" for e in endpoints]
    N = len(endpoints)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    model = WideAndDeepModel()
    model.load_state_dict(torch.load("data/models/reranker.pt",
                                     map_location="cpu", weights_only=False))
    model.eval()

    wide = WideFeatures()
    wide.load_from_feedback("data/feedback.jsonl")
    wide.snapshot_norms()

    q_emb = encoder.encode([QUERY], show_progress_bar=False)[0]
    q_t = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
    q_norm = np.linalg.norm(q_emb) + 1e-9
    ep_norms = np.linalg.norm(ep_embs, axis=1) + 1e-9
    sims = (ep_embs @ q_emb) / (ep_norms * q_norm)
    ep_embs_t = torch.tensor(ep_embs, dtype=torch.float32)

    print("\nReranker top-5 per calling LLM for the same query:")
    tool_ranks_by_model = {}
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
        top5 = [(ep_keys[i], float(logits[i])) for i in order[:5]]
        print(f"\n  {m_name}:")
        for rank, (k, s) in enumerate(top5, 1):
            print(f"    {rank}. {k:<55}  score={s:.3f}")
        # Also dump where each tool_of_interest landed for this model
        rank_map = {ep_keys[j]: int(np.where(order == j)[0][0]) + 1 for j in range(N)}
        tool_ranks_by_model[m_name] = {t: rank_map.get(t, "?") for t in TOOLS_OF_INTEREST}

    # ---------- ranking of each tool_of_interest across all 6 models ----------
    print("\nRanking of each watched tool across the 6 models:")
    print(f"  {'tool':<50}  " + "  ".join(f"{m.split('/')[-1][:10]:>10}" for m in MODEL_REGISTRY))
    for tool in TOOLS_OF_INTEREST:
        row = f"  {tool:<50}  "
        for m in MODEL_REGISTRY:
            r = tool_ranks_by_model[m].get(tool, "?")
            row += f"  {str(r):>10}"
        print(row)

    Path("data/exp3_per_query_demo.json").write_text(json.dumps({
        "query": QUERY,
        "tool_ranks_by_model": tool_ranks_by_model,
        "per_model_sr_observed": {f"{m}||{t}": list(cnt[(m, t)])
                                  for (m, t) in cnt}
    }, indent=2))
    print("\nSaved data/exp3_per_query_demo.json")


if __name__ == "__main__":
    main()
