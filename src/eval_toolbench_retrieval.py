"""
Smoke test: run our retriever against StableToolBench's 765 solvable queries
and compute recall@K / NDCG@K against the gold `relevant APIs` annotation.

No reranker, no model conditioning, no wide features. Just cosine-similarity
retrieval over the 2454-endpoint ToolBench pool. The goal is to establish
that we can (a) load the catalog, (b) retrieve, (c) score against gold.
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from retriever import Retriever


REPO = Path(__file__).resolve().parents[1]
STB = REPO / "external/StableToolBench"
QUERIES_DIR = STB / "solvable_queries/test_instruction"
TOOLS_PATH = REPO / "data/toolbench_tools.json"
EMB_PATH = REPO / "data/toolbench_embeddings.json"

SPLITS = [
    "G1_instruction", "G1_category", "G1_tool",
    "G2_category", "G2_instruction", "G3_instruction",
]


def gold_api_keys(query: dict) -> set[tuple[str, str]]:
    """Return set of (tool_name, api_name) pairs marked as relevant."""
    out = set()
    for pair in query.get("relevant APIs", []):
        if len(pair) == 2:
            out.add((pair[0], pair[1]))
    return out


def server_id_for_candidate(cand: dict) -> tuple[str, str]:
    """Retriever returns (server_id, tool_name). Map back to (tool_name, api_name)
    for gold comparison. The tool_name field in our catalog is the api_name
    (because each api became a separate endpoint). The server_id encodes
    category + original tool_name."""
    server_id = cand["server_id"]
    # server_id shape: toolbench:<std_cat>:<std_tool_name>
    return server_id, cand["tool_name"]


def recall_and_ndcg_at_k(retrieved: list[dict], gold: set, tool_name_map: dict,
                         k: int) -> tuple[float, float, bool]:
    """Metrics for one query. tool_name_map: maps std tool name -> original."""
    top_k = retrieved[:k]
    matched = 0
    dcg = 0.0
    hit_any = False
    for rank, cand in enumerate(top_k):
        server_id = cand["server_id"]
        # server_id is "toolbench:<cat>:<std_tool>". Map to original tool_name.
        std_tool = server_id.split(":")[-1]
        orig_tool = tool_name_map.get(std_tool, std_tool)
        api_name = cand["tool_name"]
        key = (orig_tool, api_name)
        if key in gold:
            matched += 1
            hit_any = True
            dcg += 1.0 / math.log2(rank + 2)
    denom = min(len(gold), k)
    recall = matched / len(gold) if gold else 0.0
    idcg = sum(1.0 / math.log2(r + 2) for r in range(denom))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg, hit_any


def main():
    # Build std_tool -> orig_tool map from the catalog so we can compare
    # against gold (which uses original tool names).
    servers = json.loads(TOOLS_PATH.read_text())
    std_to_orig = {}
    for s in servers:
        std_tool = s["id"].split(":")[-1]
        orig = s["install"]["tool_name"]
        std_to_orig[std_tool] = orig

    retriever = Retriever(str(EMB_PATH), str(TOOLS_PATH))
    print(f"Retriever loaded: {len(retriever.endpoints)} endpoints\n")

    ks = [5, 10, 20, 50]
    overall = defaultdict(list)  # metric_name -> [values]

    for split in SPLITS:
        queries = json.loads((QUERIES_DIR / f"{split}.json").read_text())
        per_split = defaultdict(list)
        for q in queries:
            gold = gold_api_keys(q)
            if not gold:
                continue
            retrieved = retriever.retrieve(q["query"], top_k=max(ks))
            for k in ks:
                recall, ndcg, hit = recall_and_ndcg_at_k(retrieved, gold, std_to_orig, k)
                per_split[f"recall@{k}"].append(recall)
                per_split[f"ndcg@{k}"].append(ndcg)
                if k == 5:
                    per_split["hit@5"].append(1.0 if hit else 0.0)
        print(f"--- {split}  (n={len(queries)}) ---")
        for metric in [f"recall@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + ["hit@5"]:
            vals = per_split[metric]
            mean = sum(vals) / len(vals) if vals else 0.0
            print(f"  {metric:<12}  {mean:.3f}")
            overall[metric].extend(vals)
        print()

    print("=== Overall (all 765 solvable queries) ===")
    for metric in [f"recall@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + ["hit@5"]:
        vals = overall[metric]
        mean = sum(vals) / len(vals) if vals else 0.0
        print(f"  {metric:<12}  {mean:.3f}   (n={len(vals)})")


if __name__ == "__main__":
    main()
