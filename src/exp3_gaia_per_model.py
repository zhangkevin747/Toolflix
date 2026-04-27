"""
Pick a handful of GAIA tasks, run each through the full pipeline (retriever
+ reranker + MCP call + judge) once per LLM in MODEL_REGISTRY. Show whether
the same task produces divergent tool-pick / success outcomes across models.

This is the direct empirical form of experiment 3: not inferred from
aggregate SR, but observed trial-by-trial on identical GAIA inputs.
"""
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from retriever import Retriever
from reranker import Reranker, MODEL_REGISTRY
from mcp_client import MCPClient
from agent import Agent


PROBE_IDS = [
    "excel-05",       # "Open the spreadsheet ...xlsx and show me what's inside"
    "pdf-03",         # Something from the pdf cluster
    "fetch-07",       # Fetch a URL
    "search-05",      # Search task
    "wikipedia-06",   # Wikipedia lookup
    "filesystem-04",  # Read a local file
    "arxiv-05",       # Arxiv paper
]


def match(expected: str, output: str) -> bool:
    return bool(expected) and bool(output) and expected.lower() in output.lower()


def run_one(task: dict, model_name: str) -> dict:
    retriever = Retriever(
        embeddings_path="data/embeddings.json",
        tools_path="data/tools.json",
    )
    reranker = Reranker(
        embeddings_path="data/embeddings.json",
        feedback_path="data/feedback.jsonl",
        model_path="data/models/reranker.pt",
        model_name=model_name,
    )
    agent = Agent(model=model_name)
    mcp = MCPClient(timeout=45)

    task_text = task["task"]
    query = agent.call_1_decompose(task_text)
    cands = retriever.retrieve(query, top_k=100)
    top5 = reranker.rerank(cands, query, top_k=5)

    # example calls from prior successes
    examples = {}
    sel = agent.call_2_select_and_call(task_text, query, top5, examples)
    tool_idx = max(0, min(int(sel.get("tool_index", 1)) - 1, len(top5) - 1))
    chosen = top5[tool_idx]
    args = sel.get("arguments", {})
    # resolve data/ paths
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("data/"):
            args[k] = os.path.abspath(v)

    result = mcp.call_tool(
        server_id=chosen["server_id"],
        tool_name=chosen["tool_name"],
        arguments=args,
        install=chosen["install"],
    )
    out_text = json.dumps(result, default=str)
    answer_found = match(task["expected_answer"], out_text)

    return {
        "task_id": task["id"],
        "model": model_name,
        "query": query,
        "picked_tool": f"{chosen['server_id']}/{chosen['tool_name']}",
        "top1_correct_category": top5[0].get("server_id", "").split("-")[0] in task["expected_category"] or task["expected_category"] in top5[0].get("server_id", ""),
        "answer_found": bool(answer_found),
        "top5": [f"{c['server_id']}/{c['tool_name']}" for c in top5],
    }


def main():
    all_tasks = {t["id"]: t for t in json.load(open("data/gaia_gt.json"))}
    probes = [all_tasks[i] for i in PROBE_IDS if i in all_tasks]
    print(f"Running {len(probes)} GAIA probes × {len(MODEL_REGISTRY)} models "
          f"= {len(probes) * len(MODEL_REGISTRY)} pipeline runs\n")

    rows = []
    jobs = [(t, m) for t in probes for m in MODEL_REGISTRY]
    start = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, t, m): (t["id"], m) for (t, m) in jobs}
        for fut in as_completed(futs):
            tid, m = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"task_id": tid, "model": m, "error": str(e)}
            rows.append(row)
            print(f"  [{row['task_id']} | {m.split('/')[-1][:14]:<14}] "
                  f"ans={'✓' if row.get('answer_found') else '✗'}  "
                  f"picked={row.get('picked_tool','ERR')}", flush=True)

    print(f"\nelapsed {time.time()-start:.1f}s")

    # ---------- cross-tabulation ----------
    print("\n\n======= TASK x MODEL matrix (picked tool / success) =======\n")
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["model"]] = r

    for tid in PROBE_IDS:
        task = all_tasks.get(tid)
        if not task:
            continue
        print(f"\n[{tid}]  {task['task'][:100]}")
        print(f"  expected: {task['expected_answer'][:60]}")
        for m in MODEL_REGISTRY:
            r = by_task.get(tid, {}).get(m, {})
            mark = "✓" if r.get("answer_found") else ("ERR" if "error" in r else "✗")
            print(f"    {mark:>3}  {m.split('/')[-1][:22]:<22}  "
                  f"{r.get('picked_tool','n/a')}")

    # ---------- divergence summary ----------
    print("\n\n======= divergence per task =======")
    for tid in PROBE_IDS:
        rs = by_task.get(tid, {})
        picks = {m: r.get("picked_tool") for m, r in rs.items()}
        oks = {m: r.get("answer_found") for m, r in rs.items()}
        n_distinct = len(set(picks.values()))
        n_ok = sum(1 for v in oks.values() if v)
        n_total = len([v for v in oks.values() if v is not None])
        print(f"  {tid:<14}  distinct_tools_picked={n_distinct}/6   "
              f"answer_found={n_ok}/{n_total}")

    Path("data/exp3_gaia_per_model.json").write_text(json.dumps(rows, indent=2))
    print("\nSaved data/exp3_gaia_per_model.json")


if __name__ == "__main__":
    main()
