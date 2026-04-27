"""
Oracle evaluation: for each GAIA-bench item, bypass retriever/reranker and
feed the agent only the tools in the task's expected_category. This measures
the ceiling of our pipeline — "if routing is perfect, can the agent solve
this task?"

A low oracle ceiling means the benchmark has tasks that are unsolvable even
with perfect tool selection (bad benchmark). A high ceiling means the
routing system has real headroom to chase.

Run:
    python src/oracle_eval.py               # all items
    python src/oracle_eval.py --limit 15    # first 15
    python src/oracle_eval.py --category arxiv
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from mcp_client import MCPClient
from agent import Agent
from eval_gt import answer_match, category_of
from retriever import BLOCKED_ENDPOINTS


BENCH_PATH = Path(__file__).parent.parent / "data/gaia_bench.json"
TOOLS_PATH = Path(__file__).parent.parent / "data/tools.json"


def load_oracle_candidates(expected_category: str) -> list[dict]:
    """Return all tools whose server has category == expected_category, in a
    form the agent can consume via call_2_select_and_call."""
    servers = json.loads(TOOLS_PATH.read_text())
    out = []
    for s in servers:
        if s.get("category") != expected_category:
            continue
        if "synth" in s["id"]:
            continue  # oracle means REAL tools only
        for t in s.get("tools", []):
            if (s["id"], t["name"]) in BLOCKED_ENDPOINTS:
                continue
            out.append({
                "server_id": s["id"],
                "server_name": s.get("name", s["id"]),
                "tool_name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {}),
                "install": s.get("install", {}),
                "similarity": 1.0,
            })
    return out


def resolve_paths(args: dict) -> dict:
    """Convert repo-relative paths like 'data/...' or '/data/...' to absolute.
    Recursively walks dicts and lists so nested values (e.g. sources: [{path: ...}]) work."""
    def fix(v):
        if isinstance(v, str):
            s = v.lstrip("/")
            if s.startswith("data/"):
                return os.path.abspath(s)
            return v
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v
    return {k: fix(v) for k, v in args.items()}


def run_one(item: dict, model_name: str, mcp_client: MCPClient) -> dict:
    agent = Agent(model=model_name)
    task = item["question"]
    cat = item.get("expected_category")

    if cat is None:
        # no_tool item — just let agent answer directly
        reported = agent.call_4_answer(task, None)
        ok = answer_match(reported, item.get("ground_truth", []))
        return {
            "id": item["id"],
            "category": None,
            "picked_tool": None,
            "reported": reported,
            "answer_match": ok,
            "both_correct": ok,
        }

    candidates = load_oracle_candidates(cat)
    if not candidates:
        return {
            "id": item["id"], "category": cat, "error": "no candidates for category",
            "answer_match": False, "both_correct": False,
        }

    # Use existing agent machinery: decompose + select-from-limited-set
    query = agent.call_1_decompose(task)
    # Give agent top-5 of category (or all if < 5) so it still picks among real siblings
    top_n = candidates[:5]
    sel = agent.call_2_select_and_call(task, query, top_n, {})
    tool_idx = max(0, min(int(sel.get("tool_index", 1)) - 1, len(top_n) - 1))
    chosen = top_n[tool_idx]
    args = resolve_paths(sel.get("arguments", {}))

    try:
        tool_result = mcp_client.call_tool(
            server_id=chosen["server_id"],
            tool_name=chosen["tool_name"],
            arguments=args,
            install=chosen["install"],
        )
    except Exception as e:
        tool_result = {"error": str(e), "response": ""}

    reported = agent.call_4_answer(task, tool_result)
    ans_ok = answer_match(reported, item.get("ground_truth", []))

    return {
        "id": item["id"],
        "category": cat,
        "picked_tool": f"{chosen['server_id']}/{chosen['tool_name']}",
        "args": args,
        "reported": reported,
        "tool_result_preview": str(tool_result)[:200],
        "answer_match": ans_ok,
        "both_correct": ans_ok,  # tool_match is guaranteed in oracle
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--model", type=str, default="gpt-5.4-nano")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    items = json.loads(BENCH_PATH.read_text())
    if args.category:
        items = [i for i in items if (i.get("expected_category") or "no_tool") == args.category]
    if args.limit:
        items = items[:args.limit]
    print(f"Oracle eval over {len(items)} items, model={args.model}")

    mcp = MCPClient(timeout=45)
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, it, args.model, mcp): it for it in items}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                it = futs[fut]
                r = {"id": it["id"], "category": it.get("expected_category"),
                     "error": str(e), "answer_match": False, "both_correct": False}
            results.append(r)
            done += 1
            mark = "✓" if r.get("both_correct") else "✗"
            print(f"  [{done:>3}/{len(items)}] {mark} {r['id']:<25}  cat={(r.get('category') or '-'):<10}  "
                  f"reported={str(r.get('reported',''))[:50]!r}", flush=True)

    print(f"\nelapsed: {time.time()-t0:.0f}s")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.get("category") or "no_tool"].append(r)

    print(f"\n{'category':<12}  {'n':>4}  {'solved':>6}")
    for cat in sorted(by_cat.keys()):
        n = len(by_cat[cat])
        ok = sum(1 for r in by_cat[cat] if r.get("both_correct"))
        print(f"  {cat:<12}  {n:>4}  {ok:>3}/{n}  ({100*ok/n:>4.0f}%)")
    total = len(results)
    total_ok = sum(1 for r in results if r.get("both_correct"))
    print(f"  {'OVERALL':<12}  {total:>4}  {total_ok:>3}/{total}  ({100*total_ok/total:>4.0f}%)")

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2, default=str))
        print(f"\n-> {args.save}")


if __name__ == "__main__":
    main()
