"""
Forced-tool × model grid for GAIA probes.

For each GAIA task, pick N candidate tools and force every model to try
each one. Success is measured by string-match on the expected answer.

The payoff: a cell grid (tool × model) per task showing where the answer
was actually retrievable, from which we can identify tasks where tool X
works for model A but not model B — and some different tool Y is the only
path for model B.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from reranker import MODEL_REGISTRY
from mcp_client import MCPClient
from agent import Agent


TOOLS_META = {s["id"]: s for s in json.load(open("data/tools.json"))}


# (task_id, [candidate tool keys to force])
PROBES = [
    ("arxiv-05", [
        "arxiv-latex-mcp/get_paper_abstract",
        "arxiv-latex-mcp/get_paper_prompt",
        "arxiv-latex-mcp/list_paper_sections",
        "arxiv-mcp-blazick/read_paper",
        "pdf-reader-mcp/read_pdf",
    ]),
    ("pdf-03", [
        "fabriqa-pdf-reader/read-pdf",
        "fabriqa-pdf-reader/pdf-metadata",
        "pdf-reader-mcp/read_pdf",
        "mcp-server-filesystem/read_text_file",
    ]),
    ("search-05", [
        "tavily-mcp/tavily_search",
        "serper-mcp/google_search",
        "fetcher-mcp/fetch_url",
        "tokenizin-mcp-npx-fetch/fetch_markdown",
        "wikipedia-mcp-rudra/search_wikipedia",
    ]),
    ("wikipedia-06", [
        "wikipedia-mcp-rudra/get_article",
        "wikipedia-mcp-rudra/search_wikipedia",
        "wikipedia-mcp-rudra/get_sections",
        "fetcher-mcp/fetch_url",
        "tokenizin-mcp-npx-fetch/fetch_markdown",
    ]),
]


def match(expected: str, text: str) -> bool:
    return bool(expected) and bool(text) and expected.lower() in text.lower()


def force_call(task: dict, tool_key: str, model_name: str) -> dict:
    """Have the model fill arguments for the forced tool, execute, grade."""
    server_id, tool_name = tool_key.split("/", 1)
    server = TOOLS_META.get(server_id)
    if not server:
        return {"success": False, "error": "unknown server"}

    # Find the tool schema
    tool_schema = next((t for t in server.get("tools", []) if t["name"] == tool_name), None)
    if not tool_schema:
        return {"success": False, "error": "unknown tool"}

    agent = Agent(model=model_name)

    # Use call_2_select_and_call machinery but with exactly one forced candidate.
    forced = [{
        "server_id": server_id,
        "tool_name": tool_name,
        "description": tool_schema.get("description", ""),
        "inputSchema": tool_schema.get("inputSchema", {}),
        "similarity": 1.0,
        "install": server.get("install", {}),
    }]

    try:
        sel = agent.call_2_select_and_call(
            task["task"], task["task"], forced, {}
        )
    except Exception as e:
        return {"success": False, "error": f"arg_fill: {e}"}

    args = sel.get("arguments", {}) or {}
    for k, v in list(args.items()):
        if isinstance(v, str) and v.startswith("data/"):
            args[k] = os.path.abspath(v)

    mcp = MCPClient(timeout=45)
    try:
        result = mcp.call_tool(
            server_id=server_id,
            tool_name=tool_name,
            arguments=args,
            install=server.get("install", {}),
        )
    except Exception as e:
        return {"success": False, "error": f"mcp: {e}"}

    out_text = json.dumps(result, default=str)
    return {
        "success": match(task["expected_answer"], out_text),
        "args": args,
        "result_preview": out_text[:200],
    }


def main():
    all_tasks = {t["id"]: t for t in json.load(open("data/gaia_gt.json"))}

    jobs = []
    for tid, tool_keys in PROBES:
        task = all_tasks.get(tid)
        if not task:
            continue
        for tool_key in tool_keys:
            for m in MODEL_REGISTRY:
                jobs.append((tid, tool_key, m, task))

    print(f"Running {len(jobs)} forced-tool trials "
          f"({len(PROBES)} tasks × ~4.5 tools × {len(MODEL_REGISTRY)} models)\n")

    results = {}  # (tid, tool_key, model) -> {success, ...}

    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(force_call, task, tool, m): (tid, tool, m)
                for (tid, tool, m, task) in jobs}
        done = 0
        for fut in as_completed(futs):
            tid, tool, m = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"success": False, "error": str(e)}
            results[(tid, tool, m)] = r
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}  elapsed={time.time()-start:.0f}s",
                      flush=True)
    print(f"\nelapsed {time.time()-start:.0f}s")

    # ---------- grid per task ----------
    print("\n\n======= tool × model success grid per task =======")
    for tid, tool_keys in PROBES:
        task = all_tasks.get(tid)
        if not task:
            continue
        print(f"\n[{tid}]  {task['task'][:90]}")
        print(f"  expected: {task['expected_answer'][:60]}")
        # Header
        short_models = [m.split('/')[-1][:10] for m in MODEL_REGISTRY]
        print(f"  {'tool':<50}  " + "  ".join(f"{m:>10}" for m in short_models))
        for tool in tool_keys:
            row_cells = []
            for m in MODEL_REGISTRY:
                r = results.get((tid, tool, m), {})
                if "error" in r and not r.get("success"):
                    cell = "ERR"
                else:
                    cell = "✓" if r.get("success") else "✗"
                row_cells.append(cell)
            print(f"  {tool:<50}  " + "  ".join(f"{c:>10}" for c in row_cells))

    # ---------- divergence summary ----------
    print("\n\n======= divergence findings =======")
    for tid, tool_keys in PROBES:
        task = all_tasks.get(tid)
        if not task:
            continue
        # Which (model, tool) cells succeed?
        per_model_tools = defaultdict(list)
        for tool in tool_keys:
            for m in MODEL_REGISTRY:
                r = results.get((tid, tool, m), {})
                if r.get("success"):
                    per_model_tools[m].append(tool)
        # For each model, which tools worked?
        print(f"\n[{tid}]")
        for m in MODEL_REGISTRY:
            short = m.split('/')[-1][:18]
            works = per_model_tools.get(m, [])
            print(f"  {short:<20}  succeeded with: {works if works else 'NOTHING'}")

        # Find a pair of models with disjoint working tools
        pairs = []
        models_list = list(MODEL_REGISTRY)
        for i, a in enumerate(models_list):
            for b in models_list[i+1:]:
                aw = set(per_model_tools.get(a, []))
                bw = set(per_model_tools.get(b, []))
                if aw and bw and not (aw & bw):
                    pairs.append((a, b, aw, bw))
        if pairs:
            print(f"  --> {len(pairs)} model pairs with DISJOINT working tools")
            for a, b, aw, bw in pairs[:3]:
                print(f"      {a.split('/')[-1]:<18} succeeds only via: {list(aw)}")
                print(f"      {b.split('/')[-1]:<18} succeeds only via: {list(bw)}")

    # Save
    serialized = {f"{k[0]}||{k[1]}||{k[2]}": v for k, v in results.items()}
    Path("data/exp3_forced_tool_grid.json").write_text(json.dumps(serialized, indent=2))
    print("\nSaved data/exp3_forced_tool_grid.json")


if __name__ == "__main__":
    main()
