"""
Ground-truth GAIA evaluator.

For each item in data/gaia_bench.json:
  1. Run pipeline: decompose -> retrieve -> (rerank) -> select -> call
  2. Extract a final agent-reported answer from the tool output.
  3. Grade three metrics:
        tool_match    : first tool call's category equals expected_category
                        (or expected_category is null and no tool was called)
        answer_match  : normalized exact match on any ground_truth alternate
        both_correct  : tool_match AND answer_match (headline metric)

Run:
    python src/eval_gt.py --mode retriever
    python src/eval_gt.py --mode reranker
    python src/eval_gt.py --mode both
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from retriever import Retriever
from reranker import Reranker, MODEL_REGISTRY
from mcp_client import MCPClient
from agent import Agent


BENCH_PATH = Path(__file__).parent.parent / "data/gaia_bench.json"
TOOLS_PATH = Path(__file__).parent.parent / "data/tools.json"


_server_to_category: dict[str, str] = {}


def _init_category_lookup():
    if _server_to_category:
        return
    for s in json.loads(TOOLS_PATH.read_text()):
        _server_to_category[s["id"]] = s.get("category", "")


def category_of(server_id: str) -> str:
    _init_category_lookup()
    return _server_to_category.get(server_id, "")


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.strip(" \"'`.,;:!?")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s


def answer_match(reported: str, ground_truths: list[str]) -> bool:
    norm_rep = _normalize(reported)
    if not norm_rep:
        return False
    for gt in ground_truths:
        norm_gt = _normalize(gt)
        if not norm_gt:
            continue
        if norm_rep == norm_gt:
            return True
        if len(norm_gt) >= 4 and norm_gt in norm_rep:
            return True
    return False


class GTEvaluator:
    def __init__(self, use_reranker: bool, round_robin: bool = True):
        self.use_reranker = use_reranker
        self.round_robin = round_robin

        self.retriever = Retriever(
            embeddings_path="data/embeddings.json",
            tools_path="data/tools.json",
        )
        self.reranker = None
        if use_reranker:
            model_path = Path("data/models/reranker.pt")
            self.reranker = Reranker(
                embeddings_path="data/embeddings.json",
                feedback_path="data/feedback.jsonl",
                model_path=str(model_path) if model_path.exists() else None,
                model_name=MODEL_REGISTRY[0],
            )
        self.mcp_client = MCPClient(timeout=45)
        self._lock = threading.Lock()

    def _resolve_paths(self, args: dict) -> dict:
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

    def run_one(self, item: dict, model_name: str) -> dict:
        agent = Agent(model=model_name)
        task = item["question"]
        query = agent.call_1_decompose(task)
        cands = self.retriever.retrieve(query, top_k=50)

        if self.reranker is not None:
            with self._lock:
                cands = self.reranker.rerank(cands, query, top_k=5)
        else:
            cands = cands[:5]

        sel = agent.call_2_select_and_call(task, query, cands, {})
        tool_idx = max(0, min(int(sel.get("tool_index", 1)) - 1, len(cands) - 1))
        chosen = cands[tool_idx]
        args = self._resolve_paths(sel.get("arguments", {}))

        tool_result = None
        try:
            tool_result = self.mcp_client.call_tool(
                server_id=chosen["server_id"],
                tool_name=chosen["tool_name"],
                arguments=args,
                install=chosen["install"],
            )
        except Exception as e:
            tool_result = {"error": str(e), "response": ""}

        reported = agent.call_4_answer(task, tool_result)

        first_cat = category_of(chosen["server_id"])
        expected_cat = item.get("expected_category")
        if expected_cat is None:
            tool_match_ok = False
        else:
            tool_match_ok = (first_cat == expected_cat)

        ans_ok = answer_match(reported, item.get("ground_truth", []))

        return {
            "id": item["id"],
            "model": model_name,
            "expected_category": expected_cat,
            "picked_tool": f"{chosen['server_id']}/{chosen['tool_name']}",
            "picked_category": first_cat,
            "reported_answer": reported,
            "tool_match": tool_match_ok,
            "answer_match": ans_ok,
            "both_correct": bool(tool_match_ok and ans_ok),
        }

    def run_notool(self, item: dict, model_name: str) -> dict:
        agent = Agent(model=model_name)
        reported = agent.call_4_answer(item["question"], None)
        ans_ok = answer_match(reported, item.get("ground_truth", []))
        return {
            "id": item["id"],
            "model": model_name,
            "expected_category": None,
            "picked_tool": None,
            "picked_category": None,
            "reported_answer": reported,
            "tool_match": True,
            "answer_match": ans_ok,
            "both_correct": ans_ok,
        }

    def evaluate(self, items: list[dict]) -> list[dict]:
        results = []
        models = MODEL_REGISTRY if self.round_robin else [MODEL_REGISTRY[0]]
        jobs = [(item, models[i % len(models)]) for i, item in enumerate(items)]

        def _do(job):
            item, m = job
            if item.get("expected_category") is None:
                return self.run_notool(item, m)
            return self.run_one(item, m)

        def _do_with_timeout(job, seconds=90):
            result = {"val": None, "err": None}
            def _worker():
                try:
                    result["val"] = _do(job)
                except Exception as e:
                    result["err"] = e
            th = threading.Thread(target=_worker, daemon=True)
            th.start()
            th.join(seconds)
            if th.is_alive():
                raise TimeoutError(f"item timed out after {seconds}s")
            if result["err"]:
                raise result["err"]
            return result["val"]

        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_do_with_timeout, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    item, m = futs[fut]
                    r = {
                        "id": item["id"], "model": m,
                        "expected_category": item.get("expected_category"),
                        "error": str(e),
                        "tool_match": False, "answer_match": False,
                        "both_correct": False,
                    }
                done += 1
                mark = "✓" if r.get("both_correct") else "✗"
                tool = (r.get("picked_tool") or "")[:40]
                print(f"  [{done:>3}/{len(jobs)}] {mark} {r.get('id',''):<25}  "
                      f"cat={(r.get('expected_category') or '-'):<10}  "
                      f"tool={tool!r:<42}", flush=True)
                results.append(r)
        return results


def summarize(label: str, results: list[dict]):
    by_cat = defaultdict(list)
    for r in results:
        cat = r.get("expected_category") or "no_tool"
        by_cat[cat].append(r)

    print(f"\n=== {label} ===")
    def stats(subset):
        if not subset:
            return (0, 0.0, 0.0, 0.0)
        n = len(subset)
        t = sum(1 for r in subset if r.get("tool_match"))
        a = sum(1 for r in subset if r.get("answer_match"))
        b = sum(1 for r in subset if r.get("both_correct"))
        return (n, t / n, a / n, b / n)

    print(f"  {'category':<12}  {'n':>4}  {'tool_match':>10}  {'answer_match':>12}  {'both':>5}")
    for cat in sorted(by_cat.keys()):
        n, t, a, b = stats(by_cat[cat])
        print(f"  {cat:<12}  {n:>4}  {t*100:>9.1f}%  {a*100:>11.1f}%  {b*100:>4.1f}%")
    n, t, a, b = stats(results)
    print(f"  {'OVERALL':<12}  {n:>4}  {t*100:>9.1f}%  {a*100:>11.1f}%  {b*100:>4.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["retriever", "reranker", "both"],
                   default="both")
    p.add_argument("--items", type=str, default=str(BENCH_PATH))
    p.add_argument("--save", type=str, default=None,
                   help="Write detailed results JSON here")
    args = p.parse_args()

    items = json.loads(Path(args.items).read_text())
    print(f"Loaded {len(items)} items from {args.items}")

    out = {}

    if args.mode in ("retriever", "both"):
        print("\nRunning retriever-only...")
        ev = GTEvaluator(use_reranker=False)
        t0 = time.time()
        results_r = ev.evaluate(items)
        print(f"  elapsed: {time.time() - t0:.0f}s")
        summarize("RETRIEVER", results_r)
        out["retriever"] = results_r

    if args.mode in ("reranker", "both"):
        print("\nRunning retriever + reranker...")
        ev = GTEvaluator(use_reranker=True)
        t0 = time.time()
        results_rr = ev.evaluate(items)
        print(f"  elapsed: {time.time() - t0:.0f}s")
        summarize("RETRIEVER + RERANKER", results_rr)
        out["reranker"] = results_rr

    if args.save:
        Path(args.save).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n-> {args.save}")


if __name__ == "__main__":
    main()
