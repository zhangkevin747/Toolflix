"""
Finish the pending items from a partial eval log. Parses completed IDs from
a log file and runs only the items that didn't complete, with a per-item
process-level timeout so no item can hang forever.

Writes results (for both modes) to data/exp_a_results.json, merging in the
partially-completed retriever+reranker data we captured from the main eval.
"""
import argparse
import json
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from eval_gt import GTEvaluator, summarize

BENCH = REPO / "data/gaia_bench.json"


def run_with_timeout(fn, args_list, seconds):
    """Run fn(*args) with a hard wall-clock timeout using a sentinel."""
    import threading
    result = {"val": None, "err": None}
    def _worker():
        try:
            result["val"] = fn(*args_list)
        except Exception as e:
            result["err"] = e
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"call exceeded {seconds}s")
    if result["err"]:
        raise result["err"]
    return result["val"]


def run_mode(items, use_reranker, label, item_timeout=90):
    ev = GTEvaluator(use_reranker=use_reranker)
    results = []
    t0 = time.time()

    def _do(item):
        if item.get("expected_category") is None:
            return ev.run_notool(item, "gpt-5.4-nano")
        return ev.run_one(item, "gpt-5.4-nano")

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_with_timeout, _do, [it], item_timeout): it for it in items}
        done = 0
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "id": it["id"],
                    "expected_category": it.get("expected_category"),
                    "error": str(e),
                    "picked_tool": None, "picked_category": None,
                    "reported_answer": "",
                    "tool_match": False, "answer_match": False,
                    "both_correct": False,
                }
            results.append(r)
            done += 1
            mark = "✓" if r.get("both_correct") else "✗"
            print(f"  [{done}/{len(items)}] {mark} {r.get('id','')} "
                  f"cat={(r.get('expected_category') or '-')}  "
                  f"tool={r.get('picked_tool','')}", flush=True)
    print(f"  [{label}] elapsed {time.time()-t0:.0f}s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="/tmp/exp_a2.log",
                    help="Completed-items log to skip")
    ap.add_argument("--save", type=str,
                    default=str(REPO / "data/exp_a_results.json"))
    ap.add_argument("--timeout", type=int, default=90,
                    help="Per-item timeout in seconds")
    args = ap.parse_args()

    log = Path(args.log).read_text() if Path(args.log).exists() else ""
    done_ids = set(re.findall(r"(?:✓|✗) (\S+)", log))
    items = json.loads(BENCH.read_text())
    pending = [x for x in items if x["id"] not in done_ids]
    print(f"Already done (from log): {len(done_ids)}")
    print(f"Pending: {len(pending)}")
    print(f"  by cat: {[(x['id'], x.get('expected_category') or 'no_tool') for x in pending]}")

    # Run both modes on pending only
    print("\n=== RETRIEVER (pending) ===")
    results_r_pend = run_mode(pending, False, "retriever", args.timeout)
    print("\n=== RERANKER (pending) ===")
    results_rr_pend = run_mode(pending, True, "reranker", args.timeout)

    out = {
        "retriever_pending": results_r_pend,
        "reranker_pending": results_rr_pend,
    }
    Path(args.save).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved -> {args.save}")

    print("\n=== SUMMARY (pending only, retriever) ===")
    summarize("RETRIEVER (pending)", results_r_pend)
    print("\n=== SUMMARY (pending only, reranker) ===")
    summarize("RERANKER (pending)", results_rr_pend)


if __name__ == "__main__":
    main()
