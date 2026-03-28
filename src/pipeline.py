"""
The Agentic Loop — Pipeline

Runs tasks through the 3-call agent loop:
1. Agent receives a task, decomposes it, outputs a query
2. Retriever returns top k tools, agent selects and calls one
3. Agent rates the tool

Each task thus requires 3 agent calls.
Feedback is stored as JSONL.
"""
import json
import os
import shutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

# Load env vars from .env
load_dotenv(Path(__file__).parent.parent / ".env")

from retriever import Retriever
from reranker import Reranker
from mcp_client import MCPClient
from agent import Agent


class Pipeline:
    """Runs the full agentic loop over a set of tasks."""

    def __init__(self, data_dir: str = "../data", retrieve_k: int = 10,
                 rerank_k: int = 5, use_reranker: bool = True,
                 batch_train_every: int = 50, concurrency: int = 1):
        self.data_dir = Path(data_dir)
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k
        self.use_reranker = use_reranker
        self.batch_train_every = batch_train_every
        self.concurrency = concurrency

        self.retriever = Retriever(
            embeddings_path=str(self.data_dir / "embeddings.json"),
            tools_path=str(self.data_dir / "tools.json"),
        )

        # Thread safety locks
        self._feedback_lock = threading.Lock()
        self._reranker_lock = threading.Lock()
        self._print_lock = threading.Lock()
        self._completed = 0
        self._completed_lock = threading.Lock()

        self.reranker = None
        if use_reranker:
            model_file = self.data_dir / "models" / "reranker.pt"
            self.reranker = Reranker(
                embeddings_path=str(self.data_dir / "embeddings.json"),
                feedback_path=str(self.data_dir / "feedback.jsonl"),
                model_path=str(model_file) if model_file.exists() else None,
                log_fn=self._locked_print,
            )

        self.mcp_client = MCPClient(timeout=30)
        self.agent = Agent()

        self.feedback_path = self.data_dir / "feedback.jsonl"

        # Snapshot clean fixtures so filesystem tasks can't corrupt future runs
        self._fixtures_dir = self.data_dir / "fixtures"
        self._fixtures_snapshot = self.data_dir / ".fixtures_snapshot"
        self._fixtures_lock = threading.Lock()
        if self._fixtures_dir.exists():
            if self._fixtures_snapshot.exists():
                shutil.rmtree(self._fixtures_snapshot)
            shutil.copytree(self._fixtures_dir, self._fixtures_snapshot)

        self._successful_calls = {}
        self._successful_calls_lock = threading.Lock()
        self._load_successful_calls()

    def _locked_print(self, *args, **kwargs):
        with self._print_lock:
            print(*args, **kwargs)

    def _resolve_paths(self, arguments: dict) -> dict:
        """Resolve relative data/ paths in tool arguments to absolute paths."""
        project_root = str(self.data_dir.parent.resolve())

        def _resolve(v):
            if isinstance(v, str) and v.startswith("data/"):
                return os.path.join(project_root, v)
            if isinstance(v, list):
                return [_resolve(item) for item in v]
            if isinstance(v, dict):
                return {k: _resolve(val) for k, val in v.items()}
            return v

        return {k: _resolve(v) for k, v in arguments.items()}

    def _load_successful_calls(self):
        if not self.feedback_path.exists():
            return
        with open(self.feedback_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("rating", {}).get("success") and "selected" in r:
                    sel = r["selected"]
                    key = f"{sel['server_id']}/{sel['tool_name']}"
                    self._successful_calls[key] = sel.get("arguments", {})

    def _record_successful_call(self, server_id: str, tool_name: str, arguments: dict):
        key = f"{server_id}/{tool_name}"
        with self._successful_calls_lock:
            self._successful_calls[key] = arguments

    def _get_example_calls(self, candidates: list[dict]) -> dict:
        examples = {}
        with self._successful_calls_lock:
            for c in candidates:
                key = f"{c['server_id']}/{c['tool_name']}"
                if key in self._successful_calls:
                    examples[key] = self._successful_calls[key]
        return examples

    def run_task(self, task: str, category: str, artifact: str) -> dict:
        """Run a single task through the 3-call agentic loop."""

        # Call 1: Agent decomposes task into a query
        query = self.agent.call_1_decompose(task)

        # Retrieve top 10
        candidates = self.retriever.retrieve(query, top_k=self.retrieve_k)

        all_retriever_candidates = [
            {
                "server_id": c["server_id"],
                "tool_name": c["tool_name"],
                "similarity": c.get("similarity", 0),
                "retriever_rank": i + 1,
            }
            for i, c in enumerate(candidates)
        ]

        # Reranker rescores, give the agent top 5
        if self.reranker:
            with self._reranker_lock:
                candidates = self.reranker.rerank(
                    candidates, query,
                    query_category=category,
                    top_k=self.rerank_k,
                )
        else:
            candidates = candidates[:self.rerank_k]

        # Call 2: Agent selects and calls a tool (native function calling)
        example_calls = self._get_example_calls(candidates)
        selection = self.agent.call_2_select_and_call(task, query, candidates, example_calls)

        tool_idx = int(selection.get("tool_index", 1)) - 1
        tool_idx = max(0, min(tool_idx, len(candidates) - 1))
        selected_tool = candidates[tool_idx]
        arguments = selection.get("arguments", {})

        # Resolve relative paths to absolute (MCP servers don't share our cwd)
        arguments = self._resolve_paths(arguments)

        # Actually call the MCP tool
        tool_result = self.mcp_client.call_tool(
            server_id=selected_tool["server_id"],
            tool_name=selected_tool["tool_name"],
            arguments=arguments,
            install=selected_tool["install"],
        )

        # Call 3: Agent rates the tool
        rating = self.agent.call_3_rate(task, selected_tool, tool_result)

        feedback = {
            "task": task,
            "category": category,
            "artifact": artifact,
            "query": query,
            "retriever_candidates": all_retriever_candidates,
            "candidates": [
                {
                    "server_id": c["server_id"],
                    "tool_name": c["tool_name"],
                    "similarity": c.get("similarity", 0),
                    "rerank_score": c.get("rerank_score"),
                    "relevance_score": c.get("relevance_score"),
                    "quality_score": c.get("quality_score"),
                }
                for c in candidates
            ],
            "selected": {
                "server_id": selected_tool["server_id"],
                "tool_name": selected_tool["tool_name"],
                "arguments": arguments,
            },
            "tool_result_preview": str(tool_result)[:500],
            "rating": rating,
            "timestamp": time.time(),
        }

        return feedback

    def save_feedback(self, feedback: dict):
        with self._feedback_lock:
            with open(self.feedback_path, "a") as f:
                f.write(json.dumps(feedback) + "\n")

    def _reset_fixtures(self):
        """Restore fixtures to clean snapshot before a filesystem task."""
        with self._fixtures_lock:
            if self._fixtures_snapshot.exists():
                if self._fixtures_dir.exists():
                    shutil.rmtree(self._fixtures_dir)
                shutil.copytree(self._fixtures_snapshot, self._fixtures_dir)

    def _run_single(self, i: int, total: int, t: dict):
        task_text = t["task"]
        category = t["category"]
        artifact = t.get("artifact", "")

        if category == "filesystem":
            self._reset_fixtures()

        try:
            feedback = self.run_task(task_text, category, artifact)
            self.save_feedback(feedback)

            if feedback["rating"].get("success"):
                sel = feedback["selected"]
                self._record_successful_call(
                    sel["server_id"], sel["tool_name"],
                    sel.get("arguments", {}))

            if self.reranker:
                with self._reranker_lock:
                    self.reranker.online_update(feedback)
                    self.reranker.maybe_batch_train(batch_size=self.batch_train_every)

            with self._completed_lock:
                self._completed += 1
                completed = self._completed

            rating = feedback["rating"]
            with self._print_lock:
                print(f"[{completed}/{total}] ({category}) {task_text[:80]}...")
                print(f"  -> {feedback['selected']['server_id']}/{feedback['selected']['tool_name']}")
                print(f"  -> relevance={rating.get('relevance')}, success={rating.get('success')}, score={rating.get('score')}")
                print()

        except Exception as e:
            with self._completed_lock:
                self._completed += 1
                completed = self._completed

            with self._print_lock:
                print(f"[{completed}/{total}] ({category}) {task_text[:80]}...")
                print(f"  -> ERROR: {e}\n")

            self.save_feedback({
                "task": task_text,
                "category": category,
                "artifact": artifact,
                "error": str(e),
                "timestamp": time.time(),
            })

    def run(self, tasks_path: str = None, limit: int = None, skip: int = 0):
        if tasks_path is None:
            tasks_path = str(self.data_dir / "tasks.json")

        with open(tasks_path, "r") as f:
            tasks = json.load(f)

        if skip:
            tasks = tasks[skip:]

        if limit:
            tasks = tasks[:limit]

        total = len(tasks)
        self._completed = 0
        print(f"Running {total} tasks (concurrency={self.concurrency})...\n")

        if self.concurrency <= 1:
            for i, t in enumerate(tasks):
                self._run_single(i, total, t)
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = []
                for i, t in enumerate(tasks):
                    futures.append(executor.submit(self._run_single, i, total, t))

                for f in as_completed(futures):
                    f.result()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent tasks")
    parser.add_argument("--no-reranker", action="store_true", help="Run without reranker (retriever only)")
    parser.add_argument("--train", action="store_true", help="Train reranker on existing feedback before running")
    parser.add_argument("--train-only", action="store_true", help="Only train reranker, don't run tasks")
    parser.add_argument("--resume", action="store_true", help="Resume from where we left off")
    parser.add_argument("--skip", type=int, default=0, help="Number of tasks to skip")
    parser.add_argument("--tasks", type=str, default=None, help="Path to tasks JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to output feedback JSONL file")
    args = parser.parse_args()

    pipeline = Pipeline(
        use_reranker=not args.no_reranker,
        concurrency=args.concurrency,
    )

    if args.train or args.train_only:
        if pipeline.reranker:
            print("Training reranker on feedback...\n")
            pipeline.reranker.train_on_feedback(str(pipeline.feedback_path))
            models_dir = pipeline.data_dir / "models"
            os.makedirs(models_dir, exist_ok=True)
            pipeline.reranker.save(str(models_dir / "reranker.pt"))

    if args.output:
        pipeline.feedback_path = Path(args.output)

    if not args.train_only:
        skip = args.skip
        if args.resume:
            if pipeline.feedback_path.exists():
                with open(pipeline.feedback_path) as f:
                    skip = sum(1 for _ in f)
                print(f"Resuming: skipping {skip} already-completed tasks.\n")
        pipeline.run(tasks_path=args.tasks, limit=args.limit, skip=skip)
