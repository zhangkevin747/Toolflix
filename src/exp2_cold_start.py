"""
Experiment 2: Cold start — new tools, good→bad, bad→good.

Synthetic replay loop that uses the real retriever + reranker, but replaces
the expensive LLM/MCP call with a scripted oracle that decides success from
the selected tool's current state. Lets us simulate thousands of feedback
steps in minutes and inject scheduled tool-quality transitions.

Three sub-experiments, each with one canary tool:

  2a — cold start:  canary hidden from candidates until step T_intro; after
                    introduction it succeeds ~95%. Measures time-to-top-1.
  2b — good→bad:    canary succeeds ~95% for T_break steps; then fails
                    deterministically. Measures time-to-downrank.
  2c — bad→good:    canary fails for T_fix steps; then succeeds ~95%.
                    Measures time-to-recovery (expected to be slow since
                    the model stops picking it once downranked).

Each run logs the canary's rank on a fixed probe set every 50 steps and
writes a jsonl file; the downstream `plot` function renders a 3-panel figure.
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from reranker import (WideAndDeepModel, WideFeatures, MODEL_REGISTRY,
                      MODEL_TO_IDX, MODEL_EMB_DIM, EMBED_DIM, PROJ_DIM)
from retriever import Retriever, BLOCKED_ENDPOINTS
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------
# Task pool — 10 natural queries per category
# --------------------------------------------------------------------------
TASK_POOL = {
    "arxiv": [
        "arxiv paper abstract",
        "find arxiv preprint",
        "download arxiv paper pdf",
        "arxiv search by author",
        "latest arxiv submissions",
        "get arxiv paper metadata",
        "arxiv paper by id",
        "research paper lookup on arxiv",
        "arxiv latex source",
        "arxiv abstract retrieval",
    ],
    "excel": [
        "read excel spreadsheet",
        "parse xlsx file",
        "open excel workbook",
        "extract cells from excel",
        "read rows from spreadsheet",
        "get data from xlsx",
        "load excel file",
        "show excel contents",
        "excel cell values",
        "read spreadsheet data",
    ],
    "fetch": [
        "fetch a url",
        "download webpage html",
        "http get request",
        "retrieve page content",
        "scrape web page",
        "fetch website contents",
        "get url body",
        "download html",
        "http fetch tool",
        "visit web page",
    ],
    "filesystem": [
        "read a local file",
        "list directory contents",
        "open text file",
        "show file contents",
        "read file from disk",
        "file read tool",
        "browse filesystem",
        "view directory",
        "read docx file",
        "read python source file",
    ],
    "pdf": [
        "read pdf document",
        "extract text from pdf",
        "parse pdf file",
        "get pdf contents",
        "open pdf",
        "pdf text extraction",
        "read pdf pages",
        "pdf parser",
        "get text from pdf",
        "load pdf document",
    ],
    "search": [
        "web search tool",
        "google search for information",
        "search the internet",
        "web search engine",
        "find news on the web",
        "search query tool",
        "web lookup",
        "google query",
        "search results",
        "online search",
    ],
    "wikipedia": [
        "wikipedia article lookup",
        "get wikipedia page",
        "wikipedia search",
        "find wikipedia entry",
        "read wikipedia article",
        "wikipedia page retrieval",
        "look up wikipedia",
        "wiki article",
        "wikipedia page by title",
        "encyclopedia lookup",
    ],
}


# --------------------------------------------------------------------------
# Oracle — scripted success for every (server_id, step)
# --------------------------------------------------------------------------
BASE_SUCCESS = {
    # In-category real tools succeed most of the time
    "real_right_cat": 0.90,
    # Real tools in wrong category (agent picked wrong bucket): failure
    "real_wrong_cat": 0.10,
    # Synthetic behaviors mirror what we see in real runs
    "truncated": 0.60,
    "hollow":    0.50,
    "flaky":     0.50,
    "stale":     0.20,
}

SYNTH_TYPES = ("truncated", "hollow", "flaky", "stale")


def classify_tool(server_id: str) -> str:
    for t in SYNTH_TYPES:
        if f"-synth-{t}-" in server_id:
            return t
    return "real"


class Oracle:
    def __init__(self, canary: dict, tool_category: dict, rng: random.Random):
        self.canary = canary          # {'server_id', 'schedule': [(from_step, p_success)]}
        self.tool_category = tool_category
        self.rng = rng

    def canary_p(self, step: int) -> float | None:
        if not self.canary:
            return None
        schedule = self.canary.get("schedule", [])
        current = None
        for from_step, p in schedule:
            if step >= from_step:
                current = p
        return current

    def success(self, server_id: str, task_category: str, step: int) -> bool:
        if self.canary and server_id == self.canary["server_id"]:
            p = self.canary_p(step)
            if p is None:
                # Pre-introduction: should never be picked (caller filters)
                return False
            return self.rng.random() < p

        kind = classify_tool(server_id)
        tool_cat = self.tool_category[server_id]
        if kind == "real":
            p = BASE_SUCCESS["real_right_cat"] if tool_cat == task_category \
                else BASE_SUCCESS["real_wrong_cat"]
        else:
            p = BASE_SUCCESS[kind]
        return self.rng.random() < p


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

class Simulator:
    def __init__(self, retriever: Retriever, canary: dict, seed: int = 0):
        self.retriever = retriever
        self.canary = canary
        self.rng = random.Random(seed)
        np.random.seed(seed); torch.manual_seed(seed)

        # Categories and mapping
        self.tool_category = {}
        for (sid, tname), meta in retriever.tool_index.items():
            self.tool_category[sid] = meta["category"]

        # Build tool-embedding index consistent with retriever ordering
        self.endpoint_keys = [f"{ep['server_id']}/{ep['tool_name']}"
                              for ep in retriever.endpoints]
        self.endpoint_to_idx = {k: i for i, k in enumerate(self.endpoint_keys)}
        self.endpoint_embs = torch.tensor(
            [ep["embedding"] for ep in retriever.endpoints],
            dtype=torch.float32,
        )

        # Fresh reranker state
        self.wide = WideFeatures()
        self.model = WideAndDeepModel()
        self.model.eval()
        self.replay_buffer = []   # [{query, selected_key, similarity, success, model_name}]
        self.model_name = MODEL_REGISTRY[0]
        self._model_idx = MODEL_TO_IDX[self.model_name]

        self.oracle = Oracle(canary, self.tool_category, self.rng)

        # Encoder (only one, no duplicate sentence-transformers instantiation)
        self.encoder = retriever.model
        self._query_emb_cache = {}

    # ------------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        if query not in self._query_emb_cache:
            self._query_emb_cache[query] = self.encoder.encode([query])[0]
        return self._query_emb_cache[query]

    def retrieve_with_canary_filter(self, query: str, step: int, top_k: int):
        """Retrieve top_k candidates, dropping the canary if it's 'hidden'."""
        cands = self.retriever.retrieve(query, top_k=top_k * 2)
        if self.canary and self.canary.get("schedule"):
            first_intro = min(s for s, p in self.canary["schedule"])
            if step < first_intro:
                # Canary is hidden
                cands = [c for c in cands
                         if c["server_id"] != self.canary["server_id"]]
        return cands[:top_k]

    def reranker_score(self, candidates, query_emb, total_selections):
        """Score candidates with current wide+deep. Returns [(cand, score, rank)]."""
        N = len(candidates)
        if N == 0:
            return []

        wide_feats = []
        tool_idxs = []
        for c in candidates:
            key = f"{c['server_id']}/{c['tool_name']}"
            usage = self.wide.usage_count[key]
            max_u = max(self.wide.usage_count.values(), default=1) or 1
            norm_u = usage / max_u if max_u else 0.0
            sr = self.wide.success_count[key] / usage if usage > 0 else 0.0
            mt_key = f"{self.model_name}||{key}"
            mtu = self.wide.model_tool_usage[mt_key]
            mtsr = self.wide.model_tool_success[mt_key] / mtu if mtu > 0 else 0.0
            sim_c = c.get("similarity", 0.0)
            wide_feats.append([norm_u, sr, mtsr, sim_c, sim_c * sr, sim_c * mtsr])
            tool_idxs.append(self.endpoint_to_idx.get(key, 0))

        wide_t = torch.tensor(wide_feats, dtype=torch.float32)
        tool_t = self.endpoint_embs[torch.tensor(tool_idxs)]
        q_t = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
        model_idx_t = torch.full((N,), self._model_idx, dtype=torch.long)

        with torch.no_grad():
            logits = self.model(wide_t, model_idx_t, q_t, tool_t).squeeze(-1).numpy()

        # UCB bonus
        total = max(total_selections, 1)
        beta = 0.05
        for i, c in enumerate(candidates):
            key = f"{c['server_id']}/{c['tool_name']}"
            u = self.wide.usage_count.get(key, 0)
            logits[i] += beta * math.sqrt(math.log(total + 1) / (u + 1))

        order = np.argsort(-logits)
        out = []
        for rank, idx in enumerate(order, 1):
            out.append((candidates[idx], float(logits[idx]), rank))
        return out

    # ------------------------------------------------------------------
    def train(self, epochs: int = 10, lr: float = 2e-3, max_records: int = 2000):
        if not self.replay_buffer:
            return
        recs = self.replay_buffer[-max_records:]
        # Encode unique queries
        unique_q = list({r["query"] for r in recs})
        q_embs = self.encoder.encode(unique_q, show_progress_bar=False)
        q_emb_map = {q: e for q, e in zip(unique_q, q_embs)}

        wide_feats = []
        tool_idxs = []
        q_list = []
        model_idxs = []
        labels = []
        for r in recs:
            key = r["selected_key"]
            idx = self.endpoint_to_idx.get(key)
            if idx is None:
                continue
            usage = self.wide.usage_count[key]
            max_u = max(self.wide.usage_count.values(), default=1) or 1
            norm_u = usage / max_u if max_u else 0.0
            sr = self.wide.success_count[key] / usage if usage > 0 else 0.0
            mt_key = f"{r['model_name']}||{key}"
            mtu = self.wide.model_tool_usage[mt_key]
            mtsr = self.wide.model_tool_success[mt_key] / mtu if mtu > 0 else 0.0
            sim_r = r["similarity"]
            wide_feats.append([norm_u, sr, mtsr, sim_r, sim_r * sr, sim_r * mtsr])
            tool_idxs.append(idx)
            q_list.append(q_emb_map[r["query"]])
            model_idxs.append(MODEL_TO_IDX.get(r["model_name"], 0))
            labels.append(float(r["success"]))

        if not labels:
            return
        wide_t = torch.tensor(wide_feats, dtype=torch.float32)
        q_t = torch.tensor(np.stack(q_list), dtype=torch.float32)
        tool_t = self.endpoint_embs[torch.tensor(tool_idxs)]
        m_t = torch.tensor(model_idxs, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            logits = self.model(wide_t, m_t, q_t, tool_t)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        self.model.eval()

    # ------------------------------------------------------------------
    def step(self, step_idx: int):
        cat = self.rng.choice(list(TASK_POOL.keys()))
        query = self.rng.choice(TASK_POOL[cat])
        q_emb = self.embed_query(query)

        cands = self.retrieve_with_canary_filter(query, step_idx, top_k=100)
        scored = self.reranker_score(cands, q_emb,
                                     total_selections=sum(self.wide.usage_count.values()))
        if not scored:
            return
        picked, _score, _rank = scored[0]
        key = f"{picked['server_id']}/{picked['tool_name']}"
        success = self.oracle.success(picked["server_id"], cat, step_idx)

        self.wide.update(key, self.model_name, success)
        self.replay_buffer.append({
            "query": query, "selected_key": key,
            "similarity": picked.get("similarity", 0.0),
            "success": success, "model_name": self.model_name,
        })

    # ------------------------------------------------------------------
    def probe(self, step_idx: int, probe_set: dict) -> list[dict]:
        """probe_set: {category: [queries...]}. Returns one row per probe."""
        rows = []
        for cat, queries in probe_set.items():
            for q in queries:
                q_emb = self.embed_query(q)
                cands = self.retrieve_with_canary_filter(q, step_idx, top_k=100)
                scored = self.reranker_score(
                    cands, q_emb, total_selections=sum(self.wide.usage_count.values()))
                # Find canary
                canary_rank = None
                canary_score = None
                if self.canary:
                    for cand, sc, rk in scored:
                        if cand["server_id"] == self.canary["server_id"]:
                            canary_rank = rk
                            canary_score = sc
                            break
                # Also record top-1
                top1 = scored[0][0] if scored else None
                rows.append({
                    "step": step_idx,
                    "probe_category": cat,
                    "probe_query": q,
                    "canary_server_id": self.canary["server_id"] if self.canary else None,
                    "canary_rank": canary_rank,
                    "canary_score": canary_score,
                    "canary_in_pool": canary_rank is not None,
                    "top1_server_id": top1["server_id"] if top1 else None,
                    "canary_server_usage": sum_usage(self.wide, self.canary["server_id"]) if self.canary else 0,
                })
        return rows


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(experiment_name: str,
        canary: dict,
        probe_set: dict,
        T: int = 2000,
        probe_every: int = 50,
        train_every: int = 50,
        seed: int = 0,
        retriever: Retriever | None = None,
        out_path: str = None):
    if retriever is None:
        retriever = Retriever(embeddings_path="data/embeddings.json",
                              tools_path="data/tools.json")
    sim = Simulator(retriever, canary, seed=seed)

    log_rows = []

    # Probe once at step 0
    log_rows.extend(sim.probe(0, probe_set))

    t0 = time.time()
    for step in range(1, T + 1):
        sim.step(step)
        if step % train_every == 0:
            sim.train(epochs=5, lr=2e-3)
        if step % probe_every == 0:
            log_rows.extend(sim.probe(step, probe_set))
            ru = sim.wide.usage_count.get(
                f"{canary['server_id']}/{_any_tool(retriever, canary['server_id'])}", 0)
            print(f"  [{experiment_name}] step={step:>5}  elapsed={time.time()-t0:.0f}s  "
                  f"buf={len(sim.replay_buffer)}  canary_usage={sum_usage(sim.wide, canary['server_id'])}",
                  flush=True)

    out_path = out_path or f"data/exp2_{experiment_name}.jsonl"
    with open(out_path, "w") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {out_path}  ({len(log_rows)} probes)")


def _any_tool(retriever, server_id):
    for (sid, tname) in retriever.tool_index:
        if sid == server_id:
            return tname
    return ""


def sum_usage(wide: WideFeatures, server_id: str) -> int:
    return sum(v for k, v in wide.usage_count.items() if k.startswith(server_id + "/"))


# --------------------------------------------------------------------------
# Canary specs
# --------------------------------------------------------------------------

CANARIES = {
    "2a_cold_start": {
        # Hidden until step T_intro, then succeeds 0.95.
        "server_id": "arxiv-latex-mcp",
        "schedule": [(500, 0.95)],
    },
    "2b_good_to_bad": {
        # Succeeds for 500 steps, then hard-fails.
        # Using a tool that the retriever puts at top-1 so it's naturally used
        # during warmup; otherwise the reranker never picks it, and nothing
        # to "degrade" when it breaks.
        "server_id": "arxiv-latex-mcp",
        "schedule": [(0, 0.95), (500, 0.00)],
    },
    "2c_bad_to_good": {
        # Fails for 500 steps, then succeeds.
        "server_id": "arxiv-latex-mcp",
        "schedule": [(0, 0.00), (500, 0.95)],
    },
}

PROBE_CATS = {
    "2a_cold_start": {"arxiv": TASK_POOL["arxiv"][:5]},
    "2b_good_to_bad": {"arxiv": TASK_POOL["arxiv"][:5]},
    "2c_bad_to_good": {"arxiv": TASK_POOL["arxiv"][:5]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=list(CANARIES.keys()) + ["all"], default="all")
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    retriever = Retriever(embeddings_path="data/embeddings.json",
                          tools_path="data/tools.json")

    names = list(CANARIES.keys()) if args.experiment == "all" else [args.experiment]
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        run(name, CANARIES[name], PROBE_CATS[name],
            T=args.T, seed=args.seed, retriever=retriever)


if __name__ == "__main__":
    main()
