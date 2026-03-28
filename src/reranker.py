"""
Wide & Deep Reranker — Two-Head Architecture

Retrieve top 10, reranker rescores, give the agent top 5.

Two prediction heads:
  - Relevance head: P(tool is semantically appropriate for query).
    Leans on deep features (query/description embeddings, interaction features,
    learned endpoint embeddings). Pretrainable without execution feedback.
  - Quality head: P(tool succeeds on this query).
    Leans on wide features (success rate, usage count, co-occurrence, category match).
    Trained on execution feedback. This is the head that beats BM25.

Final score = alpha * relevance + (1 - alpha) * quality, where alpha decays
as feedback accumulates (quality head becomes more trustworthy over time).

UCB exploration bonus to surface underexplored tools.
"""
import json
import math
import threading
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer


class WideFeatures:
    """Wide: per-endpoint success rate, usage count, query-endpoint co-occurrence, category match."""

    def __init__(self):
        self.usage_count = defaultdict(int)
        self.success_count = defaultdict(int)
        self.relevance_count = defaultdict(int)
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self._norm_max_usage = None
        self._norm_max_cooccur = {}

    def update(self, endpoint_key: str, query_category: str,
               relevance: bool, success: bool):
        self.usage_count[endpoint_key] += 1
        if success:
            self.success_count[endpoint_key] += 1
        if relevance:
            self.relevance_count[endpoint_key] += 1
        self.cooccurrence[query_category][endpoint_key] += 1

    def get_features(self, endpoint_key: str, endpoint_category: str,
                     query_category: str, similarity: float = 0.0) -> list[float]:
        usage = self.usage_count[endpoint_key]
        success_rate = (
            self.success_count[endpoint_key] / usage if usage > 0 else 0.0
        )
        relevance_rate = (
            self.relevance_count[endpoint_key] / usage if usage > 0 else 0.0
        )
        cooccur = self.cooccurrence[query_category].get(endpoint_key, 0)
        category_match = 1.0 if endpoint_category == query_category else 0.0

        if self._norm_max_usage is not None:
            max_usage = self._norm_max_usage
        else:
            max_usage = max(self.usage_count.values()) if self.usage_count else 1
        normalized_usage = usage / max_usage if max_usage > 0 else 0.0

        cat_counts = self.cooccurrence.get(query_category, {})
        if query_category in self._norm_max_cooccur:
            max_cooccur = self._norm_max_cooccur[query_category]
        else:
            max_cooccur = max(cat_counts.values()) if cat_counts else 1
        normalized_cooccur = cooccur / max_cooccur if max_cooccur > 0 else 0.0

        return [
            normalized_usage,
            success_rate,
            relevance_rate,
            normalized_cooccur,
            category_match,
            similarity,
        ]

    def snapshot_norms(self):
        self._norm_max_usage = max(self.usage_count.values()) if self.usage_count else 1
        self._norm_max_cooccur = {}
        for cat, counts in self.cooccurrence.items():
            self._norm_max_cooccur[cat] = max(counts.values()) if counts else 1

    def load_from_feedback(self, feedback_path: str):
        path = Path(feedback_path)
        if not path.exists():
            return

        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in record and "selected" not in record:
                    continue

                selected = record.get("selected", {})
                endpoint_key = f"{selected['server_id']}/{selected['tool_name']}"
                category = record.get("category", "")
                rating = record.get("rating", {})

                self.update(
                    endpoint_key=endpoint_key,
                    query_category=category,
                    relevance=rating.get("relevance", False),
                    success=rating.get("success", False),
                )


WIDE_DIM = 6        # Number of wide features
EMBED_DIM = 384     # Sentence transformer embedding dimension
PROJ_DIM = 64       # Projection dimension for query/description embeddings
LEARNED_DIM = 32    # Learned endpoint embedding dimension


class WideAndDeepModel(nn.Module):
    """
    Two-head Wide & Deep scoring model.

    Shared trunk computes deep features:
      - Cosine similarity (scalar)
      - Element-wise product of projected embeddings (feature co-activation)
      - Difference of projected embeddings (what query needs that tool lacks)
      - Learned endpoint embedding

    Two prediction heads:
      - Relevance head: deep features only → P(semantically appropriate)
      - Quality head: wide + deep features → P(execution succeeds)
    """

    def __init__(self, num_endpoints: int):
        super().__init__()

        self.query_proj = nn.Linear(EMBED_DIM, PROJ_DIM)
        self.desc_proj = nn.Linear(EMBED_DIM, PROJ_DIM)

        self.endpoint_embeddings = nn.Embedding(num_endpoints, LEARNED_DIM)

        # Deep input: cosine_sim(1) + hadamard(PROJ_DIM) + diff(PROJ_DIM) + learned(LEARNED_DIM)
        deep_input_dim = 1 + PROJ_DIM + PROJ_DIM + LEARNED_DIM
        self.deep_net = nn.Sequential(
            nn.Linear(deep_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Relevance head: deep features only → P(relevant)
        self.relevance_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Quality head: wide + deep features → P(success)
        self.quality_head = nn.Sequential(
            nn.Linear(WIDE_DIM + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, wide_features: torch.Tensor, query_emb: torch.Tensor,
                desc_emb: torch.Tensor, endpoint_idx: torch.Tensor):
        """Returns (relevance_logit, quality_logit, deep_out)."""
        q_proj = self.query_proj(query_emb)
        d_proj = self.desc_proj(desc_emb)

        cosine_sim = nn.functional.cosine_similarity(q_proj, d_proj, dim=1).unsqueeze(1)
        hadamard = q_proj * d_proj
        diff = q_proj - d_proj

        learned_emb = self.endpoint_embeddings(endpoint_idx)

        deep_input = torch.cat([cosine_sim, hadamard, diff, learned_emb], dim=1)
        deep_out = self.deep_net(deep_input)

        relevance_logit = self.relevance_head(deep_out)
        quality_input = torch.cat([wide_features, deep_out], dim=1)
        quality_logit = self.quality_head(quality_input)

        return relevance_logit, quality_logit


class Reranker:
    """Retrieve top 10, reranker rescores, give the agent top 5."""

    def __init__(self, embeddings_path: str, feedback_path: str,
                 model_path: str = None, log_fn=None):
        self._log = log_fn or print

        with open(embeddings_path) as f:
            self.endpoints = json.load(f)

        self.endpoint_to_idx = {}
        self.endpoint_desc_embs = []
        for i, ep in enumerate(self.endpoints):
            key = f"{ep['server_id']}/{ep['tool_name']}"
            self.endpoint_to_idx[key] = i
            self.endpoint_desc_embs.append(ep["embedding"])

        self.endpoint_desc_embs = torch.tensor(self.endpoint_desc_embs, dtype=torch.float32)
        num_endpoints = len(self.endpoints)

        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._encode_lock = threading.Lock()

        self.wide = WideFeatures()
        self.wide.load_from_feedback(feedback_path)

        self.model = WideAndDeepModel(num_endpoints)

        # Initialize learned endpoint embeddings from description embeddings
        with torch.no_grad():
            proj = nn.Linear(EMBED_DIM, LEARNED_DIM)
            init_embs = proj(self.endpoint_desc_embs)
            self.model.endpoint_embeddings.weight.copy_(init_embs)

        self._replay_buffer = []
        self._trained_up_to = 0
        self._load_replay_buffer(feedback_path)

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self._trained_up_to = len(self._replay_buffer)

        self.model.eval()

    def _safe_encode(self, texts):
        with self._encode_lock:
            return self.st_model.encode(texts)

    def _load_replay_buffer(self, feedback_path: str):
        path = Path(feedback_path)
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "selected" in record and "rating" in record:
                    self._replay_buffer.append(record)

    def online_update(self, feedback: dict):
        selected = feedback.get("selected", {})
        if not selected:
            return

        endpoint_key = f"{selected['server_id']}/{selected['tool_name']}"
        category = feedback.get("category", "")
        rating = feedback.get("rating", {})

        self.wide.update(
            endpoint_key=endpoint_key,
            query_category=category,
            relevance=rating.get("relevance", False),
            success=rating.get("success", False),
        )

        self._replay_buffer.append(feedback)

    def maybe_batch_train(self, batch_size: int = 50):
        new_examples = len(self._replay_buffer) - self._trained_up_to
        if new_examples < batch_size:
            return False

        self._log(f"\n  [Reranker] Batch training on {len(self._replay_buffer)} total examples ({new_examples} new)...")
        self._train_deep_on_buffer()
        self._trained_up_to = len(self._replay_buffer)
        return True

    def _build_pairwise_data(self, records: list[dict]):
        """Build pairwise training data with hard negative mining.

        Both heads train on the same success signal, but see different features:
          - Relevance pairs: deep features only. Learns query-tool affinity through
            embeddings. "For this type of query, this type of tool works."
            The personalized recommendation — forced to generalize.
          - Quality pairs: wide + deep features. Learns tool-specific execution
            history. "This specific tool has a 67% success rate."
            The star rating — memorized from aggregate stats.
        """
        query_groups = []
        for r in records:
            candidates = r.get("retriever_candidates", [])
            if len(candidates) < 2:
                continue

            selected = r["selected"]
            sel_key = f"{selected['server_id']}/{selected['tool_name']}"
            rating = r.get("rating", {})
            success = rating.get("success", False)
            query_text = r.get("query", r.get("task", ""))
            category = r.get("category", "")

            cand_info = []
            for c in candidates:
                key = f"{c['server_id']}/{c['tool_name']}"
                idx = self.endpoint_to_idx.get(key)
                if idx is None:
                    continue
                cand_info.append({
                    "key": key,
                    "idx": idx,
                    "server_id": c["server_id"],
                    "similarity": c.get("similarity", 0.0),
                    "is_selected": key == sel_key,
                })

            if not any(ci["is_selected"] for ci in cand_info):
                continue

            query_groups.append({
                "query_text": query_text,
                "category": category,
                "candidates": cand_info,
                "success": success,
            })

        return query_groups

    def _train_two_heads(self, records: list[dict], epochs: int = 20,
                         lr: float = None, margin: float = 0.5):
        """
        Both heads train on the same success signal (succeeded > others,
        failed < others), but see different features:

          - Relevance head (deep only): the personalized recommendation.
            Learns query-tool affinity through embeddings. No access to
            aggregate stats — forced to generalize. "For queries about PDF
            extraction, tools with these embedding features tend to succeed."

          - Quality head (wide + deep): the star rating.
            Learns tool-specific execution history from wide features.
            "This specific tool has a 67% success rate." Memorization.

        Three phases:
          1. Relevance head only (zero wide) — learns through embeddings
          2. Quality head (wide + deep) — learns from aggregate stats
          3. Joint fine-tune (both heads, shared trunk)
        """
        query_groups = self._build_pairwise_data(records)
        if not query_groups:
            return

        # Scale lr with dataset size — not too aggressive, prevent collapse
        if lr is None:
            lr = min(5e-3, 5.0 / (len(records) + 200))

        unique_queries = list({g["query_text"] for g in query_groups})
        query_embs = self._safe_encode(unique_queries)
        query_emb_map = dict(zip(unique_queries, query_embs))

        pair_query_embs = []
        pos_wides, pos_descs, pos_idxs = [], [], []
        neg_wides, neg_descs, neg_idxs = [], [], []
        pair_weights = []

        for g in query_groups:
            q_emb = query_emb_map[g["query_text"]]
            category = g["category"]
            selected = [c for c in g["candidates"] if c["is_selected"]][0]
            others = [c for c in g["candidates"] if not c["is_selected"]]

            if not others:
                continue

            for other in others:
                same_server = selected["server_id"] == other["server_id"]
                both_high_sim = selected["similarity"] > 0.35 and other["similarity"] > 0.35
                weight = 3.0 if (same_server or both_high_sim) else 1.0

                def _append_pair(pos, neg, w):
                    pos_ep = self.endpoints[pos["idx"]]
                    neg_ep = self.endpoints[neg["idx"]]
                    pair_query_embs.append(q_emb)
                    pos_wides.append(self.wide.get_features(
                        pos["key"], pos_ep.get("category", ""), category, pos["similarity"]))
                    pos_descs.append(pos["idx"])
                    pos_idxs.append(pos["idx"])
                    neg_wides.append(self.wide.get_features(
                        neg["key"], neg_ep.get("category", ""), category, neg["similarity"]))
                    neg_descs.append(neg["idx"])
                    neg_idxs.append(neg["idx"])
                    pair_weights.append(w)

                # Both heads learn from the same success signal:
                # succeeded → selected > others; failed → others > selected
                if g["success"]:
                    _append_pair(selected, other, w=weight)
                else:
                    _append_pair(other, selected, w=weight)

        if not pair_query_embs:
            return

        q_tensor = torch.tensor(np.array(pair_query_embs), dtype=torch.float32)
        pos_wide_t = torch.tensor(pos_wides, dtype=torch.float32)
        pos_desc_t = self.endpoint_desc_embs[pos_descs]
        pos_idx_t = torch.tensor(pos_idxs, dtype=torch.long)
        neg_wide_t = torch.tensor(neg_wides, dtype=torch.float32)
        neg_desc_t = self.endpoint_desc_embs[neg_descs]
        neg_idx_t = torch.tensor(neg_idxs, dtype=torch.long)
        weights_t = torch.tensor(pair_weights, dtype=torch.float32)

        zero_wide = torch.zeros_like(pos_wide_t)

        n_pairs = len(pair_weights)
        n_hard = sum(1 for w in pair_weights if w > 1)

        # --- Phase 1: Relevance head (deep only, zero wide) ---
        # The "personalized recommendation" — learns query-tool affinity
        # through embeddings only. No access to aggregate stats.
        rel_epochs = int(epochs * 0.4)
        rel_params = (list(self.model.query_proj.parameters()) +
                      list(self.model.desc_proj.parameters()) +
                      list(self.model.endpoint_embeddings.parameters()) +
                      list(self.model.deep_net.parameters()) +
                      list(self.model.relevance_head.parameters()))
        rel_optimizer = torch.optim.Adam(rel_params, lr=lr)

        self.model.train()
        for epoch in range(rel_epochs):
            rel_optimizer.zero_grad()
            pos_rel, _ = self.model(zero_wide, q_tensor, pos_desc_t, pos_idx_t)
            neg_rel, _ = self.model(zero_wide, q_tensor, neg_desc_t, neg_idx_t)
            raw = torch.clamp(margin - (pos_rel.squeeze() - neg_rel.squeeze()), min=0)
            loss = (raw * weights_t).mean()
            loss.backward()
            rel_optimizer.step()

        rel_loss = loss.item()

        # --- Phase 2: Quality head (wide + deep) ---
        # The "star rating" — learns tool-specific success from aggregate
        # stats (wide) plus deep features for generalization.
        qual_epochs = int(epochs * 0.4)
        qual_params = (list(self.model.quality_head.parameters()) +
                       list(self.model.deep_net.parameters()))
        qual_optimizer = torch.optim.Adam(qual_params, lr=lr)

        for epoch in range(qual_epochs):
            qual_optimizer.zero_grad()
            _, pos_qual = self.model(pos_wide_t, q_tensor, pos_desc_t, pos_idx_t)
            _, neg_qual = self.model(neg_wide_t, q_tensor, neg_desc_t, neg_idx_t)
            raw = torch.clamp(margin - (pos_qual.squeeze() - neg_qual.squeeze()), min=0)
            loss = (raw * weights_t).mean()
            loss.backward()
            qual_optimizer.step()

        qual_loss = loss.item()

        # --- Phase 3: Joint fine-tune (both heads, shared trunk) ---
        joint_epochs = epochs - rel_epochs - qual_epochs
        joint_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr * 0.3)

        for epoch in range(joint_epochs):
            joint_optimizer.zero_grad()
            # Relevance head sees zero wide (deep only)
            pos_rel, _ = self.model(zero_wide, q_tensor, pos_desc_t, pos_idx_t)
            neg_rel, _ = self.model(zero_wide, q_tensor, neg_desc_t, neg_idx_t)
            # Quality head sees real wide features
            _, pos_qual = self.model(pos_wide_t, q_tensor, pos_desc_t, pos_idx_t)
            _, neg_qual = self.model(neg_wide_t, q_tensor, neg_desc_t, neg_idx_t)

            rel_raw = torch.clamp(margin - (pos_rel.squeeze() - neg_rel.squeeze()), min=0)
            qual_raw = torch.clamp(margin - (pos_qual.squeeze() - neg_qual.squeeze()), min=0)

            loss = (rel_raw * weights_t).mean() + (qual_raw * weights_t).mean()
            loss.backward()
            joint_optimizer.step()

        joint_loss = loss.item()

        self._log(f"  [Reranker] Two-head training — rel={rel_loss:.4f} qual={qual_loss:.4f} joint={joint_loss:.4f} ({n_pairs} pairs, {n_hard} hard)")
        self.wide.snapshot_norms()
        self.model.eval()

    def _train_deep_on_buffer(self):
        records = [r for r in self._replay_buffer
                   if "selected" in r and "rating" in r]
        if records:
            # Scale epochs: enough to learn but not overfit
            # ~15 epochs for small buffers, ~10 for large
            epochs = max(10, min(15, 20 - len(records) // 200))
            self._train_two_heads(records, epochs=epochs)

    def _compute_alpha(self) -> float:
        """Compute blending weight: alpha * relevance + (1 - alpha) * quality.

        Alpha starts high (trust relevance when we have little feedback) and
        decays as feedback accumulates (quality head becomes trustworthy).
        """
        n_feedback = sum(self.wide.usage_count.values())
        # Sigmoid decay: alpha starts high (trust embeddings at cold start)
        # and settles at 0.35 (both heads contribute — relevance captures
        # query-specific patterns, quality captures per-tool reliability)
        alpha = 0.35 + 0.45 * math.exp(-n_feedback / 80)
        return alpha

    def rerank(self, candidates: list[dict], query: str,
               query_category: str = "", top_k: int = 5,
               explore: bool = True, head: str = "combined") -> list[dict]:
        """Rerank candidates using two-head scores + UCB exploration bonus.

        Args:
            head: Which head(s) to use for scoring.
                "combined" — alpha-blended relevance + quality (default)
                "relevance" — relevance head only
                "quality" — quality head only
        """
        if not candidates:
            return []

        query_emb = self._safe_encode([query])[0]
        query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)

        total_selections = max(sum(self.wide.usage_count.values()), 1)
        alpha = self._compute_alpha()

        scored = []
        for c in candidates:
            key = f"{c['server_id']}/{c['tool_name']}"
            idx = self.endpoint_to_idx.get(key)
            if idx is None:
                scored.append({**c, "rerank_score": c.get("similarity", 0),
                               "relevance_score": 0.0, "quality_score": 0.0})
                continue

            ep = self.endpoints[idx]
            wide_feat = self.wide.get_features(
                endpoint_key=key,
                endpoint_category=ep.get("category", ""),
                query_category=query_category,
                similarity=c.get("similarity", 0.0),
            )
            wide_tensor = torch.tensor([wide_feat], dtype=torch.float32)
            desc_tensor = self.endpoint_desc_embs[idx].unsqueeze(0)
            idx_tensor = torch.tensor([idx], dtype=torch.long)

            with torch.no_grad():
                rel_logit, qual_logit = self.model(wide_tensor, query_tensor, desc_tensor, idx_tensor)

            rel_score = rel_logit.item()
            qual_score = qual_logit.item()

            if head == "relevance":
                model_score = rel_score
            elif head == "quality":
                model_score = qual_score
            else:
                model_score = alpha * rel_score + (1 - alpha) * qual_score

            if explore:
                usage = self.wide.usage_count.get(key, 0)
                # Scale exploration down as feedback accumulates
                explore_weight = 0.5 * math.exp(-total_selections / 200)
                exploration_bonus = explore_weight * math.sqrt(math.log(total_selections + 1) / (usage + 1))
                model_score += exploration_bonus

            scored.append({
                **c,
                "rerank_score": model_score,
                "relevance_score": rel_score,
                "quality_score": qual_score,
            })

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    def train_on_feedback(self, feedback_path: str, epochs: int = 30,
                          lr: float = 5e-3):
        records = []
        with open(feedback_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r and "selected" not in r:
                    continue
                records.append(r)

        if not records:
            self._log("No feedback records to train on.")
            return

        self._train_two_heads(records, epochs=epochs, lr=lr)
        self._trained_up_to = len(self._replay_buffer)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        self._log(f"Saved reranker to {path}")
