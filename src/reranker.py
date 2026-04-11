"""
Wide & Deep Reranker — Single-Head Architecture (Cheng et al. 2016)

Retriever returns top K candidates, reranker rescores, returns top 5 to agent.

Single prediction head: P(tool succeeds for this query with this model).

Wide features (4): success_rate, normalized_usage, model_tool_success_rate, retriever_similarity.
Deep features:
  [emb_model (learned, 16d), emb_query (projected, 64d), emb_tool (projected, 64d),
   emb_tool_learned (per-tool, 32d), dot(model, tool_learned), dot(query, tool)]

Model embedding: nn.Embedding lookup (one-hot → dense 16d).
Query/tool embeddings: pretrained sentence-transformer with learned linear projections.
Training: pointwise logistic loss (BCEWithLogitsLoss), constant lr.
UCB1 exploration bonus (Auer et al. 2002), constant beta.
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
    """Wide: per-endpoint success rate, usage count."""

    def __init__(self):
        self.usage_count = defaultdict(int)
        self.success_count = defaultdict(int)
        # model × tool cross-feature: per-(model, tool) success rate
        self.model_tool_usage = defaultdict(int)
        self.model_tool_success = defaultdict(int)
        self._norm_max_usage = None

    def update(self, endpoint_key: str, model_name: str, success: bool):
        self.usage_count[endpoint_key] += 1
        mt_key = f"{model_name}||{endpoint_key}"
        self.model_tool_usage[mt_key] += 1
        if success:
            self.success_count[endpoint_key] += 1
            self.model_tool_success[mt_key] += 1

    def get_features(self, endpoint_key: str, model_name: str,
                     similarity: float = 0.0) -> list[float]:
        usage = self.usage_count[endpoint_key]
        success_rate = (
            self.success_count[endpoint_key] / usage if usage > 0 else 0.0
        )

        if self._norm_max_usage is not None:
            max_usage = self._norm_max_usage
        else:
            max_usage = max(self.usage_count.values()) if self.usage_count else 1
        normalized_usage = usage / max_usage if max_usage > 0 else 0.0

        # model × tool cross-feature
        mt_key = f"{model_name}||{endpoint_key}"
        mt_usage = self.model_tool_usage[mt_key]
        mt_success_rate = (
            self.model_tool_success[mt_key] / mt_usage if mt_usage > 0 else 0.0
        )

        return [normalized_usage, success_rate, mt_success_rate, similarity]

    def snapshot_norms(self):
        self._norm_max_usage = max(self.usage_count.values()) if self.usage_count else 1

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
                model_name = record.get("model", "gpt-5.4-nano")
                rating = record.get("rating", {})

                self.update(
                    endpoint_key=endpoint_key,
                    model_name=model_name,
                    success=rating.get("success", False),
                )


WIDE_DIM = 4        # Number of wide features
EMBED_DIM = 384     # Sentence transformer embedding dimension
PROJ_DIM = 64       # Projection dimension for query/tool embeddings
TOOL_EMB_DIM = 32   # Learned per-tool embedding dimension
MODEL_EMB_DIM = 32  # Learned per-model embedding dimension (matches TOOL_EMB_DIM for dot product)

# Registry of supported models — index used for nn.Embedding lookup
# Only models verified to support tool_choice="required" on OpenRouter
MODEL_REGISTRY = [
    "gpt-5.4-nano",
    "x-ai/grok-4.1-fast",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemma-4-26b-a4b-it",
    "qwen/qwen3.5-flash-02-23",
    "deepseek/deepseek-v3.2",
]
MODEL_TO_IDX = {m: i for i, m in enumerate(MODEL_REGISTRY)}


class WideAndDeepModel(nn.Module):
    """
    Single-head Wide & Deep scoring model.

    Deep features: learned model embedding, projected query/tool embeddings,
    learned per-tool embedding, and pairwise dot products.
    Wide features: success_rate, usage, model_tool_success_rate, similarity.
    """

    def __init__(self, num_endpoints: int, num_models: int = len(MODEL_REGISTRY)):
        super().__init__()

        # Learned model embedding (one-hot → dense)
        self.model_embeddings = nn.Embedding(num_models, MODEL_EMB_DIM)

        # Learned projections for query and tool description embeddings
        self.query_proj = nn.Linear(EMBED_DIM, PROJ_DIM)
        self.tool_proj = nn.Linear(EMBED_DIM, PROJ_DIM)

        # Per-tool learned embedding — initialized from description, diverges through training
        self.tool_embeddings = nn.Embedding(num_endpoints, TOOL_EMB_DIM)

        # Deep input: model emb + 2 projected embeddings + learned tool emb + 2 dot products
        # model(16) + query(64) + tool(64) + tool_learned(32) + dot(m,t)(1) + dot(q,t)(1) = 178
        deep_dim = MODEL_EMB_DIM + 2 * PROJ_DIM + TOOL_EMB_DIM + 2
        total_dim = deep_dim + WIDE_DIM

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, wide_features: torch.Tensor,
                model_idx: torch.Tensor,
                query_emb: torch.Tensor,
                tool_emb: torch.Tensor,
                tool_idx: torch.Tensor) -> torch.Tensor:
        """Returns score logit."""
        m_emb = self.model_embeddings(model_idx)
        q_proj = self.query_proj(query_emb)
        t_proj = self.tool_proj(tool_emb)
        t_learned = self.tool_embeddings(tool_idx)

        # Pairwise dot products (model×tool, query×tool)
        dot_mt = (m_emb * t_learned).sum(dim=1, keepdim=True)
        dot_qt = (q_proj * t_proj).sum(dim=1, keepdim=True)

        deep_input = torch.cat([m_emb, q_proj, t_proj, t_learned, dot_mt, dot_qt], dim=1)
        full_input = torch.cat([deep_input, wide_features], dim=1)

        return self.mlp(full_input)


class Reranker:
    """Retriever returns top K, reranker rescores, returns top 5 to agent."""

    def __init__(self, embeddings_path: str, feedback_path: str,
                 model_path: str = None, model_name: str = "gpt-5.4-nano",
                 log_fn=None):
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

        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._encode_lock = threading.Lock()

        self.model_name = model_name
        self._model_idx = MODEL_TO_IDX.get(model_name, 0)

        self.wide = WideFeatures()
        self.wide.load_from_feedback(feedback_path)

        num_endpoints = len(self.endpoints)
        self.model = WideAndDeepModel(num_endpoints)

        # Initialize per-tool embeddings from description embeddings
        with torch.no_grad():
            proj = nn.Linear(EMBED_DIM, TOOL_EMB_DIM)
            init_embs = proj(self.endpoint_desc_embs)
            self.model.tool_embeddings.weight.copy_(init_embs)

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
        rating = feedback.get("rating", {})

        model_name = feedback.get("model", self.model_name)
        self.wide.update(
            endpoint_key=endpoint_key,
            model_name=model_name,
            success=rating.get("success", False),
        )

        self._replay_buffer.append(feedback)

    def maybe_batch_train(self, batch_size: int = 50):
        new_examples = len(self._replay_buffer) - self._trained_up_to
        if new_examples < batch_size:
            return False

        self._log(f"\n  [Reranker] Batch training on {len(self._replay_buffer)} total examples ({new_examples} new)...")
        self._train_on_buffer()
        self._trained_up_to = len(self._replay_buffer)
        return True

    def _build_training_data(self, records: list[dict]):
        """Build pointwise training data: each (query, selected_tool) → binary label."""
        examples = []
        for r in records:
            selected = r.get("selected", {})
            if not selected:
                continue
            sel_key = f"{selected['server_id']}/{selected['tool_name']}"
            idx = self.endpoint_to_idx.get(sel_key)
            if idx is None:
                continue

            rating = r.get("rating", {})
            success = 1.0 if rating.get("success", False) else 0.0
            query_text = r.get("query", r.get("task", ""))
            model_name = r.get("model", self.model_name)
            similarity = 0.0
            for c in r.get("retriever_candidates", []):
                if f"{c['server_id']}/{c['tool_name']}" == sel_key:
                    similarity = c.get("similarity", 0.0)
                    break

            examples.append({
                "query_text": query_text,
                "model_name": model_name,
                "key": sel_key,
                "idx": idx,
                "similarity": similarity,
                "label": success,
            })

        return examples

    def _train(self, records: list[dict], epochs: int = 20, lr: float = 1e-3):
        """Pointwise logistic loss training (Wide & Deep, Cheng et al. 2016).

        Logits are calibrated P(success) — enables confidence thresholding.
        """
        examples = self._build_training_data(records)
        if not examples:
            return

        unique_queries = list({e["query_text"] for e in examples})
        query_embs = self._safe_encode(unique_queries)
        query_emb_map = dict(zip(unique_queries, query_embs))

        q_embs, wide_feats, tool_idxs, model_idxs, labels = [], [], [], [], []
        for e in examples:
            q_embs.append(query_emb_map[e["query_text"]])
            wide_feats.append(self.wide.get_features(e["key"], e["model_name"], e["similarity"]))
            tool_idxs.append(e["idx"])
            model_idxs.append(MODEL_TO_IDX.get(e["model_name"], 0))
            labels.append(e["label"])

        q_tensor = torch.tensor(np.array(q_embs), dtype=torch.float32)
        model_idx_t = torch.tensor(model_idxs, dtype=torch.long)
        wide_tensor = torch.tensor(wide_feats, dtype=torch.float32)
        tool_tensor = self.endpoint_desc_embs[tool_idxs]
        tool_idx_t = torch.tensor(tool_idxs, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        n_examples = len(examples)
        n_pos = sum(1 for l in labels if l > 0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.model(wide_tensor, model_idx_t, q_tensor, tool_tensor, tool_idx_t).squeeze()
            loss = loss_fn(logits, label_tensor)
            loss.backward()
            optimizer.step()

        self._log(f"  [Reranker] Training — loss={loss.item():.4f} ({n_examples} examples, {n_pos} positive)")
        self.wide.snapshot_norms()
        self.model.eval()

    def _train_on_buffer(self):
        records = [r for r in self._replay_buffer
                   if "selected" in r and "rating" in r]
        if records:
            epochs = max(10, min(15, 20 - len(records) // 200))
            self._train(records, epochs=epochs)

    def rerank(self, candidates: list[dict], query: str,
               top_k: int = 5, explore: bool = True,
               min_confidence: float = 0.0) -> list[dict]:
        """Rerank candidates using wide+deep score + UCB exploration bonus.

        Returns up to top_k tools whose predicted P(success) >= min_confidence.
        Always returns at least 1 tool (the best available) even if below threshold.
        """
        if not candidates:
            return []

        query_emb = self._safe_encode([query])[0]
        query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)
        model_idx_tensor = torch.tensor([self._model_idx], dtype=torch.long)

        total_selections = max(sum(self.wide.usage_count.values()), 1)

        scored = []
        for c in candidates:
            key = f"{c['server_id']}/{c['tool_name']}"
            idx = self.endpoint_to_idx.get(key)
            if idx is None:
                scored.append({**c, "rerank_score": c.get("similarity", 0), "confidence": 0.0})
                continue

            wide_feat = self.wide.get_features(
                endpoint_key=key,
                model_name=self.model_name,
                similarity=c.get("similarity", 0.0),
            )
            wide_tensor = torch.tensor([wide_feat], dtype=torch.float32)
            tool_tensor = self.endpoint_desc_embs[idx].unsqueeze(0)
            idx_tensor = torch.tensor([idx], dtype=torch.long)

            with torch.no_grad():
                score_logit = self.model(wide_tensor, model_idx_tensor, query_tensor, tool_tensor, idx_tensor)

            logit = score_logit.item()
            confidence = 1.0 / (1.0 + math.exp(-logit))  # sigmoid

            model_score = logit

            if explore:
                usage = self.wide.usage_count.get(key, 0)
                beta = 0.3
                exploration_bonus = beta * math.sqrt(math.log(total_selections + 1) / (usage + 1))
                model_score += exploration_bonus

            scored.append({
                **c,
                "rerank_score": model_score,
                "confidence": confidence,
            })

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Hard cutoff: only recommend tools above min_confidence, up to top_k
        filtered = [s for s in scored if s["confidence"] >= min_confidence][:top_k]
        # Always return at least the top-1
        if not filtered:
            filtered = scored[:1]

        return filtered

    def train_on_feedback(self, feedback_path: str, epochs: int = 30,
                          lr: float = 1e-3):
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

        self._train(records, epochs=epochs, lr=lr)
        self._trained_up_to = len(self._replay_buffer)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        self._log(f"Saved reranker to {path}")
