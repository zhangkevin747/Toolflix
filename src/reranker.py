"""
Wide & Deep Reranker (Cheng et al. 2016) — single head, pointwise logistic.

Retriever returns top K candidates; reranker rescores and returns top 5.

Score:
    logit = w_wide · x_wide  +  deep_mlp(x_deep)  +  b
    P(success) = sigmoid(logit)

Wide features (6): normalized_usage, success_rate, mt_success_rate,
                   retriever_similarity, sim·sr, sim·mt_sr.
Deep features: learned model embedding + projected pretrained query / tool
               embeddings + pairwise dot products ⟨m,q⟩, ⟨m,t⟩, ⟨q,t⟩.

Training: pointwise BCEWithLogitsLoss, constant lr, differential LR on
          wide path, UCB1 exploration (Auer et al. 2002) with constant β.
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
        """Return the 6 wide features: 3 aggregates + retriever similarity +
        two sim-cross products that encode 'relevant AND reliable'.
        """
        usage = self.usage_count[endpoint_key]
        success_rate = (
            self.success_count[endpoint_key] / usage if usage > 0 else 0.0
        )

        if self._norm_max_usage is not None:
            max_usage = self._norm_max_usage
        else:
            max_usage = max(self.usage_count.values()) if self.usage_count else 1
        normalized_usage = usage / max_usage if max_usage > 0 else 0.0

        # model × tool cross-feature (numeric aggregate, as Parjanya specified)
        mt_key = f"{model_name}||{endpoint_key}"
        mt_usage = self.model_tool_usage[mt_key]
        mt_success_rate = (
            self.model_tool_success[mt_key] / mt_usage if mt_usage > 0 else 0.0
        )

        return [
            normalized_usage,
            success_rate,
            mt_success_rate,
            similarity,
            similarity * success_rate,
            similarity * mt_success_rate,
        ]

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


# Wide features — six observed aggregates + similarity crosses.
#   [0] normalized_usage
#   [1] success_rate
#   [2] mt_success_rate       (model x tool)
#   [3] retriever_similarity  (cosine from retriever, as Parjanya spec'd)
#   [4] similarity * sr       (memorizes "relevant AND reliable")
#   [5] similarity * mt_sr    (same, model-specific)
WIDE_DIM = 6
EMBED_DIM = 384     # Pretrained sentence-transformer embedding dimension
PROJ_DIM = 16       # Projection dimension for query/tool/model — small by
                    # design. Limits how much W_q/W_t can reshape the
                    # embedding geometry, which is where the anti-retriever
                    # inversion came from.
MODEL_EMB_DIM = PROJ_DIM  # Must match PROJ_DIM so <m,q> and <m,t> are well-defined.

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
    Wide & Deep scoring (Cheng et al. 2016) — additive, standard form.

        logit = wide_linear(x_wide) + deep_mlp(x_deep) + bias
        P(success) = sigmoid(logit)

    Training target: BCEWithLogitsLoss against binary success label.

    Wide features (x_wide, 6 scalars):
      [normalized_usage, success_rate, mt_success_rate,
       retriever_similarity, sim * sr, sim * mt_sr]

    Deep features (x_deep):
      learned model embedding m_emb[m]              (d)
      projected pretrained query embedding q_proj   (d)
      projected pretrained tool  embedding t_proj   (d)
      pairwise dot products <m,q>, <m,t>, <q,t>     (3)

    Projection dim PROJ_DIM = 16 — small, to limit W_q/W_t reshaping
    (the earlier anti-retriever inversion came from over-parameterized
    projections fighting the retriever on adversarial training data).
    m_emb dim equals PROJ_DIM so pairwise dots are well-defined.
    """

    def __init__(self, num_models: int = len(MODEL_REGISTRY),
                 wide_dim: int = WIDE_DIM):
        super().__init__()

        self.model_embeddings = nn.Embedding(num_models, MODEL_EMB_DIM)
        self.query_proj = nn.Linear(EMBED_DIM, PROJ_DIM)
        self.tool_proj = nn.Linear(EMBED_DIM, PROJ_DIM)

        self.wide_linear = nn.Linear(wide_dim, 1, bias=False)
        nn.init.zeros_(self.wide_linear.weight)

        deep_dim = 3 * PROJ_DIM + 3
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.deep_mlp[-1].weight)
        nn.init.zeros_(self.deep_mlp[-1].bias)

        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, wide_features: torch.Tensor,
                model_idx: torch.Tensor,
                query_emb: torch.Tensor,
                tool_emb: torch.Tensor) -> torch.Tensor:
        m = self.model_embeddings(model_idx)
        q = self.query_proj(query_emb)
        t = self.tool_proj(tool_emb)

        mq = (m * q).sum(dim=1, keepdim=True)
        mt = (m * t).sum(dim=1, keepdim=True)
        qt = (q * t).sum(dim=1, keepdim=True)

        deep_input = torch.cat([m, q, t, mq, mt, qt], dim=1)
        deep_logit = self.deep_mlp(deep_input)
        wide_logit = self.wide_linear(wide_features)
        return wide_logit + deep_logit + self.bias


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

        self.model = WideAndDeepModel()

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
        """Per-rollout update of the wide path only. Deep path is retrained
        in batches of 50 via maybe_batch_train().

        Wide aggregates (success_rate, usage, mt_sr) are updated first, then
        the wide linear parameters take a single gradient step on the new
        example. Deep parameters and model embeddings are untouched.
        """
        selected = feedback.get("selected", {})
        if not selected:
            return

        endpoint_key = f"{selected['server_id']}/{selected['tool_name']}"
        rating = feedback.get("rating", {})
        success = rating.get("success", False)
        model_name = feedback.get("model", self.model_name)

        # Wide aggregates always update first (features depend on them).
        self.wide.update(
            endpoint_key=endpoint_key,
            model_name=model_name,
            success=success,
        )

        # Single-example gradient step on wide-path parameters.
        self._online_wide_step(
            endpoint_key=endpoint_key,
            model_name=model_name,
            success=success,
            feedback=feedback,
        )

        self._replay_buffer.append(feedback)

    def _online_wide_step(self, endpoint_key: str, model_name: str,
                          success: bool, feedback: dict):
        """One SGD step on wide_linear using the current example only.
        Deep MLP, projections, and model embeddings are frozen."""
        idx = self.endpoint_to_idx.get(endpoint_key)
        if idx is None:
            return

        # Similarity from the retriever candidate list (needed for wide features).
        similarity = 0.0
        for c in feedback.get("retriever_candidates", []):
            if f"{c['server_id']}/{c['tool_name']}" == endpoint_key:
                similarity = c.get("similarity", 0.0)
                break

        query_text = feedback.get("query", feedback.get("task", ""))
        if not query_text:
            return
        q_emb = torch.tensor(self._safe_encode([query_text])[0], dtype=torch.float32).unsqueeze(0)
        m_idx = torch.tensor([MODEL_TO_IDX.get(model_name, 0)], dtype=torch.long)
        t_emb = self.endpoint_desc_embs[idx].unsqueeze(0)
        wide_feats = torch.tensor(
            [self.wide.get_features(endpoint_key, model_name, similarity)],
            dtype=torch.float32,
        )
        label = torch.tensor([1.0 if success else 0.0], dtype=torch.float32)

        if not hasattr(self, "_wide_optimizer"):
            self._wide_optimizer = torch.optim.Adam(
                self.model.wide_linear.parameters(), lr=1e-2, weight_decay=1e-4
            )

        self.model.train()
        # Freeze deep + bias + embeddings: only wide_linear grads propagate.
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.wide_linear.parameters():
            p.requires_grad = True

        self._wide_optimizer.zero_grad()
        logit = self.model(wide_feats, m_idx, q_emb, t_emb).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, label)
        loss.backward()
        self._wide_optimizer.step()

        # Restore grad flags so batch training can touch all params later.
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.eval()

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
                "tool_idx": idx,
                "similarity": similarity,
                "label": success,
            })

        return examples

    def _train(self, records: list[dict], epochs: int = 20, lr: float = 2e-3):
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
            tool_idxs.append(e["tool_idx"])
            model_idxs.append(MODEL_TO_IDX.get(e["model_name"], 0))
            labels.append(e["label"])

        q_tensor = torch.tensor(np.array(q_embs), dtype=torch.float32)
        model_idx_t = torch.tensor(model_idxs, dtype=torch.long)
        wide_tensor = torch.tensor(wide_feats, dtype=torch.float32)
        tool_tensor = self.endpoint_desc_embs[tool_idxs]
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        n_examples = len(examples)
        n_pos = sum(1 for l in labels if l > 0)

        # Paper: wide trained with FTRL+L1, deep with AdaGrad. We approximate
        # with parameter groups: higher LR + L1 weight decay on wide (encourages
        # memorization of the good feature combinations, sparsity on the rest),
        # lower LR + L2 on deep.
        wide_params = list(self.model.wide_linear.parameters())
        deep_params = [p for n, p in self.model.named_parameters()
                       if not n.startswith("wide_linear.")]
        optimizer = torch.optim.Adam([
            {"params": wide_params, "lr": lr * 5.0, "weight_decay": 1e-4},
            {"params": deep_params, "lr": lr,       "weight_decay": 1e-5},
        ])
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.model(wide_tensor, model_idx_t, q_tensor, tool_tensor).squeeze()
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
            self._train(records)

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

            with torch.no_grad():
                score_logit = self.model(wide_tensor, model_idx_tensor, query_tensor, tool_tensor)

            logit = score_logit.item()
            confidence = 1.0 / (1.0 + math.exp(-logit))  # sigmoid

            model_score = logit

            if explore:
                usage = self.wide.usage_count.get(key, 0)
                beta = 0.05
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

    def train_on_feedback(self, feedback_path: str, epochs: int = 20,
                          lr: float = 2e-3):
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
