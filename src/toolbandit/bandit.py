"""Stage 2: the ToolBandit. This is the heart of the project.

It is a *contextual bandit*: each round it must pick tools to show, then it sees
the reward for the ONE tool that got used, and learns from it. Over many rounds
it learns which tools actually work, and which work for which model.

Two parts decide a tool's rank:

  predicted_success  a small neural net guesses "will this tool work for this
                     (query, model)?" as a probability in [0, 1]. It learns from
                     every reward we feed it.

  exploration_bonus  a "give rarely-tried tools a chance" term (UCB). Tools we've
                     pulled few times get a bonus so the bandit doesn't lock onto
                     an early favorite and ignore untested tools.

  score = predicted_success + exploration_bonus

The net's inputs are only [query, model, tool]. Retrieval similarity is NOT a
feature here, it was already used in stage 1 to pick the candidates. This matches
the method in Toolbandit.pdf.

Cold start: the net's final layer starts at zero, so before any feedback every
tool's predicted_success is exactly 0.5 and ranking falls back to retrieval order.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .data import MODELS, words
from .retriever import Candidate


# ---------------------------------------------------------------------------
# Turning a query string into a fixed-size vector the net can read.
# We use the "hashing trick": hash each word to a bucket and add it up. No
# vocabulary to store, no training needed for this part.
# ---------------------------------------------------------------------------
def embed_query(text: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim)
    counts = Counter(words(text))
    for word, count in counts.items():
        digest = hashlib.sha256(word.encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign * (1.0 + math.log(count))
    length = torch.linalg.vector_norm(vec)
    return vec / length if length > 0 else vec


# ---------------------------------------------------------------------------
# Per-tool memory: how often a tool was tried and how well it did, with old
# results fading over time (so the bandit can react if a tool starts failing).
# ---------------------------------------------------------------------------
@dataclass
class ToolMemory:
    pulls: float = 0.0       # how many times tried (decayed)
    reward: float = 0.0      # total reward earned (decayed)
    last_round: int = 0

    def fade(self, round_index: int, gamma: float) -> None:
        """Shrink old counts toward zero. gamma<1 means recent rounds matter more."""
        if self.last_round == 0:
            self.last_round = round_index
            return
        elapsed = round_index - self.last_round
        if elapsed > 0:
            factor = gamma ** elapsed
            self.pulls *= factor
            self.reward *= factor
            self.last_round = round_index

    def record(self, reward: float, round_index: int, gamma: float) -> None:
        self.fade(round_index, gamma)
        self.pulls += 1.0
        self.reward += reward


# ---------------------------------------------------------------------------
# The neural success model: predicts a success logit from (query, model, tool).
# ---------------------------------------------------------------------------
class SuccessNet(nn.Module):
    def __init__(self, num_tools: int, num_models: int, query_dim: int) -> None:
        super().__init__()
        model_dim, tool_dim = 16, 32
        # Each model and each tool gets its own learned vector.
        self.model_emb = nn.Embedding(num_models, model_dim)
        self.tool_emb = nn.Embedding(num_tools, tool_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim + model_dim + tool_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Start the last layer at zero -> every tool predicts 0.5 before learning.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, query: torch.Tensor, model_idx: torch.Tensor, tool_idx: torch.Tensor) -> torch.Tensor:
        parts = [query, self.model_emb(model_idx), self.tool_emb(tool_idx)]
        return self.mlp(torch.cat(parts, dim=1))


@dataclass
class ScoredTool:
    """One ranked entry in the slate shown to the caller."""
    listing_id: str
    score: float             # predicted_success + exploration_bonus
    predicted_success: float
    exploration_bonus: float
    retrieval_score: float


class ToolBandit:
    def __init__(
        self,
        listings: list[dict[str, Any]],
        exploration_weight: float = 0.45,
        gamma: float = 0.985,        # how fast old per-tool stats fade
        query_dim: int = 128,
        learning_rate: float = 0.01,
        replay_size: int = 128,      # remember the last N rewards to train on
        batch_size: int = 32,        # how many of them to train on each update
    ) -> None:
        self.exploration_weight = exploration_weight
        self.gamma = gamma
        self.query_dim = query_dim
        self.batch_size = batch_size

        self.tool_index = {l["listing_id"]: i for i, l in enumerate(listings)}
        self.model_index = {m: i for i, m in enumerate(MODELS)}

        torch.manual_seed(13)
        self.net = SuccessNet(len(listings), len(MODELS), query_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.replay: deque[dict[str, Any]] = deque(maxlen=replay_size)

        self.memory: dict[str, ToolMemory] = defaultdict(ToolMemory)
        self.total_pulls = 0.0  # decayed count of all pulls, for the UCB bonus

    def rerank(self, model_id: str, query: str, candidates: list[Candidate], slate_size: int, round_index: int) -> list[ScoredTool]:
        """Score every candidate and return the best `slate_size` of them."""
        query_vec = embed_query(query, self.query_dim).unsqueeze(0)
        model_idx = torch.tensor([self.model_index.get(model_id, 0)])
        log_total = math.log(self.total_pulls + 2.0)

        scored: list[ScoredTool] = []
        for cand in candidates:
            mem = self.memory[cand.listing_id]
            mem.fade(round_index, self.gamma)

            predicted = self._predict_success(cand.listing_id, query_vec, model_idx)
            # Fewer pulls -> bigger bonus. Encourages trying neglected tools.
            bonus = self.exploration_weight * math.sqrt(log_total / (mem.pulls + 1.0))

            scored.append(ScoredTool(
                listing_id=cand.listing_id,
                score=predicted + bonus,
                predicted_success=predicted,
                exploration_bonus=bonus,
                retrieval_score=cand.score,
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:slate_size]

    def learn(self, model_id: str, listing_id: str, query: str, reward: float, round_index: int) -> float | None:
        """Feed back the reward for the one tool that was used, and take a
        gradient step. Returns the training loss (for logging)."""
        if listing_id not in self.tool_index:
            return None

        # Update this tool's fading memory (used by the exploration bonus).
        self.memory[listing_id].record(reward, round_index, self.gamma)
        self.total_pulls = self.gamma * self.total_pulls + 1.0

        # Train the net on a recent batch of (query, model, tool) -> reward.
        self.replay.append({"model_id": model_id, "listing_id": listing_id, "query": query, "reward": float(reward)})
        batch = list(self.replay)[-self.batch_size:]
        queries = torch.stack([embed_query(b["query"], self.query_dim) for b in batch])
        models = torch.tensor([self.model_index.get(b["model_id"], 0) for b in batch])
        tools = torch.tensor([self.tool_index[b["listing_id"]] for b in batch])
        labels = torch.tensor([[b["reward"]] for b in batch])

        self.net.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.net(queries, models, tools), labels)
        loss.backward()
        self.optimizer.step()
        self.net.eval()
        return float(loss.item())

    def _predict_success(self, listing_id: str, query_vec: torch.Tensor, model_idx: torch.Tensor) -> float:
        tool_idx = torch.tensor([self.tool_index.get(listing_id, 0)])
        with torch.no_grad():
            logit = float(self.net(query_vec, model_idx, tool_idx).item())
        return 1.0 / (1.0 + math.exp(-logit))  # sigmoid -> probability
