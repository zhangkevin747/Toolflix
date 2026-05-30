from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tool_pool.faults import should_fail
from tool_pool.jsonl import write_jsonl
from tool_pool.models import FaultSpec


MODEL_REGISTRY = [
    "gpt-5.4-nano",
    "x-ai/grok-4.1-fast",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemma-4-26b-a4b-it",
    "qwen/qwen3.5-flash-02-23",
    "deepseek/deepseek-v3.2",
]


TOKEN_RE = re.compile(r"[a-z0-9]+")
CLEAN_VARIANTS = {"base_gold", "valid_schema_variant"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def listing_family(listing: dict[str, Any]) -> str:
    return listing.get("base_tool_id") or listing["listing_id"]


def listing_text(listing: dict[str, Any]) -> str:
    schema = listing.get("input_schema") or {}
    schema_terms = " ".join(schema.get("properties", {}).keys())
    return " ".join(
        str(part)
        for part in [
            listing.get("tool_name", ""),
            listing.get("server", ""),
            listing.get("category", ""),
            listing.get("description", ""),
            schema_terms,
        ]
        if part
    )


@dataclass
class RetrievalResult:
    listing_id: str
    score: float


class TfidfRetriever:
    """Small dependency-free lexical retriever for generated marketplace queries."""

    def __init__(self, listings: list[dict[str, Any]]) -> None:
        self.listing_ids = [listing["listing_id"] for listing in listings]
        docs = [Counter(tokenize(listing_text(listing))) for listing in listings]
        document_frequency: Counter[str] = Counter()
        for doc in docs:
            document_frequency.update(doc.keys())

        n_docs = max(len(docs), 1)
        self.idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1.0
            for term, df in document_frequency.items()
        }
        self.doc_vectors = [self._weight(doc) for doc in docs]
        self.doc_norms = [self._norm(vector) for vector in self.doc_vectors]

    def search(self, query: str, limit: int) -> list[RetrievalResult]:
        query_vector = self._weight(Counter(tokenize(query)))
        query_norm = self._norm(query_vector)
        if not query_vector or query_norm == 0:
            return []

        scores: list[RetrievalResult] = []
        for listing_id, doc_vector, doc_norm in zip(self.listing_ids, self.doc_vectors, self.doc_norms):
            if doc_norm == 0:
                continue
            dot = sum(weight * doc_vector.get(term, 0.0) for term, weight in query_vector.items())
            if dot <= 0:
                continue
            scores.append(RetrievalResult(listing_id=listing_id, score=dot / (query_norm * doc_norm)))
        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:limit]

    def _weight(self, counts: Counter[str]) -> dict[str, float]:
        return {term: (1.0 + math.log(count)) * self.idf.get(term, 1.0) for term, count in counts.items()}

    def _norm(self, vector: dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in vector.values()))


@dataclass
class DiscountedStats:
    pulls: float = 0.0
    reward: float = 0.0
    last_round: int = 0

    def decay_to(self, round_index: int, gamma: float) -> None:
        if self.last_round == 0:
            self.last_round = round_index
            return
        elapsed = round_index - self.last_round
        if elapsed <= 0:
            return
        factor = gamma ** elapsed
        self.pulls *= factor
        self.reward *= factor
        self.last_round = round_index

    def update(self, reward: float, round_index: int, gamma: float) -> None:
        self.decay_to(round_index, gamma)
        self.pulls += 1.0
        self.reward += reward

    @property
    def mean(self) -> float:
        return (0.5 + self.reward) / (1.0 + self.pulls)


def hashed_text_features(text: str, dim: int) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float32)
    counts = Counter(tokenize(text))
    if not counts:
        return vector
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * (1.0 + math.log(count))
    norm = torch.linalg.vector_norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


class ToolBanditNet(nn.Module):
    """Neural success model from Toolbandit.pdf.

    Input is the concatenation [query_embedding; model_embedding; tool_embedding].
    Retrieval similarity is only used to form the candidate pool, not as a rerank
    feature.
    """

    def __init__(
        self,
        num_tools: int,
        num_models: int,
        query_dim: int,
        model_emb_dim: int = 16,
        tool_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.model_embeddings = nn.Embedding(num_models, model_emb_dim)
        self.tool_embeddings = nn.Embedding(num_tools, tool_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim + model_emb_dim + tool_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Neutral cold start: before feedback, all candidates have predicted
        # success 0.5, so ties fall back to the retrieval order. Model/tool
        # embeddings and hidden layers still learn online from rewards.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        query_features: torch.Tensor,
        model_idx: torch.Tensor,
        tool_idx: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model_embeddings(model_idx)
        t = self.tool_embeddings(tool_idx)
        return self.mlp(torch.cat([query_features, m, t], dim=1))


@dataclass
class ToolBanditPolicy:
    """Neural contextual bandit from `Toolbandit.pdf`.

    Score:
      sigmoid(f_theta([query_embedding; model_embedding; tool_embedding]))
      + beta * sqrt(log(N_t + 1) / (n_t(tool) + 1))

    Retrieval similarity is used only by Stage 1 to produce candidates.
    """

    listings: list[dict[str, Any]]
    exploration_weight: float = 0.45
    neural_weight: float = 0.35  # Deprecated compatibility arg; exact ToolBandit does not use it.
    discount_gamma: float = 0.985
    query_dim: int = 128
    learning_rate: float = 0.01
    replay_window: int = 128
    replay_batch_size: int = 32
    listing_stats: dict[str, DiscountedStats] = field(default_factory=lambda: defaultdict(DiscountedStats))
    model_listing_stats: dict[tuple[str, str], DiscountedStats] = field(default_factory=lambda: defaultdict(DiscountedStats))
    discounted_total_pulls: float = 0.0
    _initialized: bool = False

    def __post_init__(self) -> None:
        self.listing_to_idx = {listing["listing_id"]: idx for idx, listing in enumerate(self.listings)}
        self.model_to_idx = {model: idx for idx, model in enumerate(MODEL_REGISTRY)}
        torch.manual_seed(13)
        self.net = ToolBanditNet(
            num_tools=len(self.listings),
            num_models=len(MODEL_REGISTRY),
            query_dim=self.query_dim,
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.replay_buffer: deque[dict[str, Any]] = deque(maxlen=self.replay_window)
        self._initialized = True

    def rerank(
        self,
        model_id: str,
        query: str,
        candidates: list[RetrievalResult],
        slate_size: int,
        round_index: int,
    ) -> list[dict[str, Any]]:
        ranked = []
        log_total = math.log(self.discounted_total_pulls + 2.0)
        query_features = hashed_text_features(query, self.query_dim).unsqueeze(0)
        model_idx = torch.tensor([self.model_to_idx.get(model_id, 0)], dtype=torch.long)
        for candidate in candidates:
            listing_stat = self.listing_stats[candidate.listing_id]
            listing_stat.decay_to(round_index, self.discount_gamma)
            model_stat = self.model_listing_stats[(model_id, candidate.listing_id)]
            model_stat.decay_to(round_index, self.discount_gamma)
            exploration = self.exploration_weight * math.sqrt(log_total / (listing_stat.pulls + 1.0))
            success_logit = self._success_logit(
                candidate.listing_id,
                query_features=query_features,
                model_idx=model_idx,
            )
            predicted_success = 1.0 / (1.0 + math.exp(-success_logit))
            score = predicted_success + exploration
            ranked.append(
                {
                    "listing_id": candidate.listing_id,
                    "retrieval_score": round(candidate.score, 6),
                    "policy_score": round(score, 6),
                    "predicted_success": round(predicted_success, 6),
                    "success_logit": round(success_logit, 6),
                    "discounted_tool_pulls": round(listing_stat.pulls, 6),
                    "discounted_model_tool_pulls": round(model_stat.pulls, 6),
                    "prior_mean_reward": round(listing_stat.mean, 6),
                    "prior_model_tool_mean_reward": round(model_stat.mean, 6),
                    "ucb_bonus": round(exploration, 6),
                }
            )
        ranked.sort(key=lambda row: row["policy_score"], reverse=True)
        return ranked[:slate_size]

    def update(
        self,
        model_id: str,
        listing_id: str,
        query: str,
        retrieval_score: float,
        reward: float,
        round_index: int,
    ) -> float | None:
        if listing_id not in self.listing_to_idx:
            return None

        listing_stat = self.listing_stats[listing_id]
        listing_stat.decay_to(round_index, self.discount_gamma)
        model_stat = self.model_listing_stats[(model_id, listing_id)]
        model_stat.decay_to(round_index, self.discount_gamma)
        self.replay_buffer.append(
            {
                "model_id": model_id,
                "listing_id": listing_id,
                "query": query,
                "reward": float(reward),
            }
        )
        batch = list(self.replay_buffer)[-min(len(self.replay_buffer), self.replay_batch_size):]
        query_features = torch.stack([hashed_text_features(item["query"], self.query_dim) for item in batch])
        model_idx = torch.tensor([self.model_to_idx.get(item["model_id"], 0) for item in batch], dtype=torch.long)
        tool_idx = torch.tensor([self.listing_to_idx[item["listing_id"]] for item in batch], dtype=torch.long)
        label = torch.tensor([[item["reward"]] for item in batch], dtype=torch.float32)

        self.net.train()
        self.optimizer.zero_grad()
        logit = self.net(query_features, model_idx, tool_idx)
        loss = self.loss_fn(logit, label)
        loss.backward()
        self.optimizer.step()
        self.net.eval()

        self.discounted_total_pulls = self.discount_gamma * self.discounted_total_pulls + 1.0
        listing_stat.update(reward, round_index, self.discount_gamma)
        model_stat.update(reward, round_index, self.discount_gamma)
        return float(loss.item())

    def _success_logit(
        self,
        listing_id: str,
        query_features: torch.Tensor,
        model_idx: torch.Tensor,
    ) -> float:
        tool_idx = torch.tensor([self.listing_to_idx.get(listing_id, 0)], dtype=torch.long)
        with torch.no_grad():
            return float(self.net(query_features, model_idx, tool_idx).item())


class SyntheticRewardModel:
    """Uses pool metadata to stand in for the future LLM reflection judge loop."""

    def __init__(self, listings_by_id: dict[str, dict[str, Any]], seed: int) -> None:
        self.listings_by_id = listings_by_id
        self.seed = seed

    def score(self, task: dict[str, Any], listing_id: str, attempt: int) -> tuple[float, str]:
        listing = self.listings_by_id[listing_id]
        if listing_family(listing) != task["gold_base_tool_id"]:
            return 0.0, "wrong_tool_family"

        variant_type = listing["variant_type"]
        if variant_type in CLEAN_VARIANTS:
            return 1.0, "clean_gold_success"
        if not variant_type.startswith("corrupted_"):
            return 0.0, "non_executable_or_distractor"

        fault = listing.get("fault_spec")
        if not fault:
            return 0.0, "corrupted_missing_fault_spec"
        spec = FaultSpec(**fault)
        deterministic_attempt = self._attempt_key(task["task_id"], listing_id, attempt)
        if should_fail(spec, deterministic_attempt):
            return 0.0, f"{variant_type}_fired"
        return 1.0, f"{variant_type}_passed_through"

    def _attempt_key(self, task_id: str, listing_id: str, attempt: int) -> int:
        payload = f"{self.seed}:{task_id}:{listing_id}:{attempt}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return int(digest[:8], 16)


def candidate_contains_clean_gold(
    candidates: list[RetrievalResult],
    listings_by_id: dict[str, dict[str, Any]],
    gold_base_tool_id: str,
) -> bool:
    for candidate in candidates:
        listing = listings_by_id[candidate.listing_id]
        if listing_family(listing) == gold_base_tool_id and listing["variant_type"] in CLEAN_VARIANTS:
            return True
    return False


def run_training(
    tasks_path: Path,
    listings_path: Path,
    out_dir: Path,
    rounds: int | None,
    candidate_count: int,
    slate_size: int,
    seed: int,
    discount_gamma: float = 0.985,
    exploration_weight: float = 0.45,
    neural_weight: float = 0.35,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    tasks = load_jsonl(tasks_path)
    listings = load_jsonl(listings_path)
    listings_by_id = {listing["listing_id"]: listing for listing in listings}

    rng = random.Random(seed)
    ordered_tasks = list(tasks)
    rng.shuffle(ordered_tasks)
    if rounds is not None:
        ordered_tasks = ordered_tasks[: min(rounds, len(ordered_tasks))]

    retriever = TfidfRetriever(listings)
    policy = ToolBanditPolicy(
        listings=listings,
        discount_gamma=discount_gamma,
        exploration_weight=exploration_weight,
        neural_weight=neural_weight,
        learning_rate=learning_rate,
    )
    reward_model = SyntheticRewardModel(listings_by_id, seed=seed)

    events: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    candidate_hits = 0
    clean_gold_selections = 0
    corrupted_gold_selections = 0
    wrong_family_selections = 0

    for round_index, task in enumerate(ordered_tasks, start=1):
        model_id = MODEL_REGISTRY[(round_index - 1) % len(MODEL_REGISTRY)]
        query = task["marketplace_query"]
        candidates = retriever.search(query, limit=candidate_count)
        has_clean_gold = candidate_contains_clean_gold(candidates, listings_by_id, task["gold_base_tool_id"])
        candidate_hits += int(has_clean_gold)

        slate = policy.rerank(
            model_id=model_id,
            query=query,
            candidates=candidates,
            slate_size=slate_size,
            round_index=round_index,
        )
        selected = slate[0] if slate else None
        policy_loss = None
        if selected is None:
            reward = 0.0
            reward_reason = "no_retrieval_candidates"
            selected_listing_id = None
            selected_variant_type = None
        else:
            selected_listing_id = selected["listing_id"]
            selected_listing = listings_by_id[selected_listing_id]
            selected_variant_type = selected_listing["variant_type"]
            reward, reward_reason = reward_model.score(task, selected_listing_id, round_index)
            policy_loss = policy.update(
                model_id=model_id,
                listing_id=selected_listing_id,
                query=query,
                retrieval_score=float(selected["retrieval_score"]),
                reward=reward,
                round_index=round_index,
            )

            selected_family = listing_family(selected_listing)
            if selected_family != task["gold_base_tool_id"]:
                wrong_family_selections += 1
            elif selected_variant_type in CLEAN_VARIANTS:
                clean_gold_selections += 1
            elif str(selected_variant_type).startswith("corrupted_"):
                corrupted_gold_selections += 1

        rewards.append(reward)
        window = rewards[-50:]
        event = {
            "round": round_index,
            "task_id": task["task_id"],
            "model_id": model_id,
            "gold_base_tool_id": task["gold_base_tool_id"],
            "marketplace_query": query,
            "candidate_count": len(candidates),
            "candidate_has_clean_gold": has_clean_gold,
            "slate": slate,
            "selected_listing_id": selected_listing_id,
            "selected_variant_type": selected_variant_type,
            "reward": reward,
            "reward_reason": reward_reason,
            "policy_loss": round(policy_loss, 6) if policy_loss is not None else None,
            "cumulative_reward": round(sum(rewards), 6),
            "mean_reward": round(sum(rewards) / len(rewards), 6),
            "rolling_50_mean_reward": round(sum(window) / len(window), 6),
        }
        events.append(event)
        learning_rows.append(
            {
                "round": round_index,
                "reward": reward,
                "mean_reward": event["mean_reward"],
                "rolling_50_mean_reward": event["rolling_50_mean_reward"],
                "candidate_has_clean_gold": int(has_clean_gold),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "events.jsonl", events)
    with (out_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["round", "reward", "mean_reward", "rolling_50_mean_reward", "candidate_has_clean_gold"],
        )
        writer.writeheader()
        writer.writerows(learning_rows)

    summary = {
        "status": "ready",
        "mode": "toolbandit_contextual_bandit_v1",
        "tasks_path": str(tasks_path.resolve()),
        "listings_path": str(listings_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "seed": seed,
        "rounds": len(events),
        "unique_task_stream": True,
        "candidate_count": candidate_count,
        "slate_size": slate_size,
        "discount_gamma": discount_gamma,
        "exploration_weight": exploration_weight,
        "neural_weight": neural_weight,
        "learning_rate": learning_rate,
        "models": MODEL_REGISTRY,
        "total_reward": round(sum(rewards), 6),
        "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "rolling_50_mean_reward": round(sum(rewards[-50:]) / min(len(rewards), 50), 6) if rewards else 0.0,
        "candidate_clean_gold_recall": round(candidate_hits / len(events), 6) if events else 0.0,
        "clean_gold_selections": clean_gold_selections,
        "corrupted_gold_selections": corrupted_gold_selections,
        "wrong_family_selections": wrong_family_selections,
        "event_log": str((out_dir / "events.jsonl").resolve()),
        "learning_curve": str((out_dir / "learning_curve.csv").resolve()),
        "notes": [
            "This v1 run is cheap and offline: it uses known listing metadata for reward instead of calling an LLM reflection judge.",
            "Each task is used at most once in this run.",
            "The selected arm is the top-ranked listing from the ToolBandit contextual bandit slate.",
            "The policy follows Toolbandit.pdf: predicted success from f_theta([query; model; tool]) plus discounted per-tool UCB. Retrieval similarity is only used for candidate generation.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
