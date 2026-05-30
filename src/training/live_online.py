from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from tool_pool.adapters import adapt_arguments
from tool_pool.faults import should_fail
from tool_pool.models import AdapterSpec, FaultSpec
from training.simulated_online import (
    MODEL_REGISTRY,
    CLEAN_VARIANTS,
    TfidfRetriever,
    ToolBanditPolicy,
    candidate_contains_clean_gold,
    listing_family,
    load_jsonl,
)


OPENAI_NATIVE_MODELS = {
    "gpt-5.4-nano",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def json_loads_lenient(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}


def client_for_model(model_id: str) -> OpenAI:
    if model_id in OPENAI_NATIVE_MODELS:
        return OpenAI()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


def create_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> Any:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    try:
        return client.chat.completions.create(
            **kwargs,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:
        return client.chat.completions.create(
            **kwargs,
            max_completion_tokens=max_tokens,
        )


@dataclass
class SelectionResult:
    listing_id: str | None
    tool_alias: str | None
    arguments: dict[str, Any]
    raw_response: str
    error: str | None = None
    caller_payload: dict[str, Any] | None = None
    internal_alias_map: dict[str, str] | None = None


LEAKAGE_RE = re.compile(
    r"(^|[^a-z0-9])("
    r"corrupted|schema_mismatch|auth_quota|upstream_api|protocol_bug|"
    r"valid_schema_variant|base_gold|background_distractor|"
    r"mild_variant|medium_variant|aggressive_variant|"
    r"valid[._-](mild|medium|aggressive)|"
    r"(timeout[._-]?[0-9]+)"
    r")([^a-z0-9]|$)",
    re.IGNORECASE,
)


SYNTHETIC_NAME_PATTERNS = [
    re.compile(r"_(mild|medium|aggressive)_variant_\d+$", re.IGNORECASE),
    re.compile(r"_valid_(mild|medium|aggressive)_\d+$", re.IGNORECASE),
    re.compile(r"_(schema_mismatch|timeout|auth_quota|upstream_api|protocol_bug)_\d+$", re.IGNORECASE),
]


class ModelToolSelector:
    def __init__(self) -> None:
        self.clients: dict[str, OpenAI] = {}
        self._client_lock = threading.Lock()

    def _client_for(self, model_id: str) -> OpenAI:
        with self._client_lock:
            if model_id not in self.clients:
                self.clients[model_id] = client_for_model(model_id)
            return self.clients[model_id]

    def select(
        self,
        model_id: str,
        task: dict[str, Any],
        slate: list[dict[str, Any]],
        listings_by_id: dict[str, dict[str, Any]],
    ) -> SelectionResult:
        client = self._client_for(model_id)
        options, alias_to_listing_id = self.build_options(slate, listings_by_id)
        leakage = detect_payload_leakage({"tool_options": options})
        if leakage:
            return SelectionResult(None, None, {}, "", f"caller_payload_leakage: {leakage[:5]}", {"tool_options": options}, alias_to_listing_id)
        caller_payload = {
            "user_task": task["user_task"],
            "marketplace_query": task["marketplace_query"],
            "tool_options": options,
            "response_schema": {"tool_alias": "one of the option tool_alias values", "arguments": {}},
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are choosing exactly one marketplace tool for a user task. "
                    "Return only valid JSON with keys tool_alias and arguments. "
                    "tool_alias must be copied exactly from the provided options. "
                    "arguments must match that tool's input schema. "
                    "toolbandit_score is the marketplace policy score for that option. "
                    "Choose the tool whose score, description, and schema best fit the user task."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(caller_payload, ensure_ascii=False),
            },
        ]
        leakage = detect_payload_leakage(messages)
        if leakage:
            return SelectionResult(None, None, {}, "", f"caller_message_leakage: {leakage[:5]}", caller_payload, alias_to_listing_id)
        try:
            response = create_chat_completion(client, model_id, messages, max_tokens=500)
            raw = response.choices[0].message.content or ""
            payload = json_loads_lenient(raw)
            tool_alias = payload.get("tool_alias")
            if tool_alias not in alias_to_listing_id:
                return SelectionResult(None, tool_alias if isinstance(tool_alias, str) else None, {}, raw, f"invalid_tool_alias: {tool_alias}", caller_payload, alias_to_listing_id)
            args = payload.get("arguments")
            return SelectionResult(alias_to_listing_id[tool_alias], tool_alias, args if isinstance(args, dict) else {}, raw, None, caller_payload, alias_to_listing_id)
        except Exception as exc:
            return SelectionResult(None, None, {}, "", str(exc), caller_payload, alias_to_listing_id)

    def build_options(
        self,
        slate: list[dict[str, Any]],
        listings_by_id: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        options = []
        alias_to_listing_id = {}
        for idx, row in enumerate(slate, start=1):
            alias = f"tool_{idx}"
            alias_to_listing_id[alias] = row["listing_id"]
            options.append(self._option_payload(alias, idx, row, listings_by_id[row["listing_id"]]))
        return options, alias_to_listing_id

    def _option_payload(self, alias: str, rank: int, slate_row: dict[str, Any], listing: dict[str, Any]) -> dict[str, Any]:
        return {
            "rank": rank,
            "tool_alias": alias,
            "toolbandit_score": round(float(slate_row.get("policy_score", 0.0)), 6),
            "server": listing.get("server"),
            "description": sanitize_public_text(str(listing.get("description", "")))[:900],
            "input_schema": sanitize_schema(listing.get("input_schema") or {"type": "object", "properties": {}}),
        }


def sanitize_public_text(text: str) -> str:
    cleaned = text
    for pattern in SYNTHETIC_NAME_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    replacements = {
        "schema_mismatch": "schema",
        "auth_quota": "authorization",
        "upstream_api": "upstream service",
        "protocol_bug": "protocol",
        "corrupted": "service",
        "valid_schema_variant": "tool",
        "base_gold": "tool",
        "background_distractor": "tool",
    }
    for source, target in replacements.items():
        cleaned = re.sub(source, target, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(valid|variant)\b", "tool", cleaned, flags=re.IGNORECASE)
    return cleaned


def sanitize_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        sanitized = {}
        for key, value in schema.items():
            if key in {"title", "description"} and isinstance(value, str):
                sanitized[key] = sanitize_public_text(value)
            elif key == "properties" and isinstance(value, dict):
                sanitized[key] = {
                    prop_name: sanitize_schema(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
            elif key == "required" and isinstance(value, list):
                sanitized[key] = list(value)
            else:
                sanitized[key] = sanitize_schema(value)
        return sanitized
    if isinstance(schema, list):
        return [sanitize_schema(item) for item in schema]
    return schema


def detect_payload_leakage(payload: Any) -> list[str]:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    matches = []
    for match in LEAKAGE_RE.finditer(text):
        token = match.group(2)
        if token not in matches:
            matches.append(token)
    return matches


@dataclass
class ExecutionPayload:
    ok: bool
    output: dict[str, Any]
    adapted_arguments: dict[str, Any]
    failure_type: str | None


class CachedMarketplaceExecutor:
    def __init__(self, listings_by_id: dict[str, dict[str, Any]], live_outputs_by_base: dict[str, str]) -> None:
        self.listings_by_id = listings_by_id
        self.live_outputs_by_base = live_outputs_by_base

    def execute(self, listing_id: str, arguments: dict[str, Any], attempt: int) -> ExecutionPayload:
        listing = self.listings_by_id[listing_id]
        missing = missing_required_fields(listing.get("input_schema") or {}, arguments)
        if missing:
            return ExecutionPayload(
                ok=False,
                output={"error": "schema_validation_failed", "missing_required_fields": missing},
                adapted_arguments={},
                failure_type="schema_validation_failed",
            )

        adapter = adapter_from_listing(listing)
        adapted = adapt_arguments(arguments, adapter)
        fault = fault_from_listing(listing)
        if fault and should_fail(fault, attempt=attempt):
            return ExecutionPayload(
                ok=False,
                output=dict(fault.failure_payload),
                adapted_arguments=adapted,
                failure_type=fault.failure_type,
            )

        base_id = listing.get("base_tool_id")
        if not base_id or base_id not in self.live_outputs_by_base:
            return ExecutionPayload(
                ok=False,
                output={"error": "tool_not_executable_in_cached_experiment"},
                adapted_arguments=adapted,
                failure_type="not_executable",
            )
        return ExecutionPayload(
            ok=True,
            output={"cached_live_output_preview": self.live_outputs_by_base[base_id]},
            adapted_arguments=adapted,
            failure_type=None,
        )


class ReflectionJudge:
    def __init__(self, judge_model: str) -> None:
        self.judge_model = judge_model
        self.clients: dict[str, OpenAI] = {}

    def judge(
        self,
        model_id: str,
        task: dict[str, Any],
        listing: dict[str, Any],
        execution: ExecutionPayload,
    ) -> tuple[float, dict[str, Any]]:
        judge_model = model_id if self.judge_model == "self" else self.judge_model
        if judge_model not in self.clients:
            self.clients[judge_model] = client_for_model(judge_model)

        messages = [
            {
                "role": "system",
                "content": (
                    "You judge whether a tool call returned enough correct raw information for the user's task. "
                    "Return only JSON: {\"success\": true/false, \"reasoning\": \"short\"}. "
                    "Success requires useful, task-relevant information. Errors, empty output, wrong topic, or malformed tool failures are not success."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_task": task["user_task"],
                        "expected_evidence_hint": task.get("evidence_quote"),
                        "selected_tool": {
                            "server": listing.get("server"),
                            "description": sanitize_public_text(str(listing.get("description", "")))[:900],
                        },
                        "execution_ok": execution.ok,
                        "tool_output": execution.output,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            response = create_chat_completion(self.clients[judge_model], judge_model, messages, max_tokens=300)
            raw = response.choices[0].message.content or ""
            payload = json_loads_lenient(raw)
            success = bool(payload.get("success"))
            return (1.0 if success else 0.0), {
                "judge_model": judge_model,
                "success": success,
                "reasoning": payload.get("reasoning", ""),
                "raw": raw[:1000],
            }
        except Exception as exc:
            return 0.0, {
                "judge_model": judge_model,
                "success": False,
                "reasoning": f"judge_error: {exc}",
                "raw": "",
            }


def missing_required_fields(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    missing = []
    for field in schema.get("required") or []:
        if field not in arguments:
            missing.append(field)
    return missing


def adapter_from_listing(listing: dict[str, Any]) -> AdapterSpec | None:
    payload = listing.get("adapter")
    return AdapterSpec(**payload) if payload else None


def fault_from_listing(listing: dict[str, Any]) -> FaultSpec | None:
    payload = listing.get("fault_spec")
    return FaultSpec(**payload) if payload else None


def load_live_outputs(path: Path) -> dict[str, str]:
    rows = load_jsonl(path)
    return {
        row["tool_id"]: row["output_preview"]
        for row in rows
        if row.get("status") == "pass" and row.get("output_preview") is not None
    }


def fallback_args(task: dict[str, Any], listing: dict[str, Any]) -> dict[str, Any]:
    if listing_family(listing) != task["gold_base_tool_id"]:
        return {}
    adapter = adapter_from_listing(listing)
    return visible_args_from_base(task.get("fixture_args") or {}, adapter)


def visible_args_from_base(base_args: dict[str, Any], adapter: AdapterSpec | None) -> dict[str, Any]:
    if not adapter:
        return dict(base_args)
    out: dict[str, Any] = {}
    for visible_path, base_field in adapter.arg_map.items():
        if base_field in base_args:
            set_path(out, visible_path, base_args[base_field])
    return out


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            nested = {}
            current[part] = nested
        current = nested
    current[parts[-1]] = value


def run_live_online_training(
    tasks_path: Path,
    listings_path: Path,
    live_outputs_path: Path,
    out_dir: Path,
    rounds: int | None,
    candidate_count: int,
    slate_size: int,
    seed: int,
    judge_model: str,
    discount_gamma: float,
    exploration_weight: float,
    neural_weight: float,
    learning_rate: float,
    sleep_seconds: float = 0.0,
    reward_mode: str = "llm",
    verbose_rollouts: bool = False,
    concurrency: int = 1,
) -> dict[str, Any]:
    tasks = load_jsonl(tasks_path)
    listings = load_jsonl(listings_path)
    listings_by_id = {listing["listing_id"]: listing for listing in listings}
    live_outputs = load_live_outputs(live_outputs_path)

    rng = random.Random(seed)
    stream = list(tasks)
    rng.shuffle(stream)
    if rounds is not None:
        stream = stream[: min(rounds, len(stream))]

    retriever = TfidfRetriever(listings)
    policy = ToolBanditPolicy(
        listings=listings,
        discount_gamma=discount_gamma,
        exploration_weight=exploration_weight,
        neural_weight=neural_weight,
        learning_rate=learning_rate,
    )
    selector = ModelToolSelector()
    executor = CachedMarketplaceExecutor(listings_by_id, live_outputs)
    judge = ReflectionJudge(judge_model)

    out_dir.mkdir(parents=True, exist_ok=True)
    event_path = out_dir / "events.jsonl"
    curve_path = out_dir / "learning_curve.csv"
    events_handle = event_path.open("w", encoding="utf-8")
    curve_handle = curve_path.open("w", encoding="utf-8")
    curve_handle.write("round,reward,mean_reward,rolling_50_mean_reward,candidate_has_clean_gold,selection_error\n")

    rewards: list[float] = []
    candidate_hits = 0
    selection_errors = 0
    clean_gold_selections = 0
    corrupted_gold_selections = 0
    wrong_family_selections = 0
    concurrency = max(1, int(concurrency))

    def prepare_round(round_index: int, task: dict[str, Any]) -> dict[str, Any]:
        model_id = MODEL_REGISTRY[(round_index - 1) % len(MODEL_REGISTRY)]
        query = task["marketplace_query"]
        candidates = retriever.search(query, limit=candidate_count)
        has_clean_gold = candidate_contains_clean_gold(candidates, listings_by_id, task["gold_base_tool_id"])
        slate = policy.rerank(model_id, query, candidates, slate_size=slate_size, round_index=round_index)
        return {
            "round_index": round_index,
            "task": task,
            "model_id": model_id,
            "query": query,
            "candidates": candidates,
            "has_clean_gold": has_clean_gold,
            "slate": slate,
        }

    def select_prepared(prepared: dict[str, Any]) -> SelectionResult:
        slate = prepared["slate"]
        if not slate:
            return SelectionResult(None, None, {}, "", "empty_slate")
        return selector.select(prepared["model_id"], prepared["task"], slate, listings_by_id)

    def run_batch_selection(batch: list[dict[str, Any]]) -> list[SelectionResult]:
        if concurrency == 1 or len(batch) <= 1:
            return [select_prepared(prepared) for prepared in batch]
        with ThreadPoolExecutor(max_workers=min(concurrency, len(batch))) as pool:
            return list(pool.map(select_prepared, batch))

    try:
        for batch_start in range(0, len(stream), concurrency):
            batch = [
                prepare_round(batch_start + offset + 1, task)
                for offset, task in enumerate(stream[batch_start: batch_start + concurrency])
            ]
            selections = run_batch_selection(batch)

            for prepared, selection in zip(batch, selections):
                round_index = prepared["round_index"]
                task = prepared["task"]
                model_id = prepared["model_id"]
                query = prepared["query"]
                candidates = prepared["candidates"]
                has_clean_gold = prepared["has_clean_gold"]
                slate = prepared["slate"]
                candidate_hits += int(has_clean_gold)

                selected_listing_id = selection.listing_id
                selected_from_fallback = False
                if not selected_listing_id and slate:
                    selection_errors += 1
                    selected_listing_id = slate[0]["listing_id"]
                    selected_from_fallback = True
                    selection.tool_alias = "tool_1"
                    selection.arguments = fallback_args(task, listings_by_id[selected_listing_id])

                policy_loss = None
                reward = 0.0
                judge_payload: dict[str, Any] = {}
                execution_payload: ExecutionPayload | None = None
                selected_listing: dict[str, Any] | None = None
                selected_variant_type = None
                reward_reason = "no_selection"

                if selected_listing_id:
                    selected_listing = listings_by_id[selected_listing_id]
                    selected_variant_type = selected_listing["variant_type"]
                    execution_payload = executor.execute(selected_listing_id, selection.arguments, attempt=round_index)
                    if reward_mode == "metadata":
                        reward, judge_payload = metadata_reward(task, selected_listing, execution_payload)
                        reward_reason = "metadata_success" if reward else "metadata_failure"
                    else:
                        reward, judge_payload = judge.judge(model_id, task, selected_listing, execution_payload)
                        reward_reason = "llm_reflection_judge_success" if reward else "llm_reflection_judge_failure"

                    selected_row = next((row for row in slate if row["listing_id"] == selected_listing_id), None)
                    retrieval_score = float(selected_row["retrieval_score"]) if selected_row else 0.0
                    policy_loss = policy.update(
                        model_id=model_id,
                        listing_id=selected_listing_id,
                        query=query,
                        retrieval_score=retrieval_score,
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
                    "selected_tool_alias": selection.tool_alias,
                    "selected_variant_type": selected_variant_type,
                    "selected_from_fallback": selected_from_fallback,
                    "selection_error": selection.error,
                    "selection_arguments": selection.arguments,
                    "execution": {
                        "ok": execution_payload.ok,
                        "failure_type": execution_payload.failure_type,
                        "adapted_arguments": execution_payload.adapted_arguments,
                        "output_preview": json.dumps(execution_payload.output, ensure_ascii=False, default=str)[:1200],
                    } if execution_payload else None,
                    "judge": judge_payload,
                    "reward": reward,
                    "reward_reason": reward_reason,
                    "policy_loss": round(policy_loss, 6) if policy_loss is not None else None,
                    "cumulative_reward": round(sum(rewards), 6),
                    "mean_reward": round(sum(rewards) / len(rewards), 6),
                    "rolling_50_mean_reward": round(sum(window) / len(window), 6),
                }
                if verbose_rollouts:
                    event["verbose_rollout"] = {
                        "user_task": task["user_task"],
                        "ground_truth_answer": task.get("ground_truth_answer"),
                        "evidence_quote": task.get("evidence_quote"),
                        "caller_visible_payload": selection.caller_payload,
                        "caller_raw_response": selection.raw_response,
                        "internal_alias_map": selection.internal_alias_map,
                        "selected_internal_listing": selected_listing if selected_listing_id else None,
                        "note": "Training reward is attached only to the selected tool. Unselected slate entries are logged for inspection but not labeled.",
                    }
                events_handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
                events_handle.flush()
                curve_handle.write(
                    f"{round_index},{reward},{event['mean_reward']},{event['rolling_50_mean_reward']},{int(has_clean_gold)},{int(selection.error is not None)}\n"
                )
                curve_handle.flush()
                print(
                    f"[exp1] {round_index}/{len(stream)} model={model_id} reward={reward:.0f} mean={event['mean_reward']:.3f} selected={selected_listing_id}",
                    flush=True,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    finally:
        events_handle.close()
        curve_handle.close()

    summary = {
        "status": "ready",
        "mode": "experiment_1_live_model_selection_cached_execution",
        "tasks_path": str(tasks_path.resolve()),
        "listings_path": str(listings_path.resolve()),
        "live_outputs_path": str(live_outputs_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "rounds": len(rewards),
        "seed": seed,
        "candidate_count": candidate_count,
        "slate_size": slate_size,
        "models": MODEL_REGISTRY,
        "judge_model": judge_model,
        "reward_mode": reward_mode,
        "concurrency": concurrency,
        "discount_gamma": discount_gamma,
        "exploration_weight": exploration_weight,
        "neural_weight": neural_weight,
        "learning_rate": learning_rate,
        "total_reward": round(sum(rewards), 6),
        "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "rolling_50_mean_reward": round(sum(rewards[-50:]) / min(len(rewards), 50), 6) if rewards else 0.0,
        "candidate_clean_gold_recall": round(candidate_hits / len(rewards), 6) if rewards else 0.0,
        "selection_errors": selection_errors,
        "clean_gold_selections": clean_gold_selections,
        "corrupted_gold_selections": corrupted_gold_selections,
        "wrong_family_selections": wrong_family_selections,
        "event_log": str(event_path.resolve()),
        "learning_curve": str(curve_path.resolve()),
        "notes": [
            "Six calling models choose from the ToolBandit top-5 slate.",
            "Execution uses cached live outputs from prior base-tool validation to avoid reconnecting MCP servers every round.",
            "reward_mode='llm' uses reflection judging; judge_model='self' means the same caller model judges its selected tool output.",
            "reward_mode='metadata' uses validated pool semantics for faster online training with real model choices.",
            "When concurrency > 1, model selections are batched; rewards and policy updates are still applied in round order after each batch returns.",
            "Caller-visible tool options include each option's ToolBandit score.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def metadata_reward(
    task: dict[str, Any],
    listing: dict[str, Any],
    execution: ExecutionPayload,
) -> tuple[float, dict[str, Any]]:
    if not execution.ok:
        return 0.0, {
            "judge_model": "metadata",
            "success": False,
            "reasoning": f"execution failed: {execution.failure_type}",
        }
    if listing_family(listing) != task["gold_base_tool_id"]:
        return 0.0, {
            "judge_model": "metadata",
            "success": False,
            "reasoning": "selected listing belongs to the wrong base tool family",
        }
    return 1.0, {
        "judge_model": "metadata",
        "success": True,
        "reasoning": "selected listing belongs to the gold base tool family and execution produced cached validated output",
    }
