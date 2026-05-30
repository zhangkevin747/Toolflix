from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "task_id",
    "user_task",
    "marketplace_query",
    "gold_base_tool_id",
    "gold_listing_ids",
    "fixture_args",
    "ground_truth_answer",
    "evidence_quote",
    "expected_answer_source",
    "reward_source",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_tasks(
    tasks: list[dict[str, Any]],
    *,
    base_tool_ids: set[str],
    listing_ids: set[str],
    live_outputs_by_tool: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, task in enumerate(tasks):
        prefix = f"task[{index}]"
        missing = sorted(REQUIRED_FIELDS - set(task))
        if missing:
            errors.append(f"{prefix}: missing fields {missing}")
            continue
        task_id = str(task["task_id"])
        if task_id in seen:
            errors.append(f"{prefix}: duplicate task_id {task_id}")
        seen.add(task_id)
        base_id = task["gold_base_tool_id"]
        if base_id not in base_tool_ids:
            errors.append(f"{task_id}: unknown gold_base_tool_id {base_id}")
        gold_listing_ids = task["gold_listing_ids"]
        if not isinstance(gold_listing_ids, list) or not gold_listing_ids:
            errors.append(f"{task_id}: gold_listing_ids must be a non-empty list")
        else:
            for listing_id in gold_listing_ids:
                if listing_id not in listing_ids:
                    errors.append(f"{task_id}: unknown gold_listing_id {listing_id}")
        if task["reward_source"] != "llm_reflection_judge":
            errors.append(f"{task_id}: reward_source must be llm_reflection_judge")
        if not str(task["user_task"]).strip():
            errors.append(f"{task_id}: empty user_task")
        if not str(task["marketplace_query"]).strip():
            errors.append(f"{task_id}: empty marketplace_query")
        evidence = str(task["evidence_quote"]).strip()
        if not evidence:
            errors.append(f"{task_id}: empty evidence_quote")
        else:
            live_output = live_outputs_by_tool.get(base_id, "")
            answer = str(task["ground_truth_answer"]).strip()
            if not evidence_supported(evidence, live_output) and not evidence_supported(answer, live_output):
                errors.append(f"{task_id}: evidence_quote not found in live output for {base_id}")
    return errors


def searchable_live_output(preview: str) -> str:
    """Return raw preview plus decoded MCP text content for evidence matching."""
    parts = [preview]
    try:
        payload = json.loads(preview)
        for item in payload.get("content", []) or []:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        structured = payload.get("structuredContent")
        if structured is not None:
            parts.append(json.dumps(structured, ensure_ascii=False, default=str))
    except Exception:
        pass
    return "\n".join(parts)


def evidence_supported(evidence: str, live_output: str) -> bool:
    if evidence in live_output:
        return True
    return normalize_for_match(evidence) in normalize_for_match(live_output)


def normalize_for_match(value: str) -> str:
    return " ".join(value.replace("\\n", " ").split())
