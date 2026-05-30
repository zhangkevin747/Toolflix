from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .models import GeneratedTask


SYSTEM_PROMPT = """You generate benchmark tasks for ToolBandit.

Design simple, one-step user tasks. The agent should need exactly one tool call.
The user task must be fuzzy and natural, not a tool-name request.
The marketplace query must be a short capability query for retrieving tools.
The ground truth answer must be directly supported by the provided tool output.
The training reward is not this answer; training uses an LLM reflection judge over whether the tool output helped.
Return strict JSON only."""


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class OpenAITaskGenerator:
    def __init__(self, model: str, temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

    def generate_for_tool(
        self,
        *,
        tool: dict[str, Any],
        fixture: dict[str, Any],
        live_output_preview: str,
        gold_listing_ids: list[str],
        task_index: int,
        tasks_per_tool: int,
    ) -> list[GeneratedTask]:
        prompt = build_prompt(
            tool=tool,
            fixture=fixture,
            live_output_preview=live_output_preview,
            gold_listing_ids=gold_listing_ids,
            task_index=task_index,
            tasks_per_tool=tasks_per_tool,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        text = response.choices[0].message.content or "{}"
        payload = json.loads(text)
        rows = payload.get("tasks", [])
        if not isinstance(rows, list):
            raise ValueError("generator response must contain a tasks array")

        tasks: list[GeneratedTask] = []
        for offset, row in enumerate(rows):
            task_id = f"task_{task_index + offset:05d}"
            evidence = clean_text(str(row.get("evidence_quote", "")))
            answer = clean_text(str(row.get("ground_truth_answer", "")))
            user_task = clean_text(str(row.get("user_task", "")))
            marketplace_query = clean_text(str(row.get("marketplace_query", "")))
            if not all([evidence, answer, user_task, marketplace_query]):
                continue
            tasks.append(GeneratedTask(
                task_id=task_id,
                user_task=user_task,
                marketplace_query=marketplace_query,
                gold_base_tool_id=tool["tool_id"],
                gold_listing_ids=gold_listing_ids,
                fixture_args=fixture["fixture_args"],
                ground_truth_answer=answer,
                evidence_quote=evidence,
                generator_model=self.model,
                tags=list(row.get("tags", [])) if isinstance(row.get("tags"), list) else [],
                metadata={
                    "server": tool["server"],
                    "tool_name": tool["tool_name"],
                    "category": tool["category"],
                    "fixture_confidence": fixture.get("fixture_confidence"),
                    "generation_style": "mcp_bench_inspired_simple_fuzzy_one_tool",
                },
            ))
        return tasks


def build_prompt(
    *,
    tool: dict[str, Any],
    fixture: dict[str, Any],
    live_output_preview: str,
    gold_listing_ids: list[str],
    task_index: int,
    tasks_per_tool: int,
) -> str:
    return f"""Create {tasks_per_tool} benchmark task(s) for this executable tool.

The task style is inspired by MCP-Bench fuzzy instructions, but our tasks are deliberately simpler:
- exactly one tool call should be sufficient
- no tool names or server names in the user task
- stable/verifiable answer
- the answer must be grounded in the tool output below
- the marketplace query is separate from the user task

Tool metadata:
{json.dumps({
    "tool_id": tool["tool_id"],
    "server": tool["server"],
    "tool_name": tool["tool_name"],
    "category": tool["category"],
    "description": tool["description"],
    "input_schema": tool["input_schema"],
    "known_good_fixture_args": fixture["fixture_args"],
    "gold_listing_ids": gold_listing_ids,
}, indent=2, ensure_ascii=False)}

Actual live tool output preview:
{live_output_preview[:5000]}

Rules:
1. The user task should sound like a real person asking for help.
2. The user task should not reveal the exact tool name, server name, API name, or argument schema.
3. The marketplace_query should be short, like "wikipedia article search" or "medical sodium correction calculator".
4. The ground_truth_answer should be short, preferably one string, number, date, title, or compact phrase.
5. The evidence_quote must be an exact short substring copied from the tool output preview.
6. If the output does not support a stable answer, return an empty tasks array.
7. Do not invent facts not present in the output.

Return JSON:
{{
  "tasks": [
    {{
      "user_task": "...",
      "marketplace_query": "...",
      "ground_truth_answer": "...",
      "evidence_quote": "exact substring from output",
      "tags": ["simple", "one_step"]
    }}
  ]
}}

The first task index is {task_index}."""


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def retry_generate(fn, retries: int = 3, delay: float = 2.0):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - CLI retry path
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(delay * (attempt + 1))
    assert last_error is not None
    raise last_error
