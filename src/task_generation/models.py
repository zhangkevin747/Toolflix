from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GeneratedTask:
    task_id: str
    user_task: str
    marketplace_query: str
    gold_base_tool_id: str
    gold_listing_ids: list[str]
    fixture_args: dict[str, Any]
    ground_truth_answer: str
    evidence_quote: str
    expected_answer_source: str = "tool_output"
    reward_source: str = "llm_reflection_judge"
    generator_model: str = ""
    difficulty: str = "simple_one_step"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
