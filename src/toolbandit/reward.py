"""Stage 5: reward. Was the chosen tool's output good enough? (0 or 1)

This single reward number is what teaches the bandit. There are two ways to get
it, with the SAME meaning, so the bandit code doesn't care which is used:

  metadata_reward  cheap and offline. We secretly know each tool's quality, so we
                   score 1 if the call succeeded AND the tool belongs to the task's
                   correct base-tool family. Used for fast training with no API calls.

  JudgeReward      realistic. A separate LLM reads the user task and the tool output
                   and decides if the output actually helped. This is the reward the
                   paper argues for, because in the real world you usually can't look
                   up whether a tool "really" worked.

Both return (reward, info) where info is a small dict logged for inspection.
"""

from __future__ import annotations

from typing import Any, Protocol

from .data import family_of
from .marketplace import Outcome


class RewardFn(Protocol):
    def __call__(self, task: dict[str, Any], listing: dict[str, Any], outcome: Outcome) -> tuple[float, dict[str, Any]]:
        ...


def metadata_reward(task: dict[str, Any], listing: dict[str, Any], outcome: Outcome) -> tuple[float, dict[str, Any]]:
    if not outcome.ok:
        return 0.0, {"judge": "metadata", "success": False, "why": f"execution failed: {outcome.failure_type}"}
    if family_of(listing) != task["gold_base_tool_id"]:
        return 0.0, {"judge": "metadata", "success": False, "why": "wrong base-tool family"}
    return 1.0, {"judge": "metadata", "success": True, "why": "correct family, execution succeeded"}


class JudgeReward:
    """Ask an LLM whether the tool output was sufficient for the user task."""

    def __init__(self, judge_model: str) -> None:
        # "self" means each caller model judges its own pick; otherwise a fixed judge.
        self.judge_model = judge_model
        self._clients: dict[str, Any] = {}

    def __call__(self, task: dict[str, Any], listing: dict[str, Any], outcome: Outcome) -> tuple[float, dict[str, Any]]:
        # Imported lazily so metadata-only runs never need the OpenAI SDK / keys.
        from .caller import call_model, client_for_model, parse_json, sanitize_text

        model = task["_caller_model"] if self.judge_model == "self" else self.judge_model
        if model not in self._clients:
            self._clients[model] = client_for_model(model)

        messages = [
            {"role": "system", "content":
                "You judge whether a tool call returned enough correct information for the user's task. "
                'Return only JSON: {"success": true/false, "reasoning": "short"}. '
                "Errors, empty output, wrong topic, or malformed failures are not success."},
            {"role": "user", "content": _json(
                {
                    "user_task": task["user_task"],
                    "expected_evidence_hint": task.get("evidence_quote"),
                    "tool": {"server": listing.get("server"),
                             "description": sanitize_text(str(listing.get("description", "")))[:900]},
                    "execution_ok": outcome.ok,
                    "tool_output": outcome.output,
                })},
        ]
        try:
            raw = call_model(self._clients[model], model, messages, max_tokens=300)
            payload = parse_json(raw)
            success = bool(payload.get("success"))
            return (1.0 if success else 0.0), {"judge": model, "success": success, "why": payload.get("reasoning", "")}
        except Exception as exc:  # a judge failure counts as no reward
            return 0.0, {"judge": model, "success": False, "why": f"judge_error: {exc}"}


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
