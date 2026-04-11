"""
The Agentic Loop

Agent receives a task from the user. It has a system prompt telling it
decompose this task, and query a MCP marketplace for a tool to use for this task.

It outputs a query that we send into the retriever, and we return top k tools with their schemas.

Agent calls the tools, gets the information.

Finally, it rates the tool itself.

Each task thus requires 3 agent calls.
"""
import json
import os
from openai import OpenAI

# Models that use native OpenAI API; everything else routes through OpenRouter
OPENAI_NATIVE_MODELS = {"gpt-5.4-nano", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"}


class Agent:
    """Orchestrates the 3-call agentic loop for a single task."""

    def __init__(self, model: str = "gpt-5.4-nano"):
        if model in OPENAI_NATIVE_MODELS:
            self.client = OpenAI()
        else:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        self.model = model

    def _sanitize_schema(self, schema: dict) -> dict:
        """Recursively sanitize a JSON schema for OpenAI function calling compatibility."""
        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
            # Remove unsupported keywords
            if key in ("$schema", "additionalProperties", "propertyNames",
                       "minItems", "maxItems", "minimum", "maximum"):
                continue
            # anyOf/oneOf → collapse to first option
            if key in ("anyOf", "oneOf") and isinstance(value, list) and value:
                first = value[0] if isinstance(value[0], dict) else {}
                for k, v in first.items():
                    if k not in result:
                        result[k] = self._sanitize_schema(v) if isinstance(v, dict) else v
                continue
            # Tuple-style items (list) → single schema
            if key == "items" and isinstance(value, list):
                result["items"] = {"type": "number"}
                continue
            # Recurse into dicts
            if isinstance(value, dict):
                result[key] = self._sanitize_schema(value)
            # Recurse into lists of dicts (e.g., items in arrays)
            elif isinstance(value, list):
                result[key] = [self._sanitize_schema(v) if isinstance(v, dict) else v for v in value]
            else:
                result[key] = value

        # Fix arrays missing items
        if result.get("type") == "array" and "items" not in result:
            result["items"] = {"type": "string"}

        return result

    def call_1_decompose(self, task: str) -> str:
        """
        Agent receives a task from the user.
        It has a system prompt telling it decompose this task,
        and query a MCP marketplace for a tool to use for this task.
        It outputs a query that we send into the retriever.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI agent. You receive a task from a user. "
                        "You have access to an MCP tool marketplace. "
                        "Determine what type of tool you need to accomplish this task. "
                        "Describe the tool capability, not the task itself. "
                        "Focus on what the tool DOES (e.g., searches the web, reads Excel files, fetches URLs). "
                        "Output ONLY a short tool capability description, nothing else."
                    ),
                },
                {"role": "user", "content": task},
            ],
            temperature=0,
            max_completion_tokens=150,
        )
        return response.choices[0].message.content.strip()

    def call_2_select_and_call(self, task: str, query: str,
                                candidates: list[dict],
                                example_calls: dict = None) -> dict:
        """
        Present candidates as native function tools and let the model call one.
        The model's built-in tool-selection picks the best function and fills arguments.
        """
        # Normalize rerank scores to match percentages (30-99% range)
        # Wide spread so rank 1 is clearly differentiated from rank 3
        raw_scores = [c.get("rerank_score") for c in candidates]
        valid_scores = [s for s in raw_scores if s is not None]
        if valid_scores:
            min_s, max_s = min(valid_scores), max(valid_scores)
            spread = max_s - min_s if max_s > min_s else 1.0
            mid_pct = 30 + 69 * 0.5
            match_pcts = [
                30 + 69 * (s - min_s) / spread if s is not None else mid_pct
                for s in raw_scores
            ]
        else:
            match_pcts = [None] * len(candidates)

        # Build OpenAI function tools from candidates
        tools = []
        tool_map = {}  # function name -> (tool_index 1-based, candidate)
        for i, c in enumerate(candidates):
            # Function names must be unique and alphanumeric
            func_name = f"{c['server_id']}__{c['tool_name']}".replace("-", "_")
            match_str = f" ({match_pcts[i]:.0f}% match)" if match_pcts[i] is not None else ""

            desc = f"{c['description']}{match_str}"
            # Include few-shot example in description if available
            key = f"{c['server_id']}/{c['tool_name']}"
            if example_calls and key in example_calls:
                desc += f"\nExample call: {json.dumps(example_calls[key])}"

            # Build parameters from inputSchema, sanitizing for OpenAI compatibility
            schema = c.get("inputSchema", {})
            parameters = self._sanitize_schema(schema)
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
            if "required" in schema:
                parameters["required"] = schema["required"]

            tools.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": desc[:1024],
                    "parameters": parameters,
                },
            })
            tool_map[func_name] = (i + 1, c)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI agent. You MUST call exactly one tool to accomplish the task. "
                        "Do not respond with text — call a tool. "
                        "The match % is based on real-world execution data — tools with "
                        "higher match % have been tested and verified to work for tasks "
                        "like yours. STRONGLY prefer the highest match % tool unless its "
                        "description is clearly wrong for your task."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Task: {task}",
                },
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0,
            max_completion_tokens=300,
        )

        # Extract the tool call
        msg = response.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            func_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            if func_name in tool_map:
                tool_index, _ = tool_map[func_name]
                return {"tool_index": tool_index, "arguments": arguments}

        # Fallback: pick tool 1
        return {"tool_index": 1, "arguments": {}}

    def call_3_rate(self, task: str, tool_info: dict,
                     tool_result: dict) -> dict:
        """
        Finally, it rates the tool itself.
        Returns structured feedback: relevance, success, score, reasoning.
        """
        result_text = json.dumps(tool_result, indent=2, default=str)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI agent evaluating a tool you just used. "
                        "Be STRICT: success means the tool result DIRECTLY ANSWERS the original task. "
                        "If the tool ran without errors but returned data that doesn't answer the question, that is NOT success. "
                        "If the tool returned metadata, sheet names, or partial info instead of the actual answer, that is NOT success. "
                        "Respond with ONLY a JSON object:\n"
                        "{\n"
                        '  "success": true/false,    // did the result DIRECTLY ANSWER the task? (not just run without error)\n'
                        '  "reasoning": "..."        // brief explanation\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task}\n"
                        f"Tool used: {tool_info['server_id']}/{tool_info['tool_name']}\n"
                        f"Tool description: {tool_info['description']}\n"
                        f"\nTool result:\n{result_text}"
                    ),
                },
            ],
            temperature=0,
            max_completion_tokens=200,
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            rating = json.loads(raw)
        except json.JSONDecodeError:
            rating = {
                "relevance": False,
                "success": False,
                "score": 1,
                "reasoning": f"Failed to parse rating: {raw[:200]}",
            }

        return rating
