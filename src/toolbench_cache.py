"""
Read-only access to StableToolBench's cached API responses, with the GPT-4
simulator fallback disabled.

Rationale: the StableToolBench simulator fabricates plausible-looking JSON
for broken/deprecated APIs, which homogenizes tool quality and erases the
signal our reranker learns from. The cache itself contains genuine
responses from the original ToolBench API calls, including real failures
(404s, 5xx, rate-limits, deprecation messages). By reading the cache
directly and treating cache-miss as explicit tool failure, we preserve
real-world quality variance while keeping evaluation reproducible.

Cache layout (on disk):
    tool_response_cache/
      <standard_category>/
        <tool_name>_for_<standard_category>/
          <api_name>.json        # dict of {str(tool_input): response}
"""
import json
import re
from pathlib import Path
from typing import Any, Optional


def _standardize(s: str) -> str:
    """Replicate StableToolBench server/utils.py standardize()."""
    s = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).lower().strip("_")
    if s and s[0].isdigit():
        s = "get_" + s
    return s


def _standardize_category(c: str) -> str:
    c = c.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in c or "," in c:
        c = c.replace(" ", "_").replace(",", "_")
    return c.replace("__", "_")


def _change_name(name: str) -> str:
    reserved = {"from", "class", "return", "false", "true", "id", "and"}
    return f"is_{name}" if name in reserved else name


class ToolBenchCache:
    """Read-only access to the StableToolBench response cache.

    Miss behavior: returns an explicit failure dict rather than calling a
    simulator or a live API. The missing signal is exactly what we want
    the reranker to learn to avoid.
    """

    CACHE_MISS = {
        "error": "cache_miss",
        "response": "",
    }

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"cache_dir not found: {cache_dir}")
        self._file_cache: dict[Path, dict] = {}

    def call(self, category: str, tool_name: str, api_name: str,
             tool_input: dict | str) -> dict:
        """Return the cached response for (category, tool, api, input).

        Response shape matches StableToolBench: {"error": str, "response": str}.
        Cache miss -> {"error": "cache_miss", "response": ""}.
        """
        std_cat = _standardize_category(category)
        std_api = _change_name(_standardize(api_name)).split(f"_for_{tool_name}")[0]
        if not tool_name.endswith(f"_for_{std_cat}"):
            std_tool = _standardize(tool_name) + f"_for_{std_cat}"
        else:
            std_tool = tool_name

        path = self.cache_dir / std_cat / std_tool / f"{std_api}.json"
        file_data = self._load(path)
        if file_data is None:
            return dict(self.CACHE_MISS)

        if isinstance(tool_input, dict):
            d = tool_input
        else:
            try:
                d = json.loads(tool_input)
            except Exception:
                d = None

        # Keys in cache files are inconsistent: some use str(dict) Python
        # repr form, some use json.dumps (compact), some use indent=2.
        # Try all plausible variants.
        candidates = [str(tool_input)]
        if d is not None:
            candidates += [
                str(d),
                json.dumps(d),
                json.dumps(d, indent=2),
                json.dumps(d, sort_keys=True),
            ]

        for key in candidates:
            if key in file_data:
                return file_data[key]
        return dict(self.CACHE_MISS)

    def _load(self, path: Path) -> Optional[dict]:
        if path in self._file_cache:
            return self._file_cache[path]
        if not path.is_file():
            self._file_cache[path] = None
            return None
        try:
            with open(path) as f:
                d = json.load(f)
            self._file_cache[path] = d
            return d
        except Exception:
            self._file_cache[path] = None
            return None

    def has_entry(self, category: str, tool_name: str, api_name: str) -> bool:
        """True iff we have any cached input/response pair for this api."""
        std_cat = _standardize_category(category)
        std_api = _change_name(_standardize(api_name)).split(f"_for_{tool_name}")[0]
        if not tool_name.endswith(f"_for_{std_cat}"):
            std_tool = _standardize(tool_name) + f"_for_{std_cat}"
        else:
            std_tool = tool_name
        return (self.cache_dir / std_cat / std_tool / f"{std_api}.json").is_file()

    def stats(self) -> dict[str, int]:
        """Crude cache-wide stats: category/tool/api/response counts."""
        cats = [p for p in self.cache_dir.iterdir() if p.is_dir()]
        n_tools = 0
        n_files = 0
        n_responses = 0
        for cat in cats:
            for tool in cat.iterdir():
                if not tool.is_dir():
                    continue
                n_tools += 1
                for f in tool.iterdir():
                    if f.suffix == ".json":
                        n_files += 1
                        try:
                            n_responses += len(json.load(open(f)))
                        except Exception:
                            pass
        return {
            "categories": len(cats),
            "tools": n_tools,
            "api_files": n_files,
            "responses": n_responses,
        }


def classify_response(resp: dict) -> str:
    """Classify a cached response into one of: ok, error_field, http_error,
    rate_limit, auth_error, deprecated, empty, unknown.
    """
    if not isinstance(resp, dict):
        return "unknown"
    body = str(resp.get("response", "")).lower()
    err = str(resp.get("error", "")).lower()
    if err == "cache_miss":
        return "cache_miss"
    if not body and not err:
        return "empty"
    if err:
        return "error_field"
    combined = body + " " + err
    if any(k in combined for k in ["rate limit", "time out", "timed out",
                                    "429", "504", "500", "internal error"]):
        return "rate_limit"
    if any(k in combined for k in ["unauthorized", "unauthenticat", "forbidden",
                                    "401", "403", "access_denied", "invalid consumer key"]):
        return "auth_error"
    if any(k in combined for k in ["deprecated", "no longer available", "sunset",
                                    "api doesn't exists", "does not exist"]):
        return "deprecated"
    if any(k in combined for k in ["404", "not found", "502", "bad gateway"]):
        return "http_error"
    return "ok"
