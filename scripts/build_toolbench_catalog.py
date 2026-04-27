"""
Build a Toolflix-compatible tool catalog from the subset of ToolBench tools
referenced by StableToolBench's 765 solvable queries.

Output: data/toolbench_tools.json (same shape as data/tools.json)

Rationale for the subset: we only evaluate against the solvable-queries test
set, so we don't need the full 16k-tool pool on disk. 814 tools / ~2500
endpoints is enough pool size to stress retrieval and keep memory/compute
reasonable.
"""
import json
import re
from pathlib import Path
from collections import defaultdict


REPO = Path(__file__).resolve().parents[1]
STB  = REPO / "external/StableToolBench"
QUERIES_DIR = STB / "solvable_queries/test_instruction"
TOOLS_DIR = STB / "server/tools"
OUT_PATH = REPO / "data/toolbench_tools.json"

SPLITS = [
    "G1_instruction", "G1_category", "G1_tool",
    "G2_category", "G2_instruction", "G3_instruction",
]


def standardize(s: str) -> str:
    s = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).lower().strip("_")
    if s and s[0].isdigit():
        s = "get_" + s
    return s


def standardize_category(c: str) -> str:
    c = c.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in c or "," in c:
        c = c.replace(" ", "_").replace(",", "_")
    return c.replace("__", "_")


def collect_referenced_tools() -> dict[tuple[str, str], set[str]]:
    """Return {(category, tool_name): {api_name, ...}}."""
    refs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in SPLITS:
        data = json.loads((QUERIES_DIR / f"{split}.json").read_text())
        for q in data:
            for api in q.get("api_list", []):
                k = (api["category_name"], api["tool_name"])
                refs[k].add(api["api_name"])
    return refs


def load_tool_schema(category: str, tool_name: str) -> dict | None:
    std_cat = standardize_category(category)
    std_tool = standardize(tool_name)
    path = TOOLS_DIR / std_cat / f"{std_tool}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def api_to_toolflix_tool(api: dict) -> dict:
    """Convert a ToolBench api_list entry to Toolflix's tool shape."""
    required = api.get("required_parameters", []) or []
    optional = api.get("optional_parameters", []) or []
    properties = {}
    required_names = []
    for p in required:
        name = p.get("name")
        if not name:
            continue
        properties[name] = {
            "type": _map_type(p.get("type")),
            "description": p.get("description") or "",
        }
        if "default" in p and p["default"] not in (None, ""):
            properties[name]["default"] = p["default"]
        required_names.append(name)
    for p in optional:
        name = p.get("name")
        if not name:
            continue
        properties[name] = {
            "type": _map_type(p.get("type")),
            "description": p.get("description") or "",
        }
        if "default" in p and p["default"] not in (None, ""):
            properties[name]["default"] = p["default"]
    return {
        "name": api["name"],
        "description": api.get("description") or "",
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required_names,
        },
    }


def _map_type(t: str | None) -> str:
    if not t:
        return "string"
    t = t.lower()
    if t in ("string", "str"): return "string"
    if t in ("integer", "int"): return "integer"
    if t in ("number", "float", "double"): return "number"
    if t in ("boolean", "bool"): return "boolean"
    if t in ("array", "list"): return "array"
    if t in ("object", "dict"): return "object"
    return "string"


def main():
    refs = collect_referenced_tools()
    print(f"Referenced tools: {len(refs)}")

    servers = []
    missing_schemas = 0
    missing_apis = 0
    for (category, tool_name), used_apis in sorted(refs.items()):
        schema = load_tool_schema(category, tool_name)
        if schema is None:
            missing_schemas += 1
            continue

        api_map = {a["name"]: a for a in schema.get("api_list", [])}
        endpoints = []
        for api_name in sorted(used_apis):
            api = api_map.get(api_name)
            if api is None:
                missing_apis += 1
                continue
            endpoints.append(api_to_toolflix_tool(api))
        if not endpoints:
            continue

        server_id = f"toolbench:{standardize_category(category)}:{standardize(tool_name)}"
        servers.append({
            "id": server_id,
            "name": schema.get("name") or tool_name,
            "description": schema.get("tool_description") or "",
            "category": category,
            "tools": endpoints,
            "install": {
                "type": "toolbench_cache",
                "category": category,
                "tool_name": tool_name,
            },
        })

    total_endpoints = sum(len(s["tools"]) for s in servers)
    print(f"Wrote {len(servers)} servers, {total_endpoints} endpoints")
    print(f"Missing tool schemas (dropped): {missing_schemas}")
    print(f"Missing API entries within found schemas: {missing_apis}")

    OUT_PATH.write_text(json.dumps(servers, indent=2))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
