"""
Generate 4 synthetic variants for every real tool in the pool.

Failure modes:
  - truncated: returns partial/truncated results (description overpromises)
  - hollow:    returns right structure but critical fields empty (description vague)
  - flaky:     fails ~50% of the time (description normal; only discoverable by trying)
  - stale:     returns cached/fixed response regardless of input (description mentions caching)

Each synthetic endpoint wraps a specific real endpoint. Descriptions are
derived from the real tool's description with a failure-mode-specific rewrite,
so (a) each synthetic lives near its real base in embedding space, and
(b) the reranker cannot separate synthetics from reals by memorizing
pool-level templates.

Writes:
  data/tools.json              (real servers unchanged, synthetics appended)
  data/synthetic_configs.json  (failure-injection configs keyed by server id)
"""
import copy
import json
import random
from pathlib import Path

random.seed(42)

REPO = Path(__file__).resolve().parents[1]
TOOLS_PATH = REPO / "data/tools.json"
CONFIGS_PATH = REPO / "data/synthetic_configs.json"

FAILURE_TYPES = ["truncated", "hollow", "flaky", "stale"]

FAILURE_CONFIGS = {
    "truncated": {"type": "truncate", "max_chars": 200},
    "hollow":    {"type": "truncate", "max_chars": 50},
    "flaky":     {"type": "random_failure", "fail_rate": 0.5},
    "stale":     {"type": "stale", "stale_rate": 0.7},
}

STALE_RESPONSES = {
    "fetch":      "Error: Connection reset by peer. The server closed the connection unexpectedly.",
    "pdf":        "Error: Unable to parse document. The file may be corrupted or in an unsupported format.",
    "search":     '{"results": [], "message": "No results found. Try broadening your search terms."}',
    "filesystem": "Error: Permission denied. The process does not have sufficient privileges to access this resource.",
    "excel":      "Error: Unable to open workbook. The file format may be unsupported or the file is corrupted.",
    "wikipedia":  '{"error": "Article not found. The requested page does not exist on Wikipedia."}',
    "arxiv":      '{"error": "Paper not found. The requested arXiv ID does not exist or has been withdrawn."}',
}


def synth_description(base_desc: str, failure_type: str, seed_rng: random.Random) -> str:
    """Generate a failure-mode-appropriate description from a base tool description."""
    d = (base_desc or "").strip().rstrip(".")
    if not d:
        d = "tool for data access"

    if failure_type == "truncated":
        boosters = [
            "Delivers complete extraction with full content coverage.",
            "Returns comprehensive output across all fields and sections.",
            "Enterprise-grade extraction with full-depth retrieval.",
            "Exhaustive content retrieval with no truncation or field omission.",
        ]
        return f"{d}. {seed_rng.choice(boosters)}"

    if failure_type == "hollow":
        # Vague, short, generic. Keep tool name-ish context, drop specifics.
        first_phrase = d.split(".")[0].strip()
        short = first_phrase[:60].rstrip(",").rstrip()
        templates = [
            f"Utility for {short.lower()}.",
            f"General-purpose handler for {short.lower()} operations.",
            f"Access point for {short.lower()}.",
            f"Interface for {short.lower()} requests.",
        ]
        return seed_rng.choice(templates)

    if failure_type == "flaky":
        # Normal-sounding paraphrase, failure invisible from description alone.
        suffixes = [
            "Lightweight and easy to integrate.",
            "Straightforward interface for common use cases.",
            "Standard-compliant and ready to use.",
            "Minimal configuration required.",
        ]
        return f"{d}. {seed_rng.choice(suffixes)}"

    if failure_type == "stale":
        caching_templates = [
            f"High-speed {d.lower()} with intelligent response caching.",
            f"Cached {d.lower()} for consistent, repeatable results.",
            f"Pre-indexed {d.lower()} for fast retrieval.",
            f"{d}. Built-in result caching for efficient repeated access.",
        ]
        return seed_rng.choice(caching_templates)

    raise ValueError(f"unknown failure_type: {failure_type}")


def main():
    servers = json.loads(TOOLS_PATH.read_text())
    # Keep only real servers; drop any previously-generated synthetics so we
    # can regenerate from scratch deterministically.
    real_servers = [s for s in servers if "synth" not in s["id"]]
    print(f"Real servers: {len(real_servers)}")
    real_endpoints = sum(len(s.get("tools", [])) for s in real_servers)
    print(f"Real endpoints: {real_endpoints}")

    rng = random.Random(42)
    synthetic_servers = []
    synthetic_configs = {}
    counter = 0

    for server in real_servers:
        category = server.get("category", "other")
        for tool in server.get("tools", []):
            base_desc = tool.get("description", "")
            for failure_type in FAILURE_TYPES:
                counter += 1
                synth_id = f"{category}-synth-{failure_type}-{counter:04d}"

                synth_tool = copy.deepcopy(tool)
                synth_tool["description"] = synth_description(base_desc, failure_type, rng)

                synth_server = {
                    "id": synth_id,
                    "name": f"{category}-{failure_type}-{counter}",
                    "category": category,
                    "github": f"https://github.com/synthetic/{synth_id}",
                    "install": copy.deepcopy(server["install"]),
                    "tools": [synth_tool],
                    "_synthetic": {
                        "real_server_id": server["id"],
                        "real_tool_name": tool["name"],
                        "failure_type": failure_type,
                    },
                }
                synthetic_servers.append(synth_server)

                cfg = {
                    "real_server_id": server["id"],
                    "real_tool_name": tool["name"],
                    **FAILURE_CONFIGS[failure_type],
                }
                if failure_type == "stale":
                    cfg["stale_response"] = STALE_RESPONSES.get(
                        category,
                        '{"error": "Stale cached response. No fresh data available."}',
                    )
                synthetic_configs[synth_id] = cfg

    all_servers = real_servers + synthetic_servers
    TOOLS_PATH.write_text(json.dumps(all_servers, indent=2))
    CONFIGS_PATH.write_text(json.dumps(synthetic_configs, indent=2))

    total_endpoints = sum(len(s.get("tools", [])) for s in all_servers)
    print(f"Synthetic servers: {len(synthetic_servers)}")
    print(f"Total servers: {len(all_servers)}")
    print(f"Total endpoints: {total_endpoints}")

    from collections import Counter
    cats = Counter(s["category"] for s in all_servers)
    print("\nPer-category:")
    for cat, n in sorted(cats.items()):
        real_n = sum(1 for s in real_servers if s["category"] == cat)
        synth_n = n - real_n
        print(f"  {cat:<12}  real={real_n:>2}  synth={synth_n:>3}  total={n:>3}")


if __name__ == "__main__":
    main()
