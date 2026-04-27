"""
Merge the 5 batch outputs from the subagent rewrite pipeline, add the
no-tool negative controls, validate, and write the final gaia_bench.json.

Validation:
  1. Ground-truth not leaked in question text (verbatim).
  2. expected_category is in the allowed set.
  3. Attachments exist on disk when specified.
  4. ground_truth[0] matches original GAIA final_answer after whitespace strip.
"""
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BATCHES_IN  = REPO / "data/gaia_rewrite_batches"
BATCHES_OUT = REPO / "data/gaia_rewrite_output"
OUT = REPO / "data/gaia_bench.json"

ALLOWED_CATS = {"excel", "pdf", "filesystem", "arxiv", "wikipedia", "fetch", "search"}

NO_TOOL_ITEMS = [
    ("notool_math_01", "What is 127 multiplied by 43? Return just the product.", ["5461", "5,461"]),
    ("notool_math_02", "What is the sum of the first ten prime numbers? Return just the number.", ["129"]),
    ("notool_math_03", "What is 1024 divided by 16? Return just the result.", ["64"]),
    ("notool_math_04", "How many seconds are in two hours and thirty minutes? Return just the number.", ["9000"]),
    ("notool_logic_01", "If all Blerps are Florps and some Florps are Quarks, is it necessarily true that some Blerps are Quarks? Return just yes or no.", ["no"]),
    ("notool_logic_02", "A is taller than B. B is taller than C. Who is the shortest of the three? Return just the letter.", ["c"]),
    ("notool_logic_03", "How many sides does a regular hexagon have? Return just the number.", ["6"]),
    ("notool_common_01", "What day of the week comes immediately after Thursday? Return just the day.", ["friday"]),
    ("notool_common_02", "Which planet is closest to the Sun? Return just the planet name.", ["mercury"]),
    ("notool_common_03", "How many continents are commonly listed on Earth? Return just the number.", ["7"]),
    ("notool_common_04", "What is the chemical symbol for gold? Return just the symbol.", ["au"]),
    ("notool_common_05", "How many players are on a standard soccer team on the field at one time (per side)? Return just the number.", ["11"]),
    ("notool_word_01", "What is the last letter of the word \"elephant\"? Return just the letter.", ["t"]),
    ("notool_word_02", "How many vowels are in the word \"strawberry\"? Return just the number.", ["2"]),
    ("notool_word_03", "Reverse the word \"hello\". Return just the reversed word.", ["olleh"]),
    ("notool_time_01", "What month comes immediately after June? Return just the month name.", ["july"]),
    ("notool_time_02", "How many hours are in three days? Return just the number.", ["72"]),
]


def load_original_answers() -> dict[str, str]:
    """Map full GAIA task_id -> final_answer from the batch input files."""
    out = {}
    for p in sorted(BATCHES_IN.glob("batch_*.json")):
        for t in json.loads(p.read_text()):
            out[t["task_id"]] = t["final_answer"]
    return out


def normalize(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def main():
    originals = load_original_answers()

    accepted = []
    drops = {"skipped": 0, "cat_invalid": 0, "gt_mismatch": 0,
             "leak": 0, "missing_attach": 0, "malformed": 0}

    for p in sorted(BATCHES_OUT.glob("batch_*.json")):
        items = json.loads(p.read_text())
        for x in items:
            if x.get("skip_reason"):
                drops["skipped"] += 1; continue
            try:
                cat = x["expected_category"]
                q = x["question"]
                gts = x["ground_truth"]
            except KeyError:
                drops["malformed"] += 1; continue
            if cat not in ALLOWED_CATS:
                drops["cat_invalid"] += 1; continue
            tid = x["source"].split(":", 1)[1]
            orig = originals.get(tid)
            if orig is None or normalize(gts[0]) != normalize(orig):
                drops["gt_mismatch"] += 1; continue
            # Leakage: ground-truth verbatim in question (case-insensitive).
            if gts[0].lower() and gts[0].lower() in q.lower():
                drops["leak"] += 1; continue
            # Attachment exists on disk (if any).
            for a in x.get("attachments") or []:
                if not (REPO / a).is_file():
                    drops["missing_attach"] += 1
                    break
            else:
                accepted.append(x)

    print(f"Accepted from subagent output: {len(accepted)}")
    print("Drop breakdown:")
    for k, v in drops.items():
        print(f"  {k:<20}  {v:>3}")

    # Add no-tool controls.
    for nid, q, gts in NO_TOOL_ITEMS:
        accepted.append({
            "id": nid,
            "source": "hand",
            "level": 1,
            "question": q,
            "expected_category": None,
            "ground_truth": list(gts),
            "attachments": [],
        })
    print(f"\nAdded no-tool negative controls: {len(NO_TOOL_ITEMS)}")

    from collections import Counter
    cats = Counter(i.get("expected_category") or "no_tool" for i in accepted)
    print("\nFinal distribution:")
    for c, n in sorted(cats.items()):
        print(f"  {c:<12}  {n:>3}")
    print(f"  {'TOTAL':<12}  {len(accepted):>3}")

    OUT.write_text(json.dumps(accepted, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
