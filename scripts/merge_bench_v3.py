"""
Merge the 7 per-category subagent outputs into the final Toolflix benchmark.

Validation:
  1. Schema: id, question, expected_category, ground_truth, attachments present.
  2. expected_category in allowed set.
  3. Attachments exist on disk.
  4. ground_truth[0] NOT a case-insensitive substring of question.
  5. Each question <= 500 chars; each ground_truth[0] <= 50 chars.
"""
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "data/bench_v3_output"
OUT = REPO / "data/gaia_bench.json"

ALLOWED_CATS = {"arxiv", "excel", "fetch", "filesystem", "pdf", "search", "wikipedia"}
CATS = sorted(ALLOWED_CATS)

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


def main():
    accepted = []
    drops = {"schema": 0, "cat_invalid": 0, "leak": 0, "missing_attach": 0,
             "too_long": 0, "empty_gt": 0}

    for cat in CATS:
        p = SRC_DIR / f"{cat}.json"
        items = json.loads(p.read_text())
        cat_accepted = 0
        for x in items:
            try:
                iid = x["id"]
                q = x["question"]
                gts = x["ground_truth"]
                ec = x["expected_category"]
                atts = x.get("attachments", [])
            except KeyError:
                drops["schema"] += 1; continue
            if ec not in ALLOWED_CATS:
                drops["cat_invalid"] += 1; continue
            if not gts or not any(str(g).strip() for g in gts):
                drops["empty_gt"] += 1; continue
            if len(q) > 500 or len(str(gts[0])) > 50:
                drops["too_long"] += 1; continue
            if str(gts[0]).lower() and str(gts[0]).lower() in q.lower():
                drops["leak"] += 1
                print(f"  LEAK: {iid}  gt={gts[0]!r}")
                continue
            ok_attach = True
            for a in atts:
                if not (REPO / a).is_file():
                    # Directory-path attachments OK; missing files not OK
                    if not (REPO / a).is_dir():
                        ok_attach = False
                        drops["missing_attach"] += 1
                        print(f"  MISSING ATTACH: {iid}  {a}")
                        break
            if not ok_attach:
                continue
            x.setdefault("source", "constructed")
            x.setdefault("level", 1)
            accepted.append(x)
            cat_accepted += 1
        print(f"  {cat:<12}  {cat_accepted:>3}/{len(items)}")

    print(f"\nAccepted from subagent output: {len(accepted)}")
    print("Drops:")
    for k, v in drops.items():
        print(f"  {k:<16}  {v:>3}")

    for nid, q, gts in NO_TOOL_ITEMS:
        accepted.append({
            "id": nid, "source": "hand", "level": 1,
            "question": q,
            "expected_category": None,
            "ground_truth": list(gts),
            "attachments": [],
        })
    print(f"\nAdded no_tool controls: {len(NO_TOOL_ITEMS)}")

    cats = Counter(i.get("expected_category") or "no_tool" for i in accepted)
    print("\nFinal distribution:")
    for c, n in sorted(cats.items()):
        print(f"  {c:<12}  {n:>3}")
    print(f"  {'TOTAL':<12}  {len(accepted):>3}")

    OUT.write_text(json.dumps(accepted, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
