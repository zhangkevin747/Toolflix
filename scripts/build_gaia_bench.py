"""
Build a ground-truth GAIA benchmark from the real GAIA validation split.

Schema per item:
    {
      "id": "gaia_<task_id>",
      "source": "gaia:<task_id>",
      "level": 1 | 2,
      "question": "<natural-language task>",
      "expected_category": "excel" | "pdf" | ... | null,
      "ground_truth": ["exact answer", "alternate phrasing"],
      "attachments": ["data/gaia_files/<file>"] | [],
    }

Three task types:
  1. File-based GAIA tasks whose attachment we have locally and whose file
     type is in our supported set (xlsx, pdf, docx, txt, pdb).
  2. Web-based Level-1 GAIA tasks targeting one of our 7 categories
     (search / wikipedia / fetch / arxiv), with any multi-hop intermediates
     inlined.
  3. No-tool negative controls: GAIA items whose Annotator Metadata says
     "Tools: None" plus a few hand-added arithmetic/logic items.

Grading (enforced in src/eval_gt.py):
  - tool_match: first tool call's category matches expected_category
                (or no tool called when expected is null).
  - answer_match: normalized exact match on at least one ground_truth.
  - both_correct: tool_match AND answer_match.
"""
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FILES_DIR = REPO / "data/gaia_files"
OUT_PATH = REPO / "data/gaia_bench.json"

# ----------------------------------------------------------------------
# Category assignment by file extension
# ----------------------------------------------------------------------

EXT_TO_CATEGORY = {
    "xlsx": "excel",
    "xls":  "excel",
    "pdf":  "pdf",
    "docx": "filesystem",   # our docx-reader lives under filesystem family
    "doc":  "filesystem",
    "txt":  "filesystem",
    "pdb":  "filesystem",   # read as plain text
}

# File extensions we cannot route to any tool in the pool — skip.
EXT_UNSUPPORTED = {"mp3", "wav", "png", "jpg", "jpeg", "zip",
                   "pptx", "ppt", "py", "m4a"}


# ----------------------------------------------------------------------
# Hand-curated web-based Level-1 tasks (GAIA task_ids) with single-hop rewrites.
# Keep only those where a single tool call gets the data; inline intermediates.
# ----------------------------------------------------------------------

WEB_TASK_REWRITES = [
    # (gaia_task_id, expected_category, rewritten_question, ground_truth alternates)
    ("8e867cd7-cff9-4e6c-867a-ff5ddc2550be", "wikipedia",
     "According to Wikipedia, how many studio albums did Mercedes Sosa publish between 2000 and 2009 (both inclusive)? Return just the count.",
     ["3"]),
    ("46719c30-f4c3-4cad-be07-d5cb21eee6bb", "search",
     "Search for the 1997 paper by Hopkins et al. titled \"Pie Menus or Linear Menus, Which Is Better?\" and return the title of one of the authors' other papers with \"Menu\" in its title.",
     ["Mapping Human Oriented Information to Software Agents for Online Systems Usage"]),
    ("72e110e7-464c-453c-a309-90a95aed6538", "search",
     "Under DDC 633 on Bielefeld University Library's BASE, as of 2020, from which Latin American country did the single article in an unknown language originate? Return the country name.",
     ["Guatemala"]),
    ("4fc2f1ae-8625-45b5-ab34-ad4433bc21f8", "wikipedia",
     "On English Wikipedia, what is the username of the nominator of the only Featured Article about a dinosaur? Return just the username.",
     ["FunkMonk"]),
    ("a0c07678-e491-4bbc-8f0b-07405144218f", "wikipedia",
     "On Hokkaido Nippon-Ham Fighters' 2023 roster page on Wikipedia, who are the pitchers with the uniform numbers immediately before and after Taishō Tamai's? Return the two surnames separated by a comma.",
     ["Yoshida, Uehara", "Uehara, Yoshida"]),
    ("cabe07ed-9eca-40ea-8ead-410ef5e83f91", "search",
     "What is the surname of the equine veterinarian mentioned in the OpenStax Microbiology textbook, Chapter 12 section 1.E, exercise 29? Return just the surname.",
     ["Louvrier"]),
    ("bda648d7-d618-4883-88f4-3466eabd860e", "search",
     "In Nedoshivina's 2010 paper, where are the Vietnamese specimens described by Kuznetzov deposited? Return the city.",
     ["Saint Petersburg"]),
    ("7d4a7d1d-cac6-44a8-96e8-ea9584a70825", "search",
     "According to Girls Who Code, how many years did it take for the percentage of women in computer science to change from its peak to its 2022 level? Return just the number.",
     ["22"]),
    # --- added: wikipedia, arxiv, fetch balance ---
    ("f3917a3d-e0a1-40b7-b95d-e946c4a2bfc6", "wikipedia",
     "How many edits were made to the Wikipedia page on Antidisestablishmentarianism from its inception until June of 2023? Return just the number.",
     ["2732"]),
    ("f0f46385-fc03-4599-b5d3-f56496c3e69f", "wikipedia",
     "In terms of geographical distance between capital cities, which two countries are the furthest from each other within the ASEAN bloc according to Wikipedia? Return the two country names separated by a comma.",
     ["Indonesia, Myanmar", "Myanmar, Indonesia"]),
    ("33d8ea3b-6c6b-4ff1-803d-7e270dea8a57", "wikipedia",
     "Using the English Wikipedia link graph, what is the minimum number of page links a person must click to go from The Lord of the Rings (the book) to A Song of Ice and Fire? Return just the number.",
     ["2"]),
    ("71345b0a-9c7d-4b50-b2bf-937ec5b34cb8", "wikipedia",
     "On a leap day before 2008, a joke was removed from the Wikipedia page for \"Dragon.\" What was the phrase that was removed? Return the phrase exactly.",
     ["Here be dragons"]),
    ("a7feb290-76bb-4cb7-8800-7edaf7954f2f", "arxiv",
     "How many High Energy Physics - Lattice articles listed on arXiv in January 2020 had PostScript (.ps) versions available? Return just the number.",
     ["31"]),
    ("2a649bb1-795f-4a01-b3be-9a01868dae73", "arxiv",
     "According to the arXiv paper on SPFMV and SPCSV in Uganda from 2016, what are the EC numbers of the two most commonly used chemicals for the virus testing method? Return them semicolon-separated.",
     ["3.1.3.1; 1.11.1.7"]),
    ("5d0080cb-90d7-4712-bc33-848150e917d3", "arxiv",
     "In the University of Leicester paper \"Can Hiccup Supply Enough Fish to Maintain a Dragon's Diet?\", what is the volume of the fish bag in cubic meters? Return just the number.",
     ["0.1777"]),
    ("04a04a9b-8e09-4d3c-91e2-ee6d3d13d1bf", "arxiv",
     "Assume all articles published by Nature in 2020 used statistical significance with alpha = 0.05 and averaged three independent tests per conclusion. Approximately how many such published conclusions should be false positives? Return just the number.",
     ["41"]),
    ("ad37a656-61ad-4a2b-b33b-1a1fc6ebc3ed", "fetch",
     "Fetch the Phys.org article published on July 15, 2008 about an explosion of the US military on Bikini Atoll, and return the name of the castle-series test specified in the article's discussion of Bravo.",
     ["Bravo"]),
    ("f2feb6a4-363c-4c09-a307-dd94e5bbbb9c", "fetch",
     "Fetch the Honolulu Board of Realtors listings for Pearl City, Hawaii and report the median sale price (USD) from the most recent month. Return just the number.",
     ["900000"]),
    ("8e867cd7-cff9-4e6c-867a-ff5ddc2550be", "fetch",
     "Fetch https://en.wikipedia.org/wiki/Mercedes_Sosa and from the discography section, return the count of studio albums released between 2000 and 2009 inclusive.",
     ["3"]),
    ("3f57289b-8c60-48be-bd80-01f8099ca449", "search",
     "How many at-bats did the 1977 New York Yankees player with the most walks in the regular season have that season? Return just the number.",
     ["519"]),
    ("b415aba4-4b68-4fc6-9b89-2c812e55a3e1", "search",
     "In Nature Scientific Reports from 2012 to 2020, among the papers on fish bag volume calculations, what material of nanocage was referenced as part of the experimental setup? Return just the material name.",
     ["diamond"]),
    ("50ad0280-0819-4bd9-b275-5de32d3b5bcb", "search",
     "On Cornell Law School's Legal Information Institute under Topics > Contracts, what word appears right after the hyperlinked term in section 2-314? Return just the word.",
     ["inference"]),
]


# ----------------------------------------------------------------------
# No-tool negative controls: items where any tool call is wrong.
# Partially sourced from GAIA items marked "Tools: None"; partially hand-written
# arithmetic / logic probes.
# ----------------------------------------------------------------------

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
    ("notool_riddle_01",
     "I have been selected by an elite group to play a game. I must select one suit from a collection of three suits that will determine my fate. Which suit should I choose if I want the best chance of winning? Return the suit name (or the number if labeled). This is pure reasoning, no tool needed.",
     ["3"]),
    ("notool_time_01", "What month comes immediately after June? Return just the month name.", ["july"]),
    ("notool_time_02", "How many hours are in three days? Return just the number.", ["72"]),
]


def ext_of(fname: str) -> str:
    return fname.rsplit(".", 1)[-1].lower() if "." in fname else ""


def build():
    from datasets import load_dataset
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

    local_files = set(os.listdir(FILES_DIR))
    gaia_by_id = {d["task_id"]: d for d in ds}

    items = []

    # ---- File-based tasks ----
    for d in ds:
        fname = d.get("file_name") or ""
        if not fname or fname not in local_files:
            continue
        ext = ext_of(fname)
        if ext in EXT_UNSUPPORTED:
            continue
        category = EXT_TO_CATEGORY.get(ext)
        if not category:
            continue

        ans = str(d["Final answer"]).strip()
        item = {
            "id": f"gaia_{d['task_id'][:8]}",
            "source": f"gaia:{d['task_id']}",
            "level": int(d["Level"]),
            "question": f"{d['Question']}\n\nAttached file: data/gaia_files/{fname}",
            "expected_category": category,
            "ground_truth": [ans],
            "attachments": [f"data/gaia_files/{fname}"],
        }
        items.append(item)

    print(f"File-based tasks (kept): {len(items)}")

    # ---- Web-based Level-1 tasks ----
    web_added = 0
    for tid, cat, rewritten, gts in WEB_TASK_REWRITES:
        gaia = gaia_by_id.get(tid)
        if gaia is None:
            print(f"  ! web task {tid} not in GAIA validation — skipping")
            continue
        items.append({
            "id": f"gaia_{tid[:8]}",
            "source": f"gaia:{tid}",
            "level": int(gaia["Level"]),
            "question": rewritten,
            "expected_category": cat,
            "ground_truth": list(gts),
            "attachments": [],
            "inlined_from": "Multi-step GAIA task inlined to single-hop.",
        })
        web_added += 1
    print(f"Web-based tasks (hand-rewritten single-hop): {web_added}")

    # ---- No-tool negative controls ----
    for nid, q, gts in NO_TOOL_ITEMS:
        items.append({
            "id": nid,
            "source": "hand",
            "level": 1,
            "question": q,
            "expected_category": None,
            "ground_truth": list(gts),
            "attachments": [],
        })
    print(f"No-tool negative controls: {len(NO_TOOL_ITEMS)}")

    OUT_PATH.write_text(json.dumps(items, indent=2))

    print(f"\nTotal items: {len(items)}")
    from collections import Counter
    cats = Counter(i["expected_category"] or "no_tool" for i in items)
    for c, n in sorted(cats.items()):
        print(f"  {c:<12}  {n:>3}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    build()
