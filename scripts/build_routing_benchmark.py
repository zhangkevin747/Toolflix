"""
Builds a balanced, single-step tool-routing benchmark.

Each task maps to exactly one of 7 tool categories. Ground truth is the
category, not a string in the tool output — this tests tool routing
(retriever + reranker), not downstream tool-use quality.

Task sources:
- Real GAIA entities, URLs, and files where they cleanly route
- Hand-written natural-language prompts that realistic users would send

Categories: wikipedia, arxiv, fetch, search, pdf, excel, filesystem
"""
import json
from pathlib import Path


WIKIPEDIA = [
    "Find the Wikipedia page for Mercedes Sosa",
    "Look up the Wikipedia article on the Moon",
    "Get the Wikipedia entry for Giganotosaurus",
    "What does Wikipedia say about Antidisestablishmentarianism?",
    "Open the Wikipedia page for the Lord of the Rings novel",
    "Show me the Wikipedia article for A Song of Ice and Fire",
    "Find the Wikipedia page on Eliud Kipchoge",
    "Look up commutative property on Wikipedia",
    "Get the Wikipedia article on Carl Nebel",
    "Find the Wikipedia entry for chinstrap penguin",
    "Search Wikipedia for the principle of double effect",
    "Open the Wikipedia page for the 2019 BAFTA Games Award winner",
]

ARXIV = [
    "Find the arXiv paper with ID 1706.03762",
    "Get the abstract of arXiv 2005.14165",
    "Download the arXiv preprint 2203.02155 about InstructGPT",
    "Look up the arXiv paper titled 'Attention Is All You Need'",
    "Find the LaTeX source for arXiv paper 1810.04805",
    "Retrieve the arXiv listing for hep-lat submissions from January 2020",
    "Search arXiv for recent papers on diffusion models",
    "Get the metadata for arXiv 2310.06770",
    "Find the arXiv paper on GPT-4 technical report",
    "Retrieve arXiv preprint 1512.03385 on residual learning",
    "Look up arXiv paper 2106.09685 on LoRA",
    "Download the PDF of arXiv 1409.0473 on attention mechanisms",
]

FETCH = [
    "Fetch the content at https://en.wikipedia.org/wiki/Moon",
    "Fetch the page at http://journal.finfar.org/journal/archive/fafnir-22014/",
    "Retrieve the HTML at https://oda.oslomet.no/oda-xmlui/handle/10642/3162",
    "Get the webpage at https://journals.le.ac.uk/ojs1/index.php/jist/article/view/733",
    "Download https://www.nature.com/srep/ and show me the content",
    "Fetch the URL https://www.base-search.net/",
    "Grab the content from https://www.bbc.com/news",
    "Retrieve the page at https://example.com/",
    "Fetch https://github.com/openai/gpt-2 readme",
    "Get the HTML content at https://arxiv.org/list/hep-lat/2020-01",
    "Download the page at https://news.ycombinator.com/",
    "Fetch the content of https://www.python.org/downloads/",
]

SEARCH = [
    "Search the web for the current price of Bitcoin",
    "Google who won the 2023 Super Bowl",
    "Find news articles about the latest OpenAI release",
    "Search for restaurants near Times Square",
    "Look up web results for 'best laptops 2024'",
    "Find scholarly articles on transformer architectures",
    "Web search for recent earthquakes in Japan",
    "Search Google Scholar for papers on reinforcement learning from human feedback",
    "Find the latest news about Tesla stock",
    "Search the web for 'open source alternatives to Figma'",
    "Run a web search for recipes for vegan lasagna",
    "Google 'population of Iceland 2024'",
]

PDF = [
    "Extract text from data/gaia_files/e9a2c537-8232-4c3f-85b0-b52de6bcba99.pdf",
    "Read the PDF at data/gaia_files/366e2f2b-8632-4ef2-81eb-bc3877489217.pdf",
    "Parse the PDF file at data/gaia_files/67e8878b-5cef-4375-804e-e6291fdbe78a.pdf",
    "Get the text content of the PDF data/gaia_files/e9a2c537-8232-4c3f-85b0-b52de6bcba99.pdf",
    "Open and read data/gaia_files/366e2f2b-8632-4ef2-81eb-bc3877489217.pdf",
    "What does the PDF at data/gaia_files/67e8878b-5cef-4375-804e-e6291fdbe78a.pdf say?",
    "Extract the table of contents from data/gaia_files/e9a2c537-8232-4c3f-85b0-b52de6bcba99.pdf",
    "Read page 1 of data/gaia_files/366e2f2b-8632-4ef2-81eb-bc3877489217.pdf",
    "Parse the PDF document at data/gaia_files/67e8878b-5cef-4375-804e-e6291fdbe78a.pdf",
    "Get the text from the PDF file data/gaia_files/e9a2c537-8232-4c3f-85b0-b52de6bcba99.pdf",
    "Extract all text from the PDF located at data/gaia_files/366e2f2b-8632-4ef2-81eb-bc3877489217.pdf",
    "Read the PDF document data/gaia_files/67e8878b-5cef-4375-804e-e6291fdbe78a.pdf",
]

EXCEL = [
    "Read the Excel file at data/gaia_files/076c8171-9b3b-49b9-a477-244d2a532826.xlsx",
    "Open the spreadsheet data/gaia_files/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx",
    "Get the contents of data/gaia_files/4d0aa727-86b1-406b-9b33-f870dd14a4a5.xlsx",
    "Parse the xlsx file at data/gaia_files/4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2.xlsx",
    "Read the Excel workbook data/gaia_files/54612da3-fd56-4941-80f4-5eb82330de25.xlsx",
    "Load the spreadsheet at data/gaia_files/5cfb274c-0207-4aa7-9575-6ac0bd95d9b2.xlsx",
    "Show me the rows in data/gaia_files/65afbc8a-89ca-4ad5-8d62-355bb401f61d.xlsx",
    "Read the cells of data/gaia_files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx",
    "Parse the Excel sheet data/gaia_files/7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f.xlsx",
    "Get the table from data/gaia_files/edd4d4f2-1a58-45c4-b038-67337af4e029.xlsx",
    "Read Sheet1 of data/gaia_files/076c8171-9b3b-49b9-a477-244d2a532826.xlsx",
    "Extract data from the Excel file data/gaia_files/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx",
]

FILESYSTEM = [
    "Read the file at data/gaia_files/389793a7-ca17-4e82-81cb-2b3a2391b4b9.txt",
    "Open data/gaia_files/f918266a-b3e0-4914-865d-4faa564f1aef.py and show its contents",
    "List the files in the directory data/gaia_files/",
    "Read the text file data/gaia_files/389793a7-ca17-4e82-81cb-2b3a2391b4b9.txt",
    "Get the contents of the Python script at data/gaia_files/f918266a-b3e0-4914-865d-4faa564f1aef.py",
    "Show me what's inside data/fixtures/",
    "Check if data/gaia_files/cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb.docx exists on disk",
    "Read the docx file at data/gaia_files/cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb.docx",
    "Get file metadata for data/gaia_files/a3fbeb63-0e8c-4a11-bff6-0e3b484c3e9c.pptx",
    "List all files under data/gaia_files/ recursively",
    "Read the contents of data/gaia_files/7dd30055-0198-452e-8c25-f73dbe27dcb8.pdb",
    "Show the first lines of data/gaia_files/389793a7-ca17-4e82-81cb-2b3a2391b4b9.txt",
]


def build():
    buckets = {
        "wikipedia": WIKIPEDIA,
        "arxiv": ARXIV,
        "fetch": FETCH,
        "search": SEARCH,
        "pdf": PDF,
        "excel": EXCEL,
        "filesystem": FILESYSTEM,
    }
    out = []
    for cat, prompts in buckets.items():
        for i, p in enumerate(prompts):
            out.append({
                "task": p,
                "expected_category": cat,
                "id": f"{cat}-{i:02d}",
            })
    return out


if __name__ == "__main__":
    tasks = build()
    path = Path(__file__).parent.parent / "data" / "gaia_routing.json"
    path.write_text(json.dumps(tasks, indent=2))
    from collections import Counter
    counts = Counter(t["expected_category"] for t in tasks)
    print(f"wrote {len(tasks)} tasks to {path}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
