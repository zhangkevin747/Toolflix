"""
Generate user tasks with concrete inputs baked in.
The task should be something that requires an agent to decompose and reason what kind of tool it needs.
Do not use knowledge about our actual toolset.
"""
import json
import random

random.seed(42)

# --- Artifacts ---

fetch_static = [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://www.paulgraham.com/greatwork.html",
    "https://www.gnu.org/philosophy/free-sw.html",
    "https://motherfuckingwebsite.com",
]

fetch_js = [
    "https://react.dev/learn",
    "https://www.airbnb.com/s/San-Francisco",
    "https://news.ycombinator.com",
    "https://www.zillow.com/homes/San-Francisco,-CA_rb/",
    "https://weather.com/weather/today/l/USCA0987:1:US",
]

fetch_readability = [
    "https://www.nytimes.com/2024/01/15/technology/ai-openai-chatgpt.html",
    "https://arstechnica.com/science/2024/01/the-year-in-science-2023/",
    "https://medium.com/@karpathy/software-2-0-a64152b37c35",
]

fetch_youtube = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=aircAruvnKk",
    "https://www.youtube.com/watch?v=kCc8FmEb1nY",
]

fetch_antibot = [
    "https://www.linkedin.com/in/satyanadella",
    "https://www.amazon.com/dp/B0D5B9MT91",
]

pdf_url_text = [
    "https://arxiv.org/pdf/1706.03762",
    "https://arxiv.org/pdf/2005.14165",
    "https://www.berkshirehathaway.com/2024ar/2024ar.pdf",
    "https://corporate.lululemon.com/~/media/Files/L/Lululemon/investors/annual-reports/lululemon-2024-annual-report.pdf",
]

pdf_url_form = [
    "https://www.irs.gov/pub/irs-pdf/fw9.pdf",
    "https://themodernfirm.com/wp-content/uploads/2017/12/Sample-Fillable-PDF.pdf",
    "http://foersom.com/net/HowTo/data/OoPdfFormExample.pdf",
]

pdf_url_scanned = [
    "http://solutions.weblite.ca/pdfocrx/scansmpl.pdf",
    "https://www.csun.edu/sites/default/files/pdf_scanned_ocr.pdf",
]

pdf_url_simple = [
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "https://www.govinfo.gov/media/govinfo_Overview_1019.pdf",
]

pdf_url_large = [
    "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2024.pdf",
    "https://www.unilever.com/files/unilever-annual-report-and-accounts-2024.pdf",
]

pdf_local = [
    "data/pdfs/attention.pdf",
    "data/pdfs/w9.pdf",
    "data/pdfs/fillable_sample.pdf",
    "data/pdfs/form_example.pdf",
    "data/pdfs/scansmpl.pdf",
    "data/pdfs/dummy.pdf",
]

pdf_local_forms = ["data/pdfs/w9.pdf", "data/pdfs/fillable_sample.pdf", "data/pdfs/form_example.pdf"]
pdf_local_text = ["data/pdfs/attention.pdf", "data/pdfs/dummy.pdf"]
pdf_local_scanned = ["data/pdfs/scansmpl.pdf"]

search_precise = [
    "Stripe PaymentIntents API documentation",
    "Python requests library timeout parameter",
    "RFC 7231 HTTP semantics",
]

search_recent = [
    "OpenAI latest model release 2024",
    "NVIDIA stock price today",
    "latest CVE for log4j",
]

search_broad = [
    "best open source alternatives to Figma",
    "how to implement RAG with LangChain",
    "comparison of vector databases 2024",
]

search_niche = [
    "mcp model context protocol server specification",
    "reciprocal rank fusion algorithm implementation",
    "FAISS IVF vs HNSW index tradeoffs",
]

FIXTURES_DIR = "data/fixtures"
_fs_simple = ["config.json", "README.md", "data.csv"]
_fs_nested = ["project/src/", "project/tests/", "project/docs/"]
_fs_deep = ["deeply/nested/path/to/file.txt"]
_fs_large = ["large_log.txt", "big_data.json"]
_fs_special = ["image.png", ".env", "empty_file.txt"]

# All filesystem paths are sandboxed under data/fixtures/
fs_simple = [f"{FIXTURES_DIR}/{f}" for f in _fs_simple]
fs_nested = [f"{FIXTURES_DIR}/{f}" for f in _fs_nested]
fs_deep = [f"{FIXTURES_DIR}/{f}" for f in _fs_deep]
fs_large = [f"{FIXTURES_DIR}/{f}" for f in _fs_large]
fs_special = [f"{FIXTURES_DIR}/{f}" for f in _fs_special]

# --- Task templates ---

# Fetch: user tasks that require fetching web content
fetch_task_templates = [
    "Summarize the main content of this page: {url}",
    "Go to {url} and extract the key points.",
    "Read {url} and give me a bullet-point summary.",
    "I need you to pull all the text from {url} and find mentions of machine learning.",
    "Get the content of {url} and tell me what it's about.",
    "Fetch {url} and extract any data tables you find.",
    "Pull the main article from {url}, ignoring navigation and ads.",
    "I want the raw HTML source code of {url}.",
    "Go through {url} and list all the external links.",
    "Read {url} and extract the author's main argument.",
    "Check {url} and tell me if it mentions pricing information.",
    "Get the full text content from {url} so I can search through it.",
    "I need a clean, readable version of the article at {url}.",
    "Look at {url} and extract any structured data or lists.",
    "Retrieve the page at {url} and convert it to markdown.",
    "Scrape {url} and tell me the page title and first paragraph.",
    "Go to {url} and extract all headings and subheadings.",
    "I need you to grab the content from {url} — make sure you get the dynamically loaded parts too.",
    "Fetch the page at {url} and tell me how many words are on it.",
    "Read through {url} and extract any contact information.",
]

# YouTube-specific
youtube_task_templates = [
    "Get the transcript of this video and summarize it: {url}",
    "I need the full text of what's said in this video: {url}",
    "Pull the captions from {url} and find where they discuss neural networks.",
    "Watch this video and give me the key takeaways: {url}",
    "Extract the transcript from {url} so I can search for specific topics.",
    "Get the subtitles from this YouTube video: {url}",
    "I need a text version of this lecture: {url}",
    "Grab the captions from {url} and tell me the main topics covered.",
]

# PDF: user tasks involving PDF documents
pdf_text_templates = [
    "Read this PDF and summarize its key findings: {path}",
    "Extract all the text from {path} and search for mentions of 'revenue'.",
    "I need you to read {path} and give me a summary of each section.",
    "Pull the text from pages 1-5 of {path}.",
    "Read {path} and extract the abstract and conclusion.",
    "Go through {path} and list all the references cited.",
    "Extract the table of contents from {path}.",
    "I need the full text of {path} for indexing.",
    "Read {path} and find any mentions of specific dates or deadlines.",
    "Summarize the executive summary section of {path}.",
    "Extract all the numbered lists from {path}.",
    "Read this research paper and explain the methodology: {path}",
    "Pull out the key statistics and figures mentioned in {path}.",
    "I need you to read {path} and identify the main conclusions.",
    "Go through {path} and extract any financial figures.",
]

pdf_form_templates = [
    "What form fields are in this PDF: {path}",
    "Extract the filled-in values from all form fields in {path}.",
    "Read the form at {path} and tell me what information it requires.",
    "I need to see what data has been entered into the form at {path}.",
    "List all the fillable fields and their current values in {path}.",
    "Check {path} and tell me which form fields are required vs optional.",
    "Extract the taxpayer information from {path}.",
    "What fields does this form have and what are they set to: {path}",
]

pdf_table_templates = [
    "Extract the tables from {path} and preserve the row/column structure.",
    "Pull the financial data tables from {path} into a structured format.",
    "I need the data from Table 1 in {path}.",
    "Read {path} and extract all tabular data as CSV.",
    "Go through {path} and pull out the performance metrics table.",
]

pdf_scanned_templates = [
    "This is a scanned document. Extract whatever text you can from {path}.",
    "Read the text from this scanned PDF: {path}",
    "I need you to OCR this document and extract the content: {path}",
    "Try to read the text from this image-based PDF: {path}",
]

# Search: user tasks that require web searching
search_task_templates = [
    "Search the web and find {query}.",
    "Look up {query} and give me the top results.",
    "Find me the most authoritative source for {query}.",
    "I need you to research {query} and give me a summary.",
    "Search online for {query} and return the URLs of relevant pages.",
    "Find recent information about {query}.",
    "Do a web search for {query} and compile what you find.",
    "I need to find {query} — search for it and give me what you get.",
    "Look up {query} online and tell me what the consensus is.",
    "Search for {query} and return structured results I can work with.",
    "Find the official source for {query}.",
    "Research {query} and give me links to the best resources.",
    "Search the web for {query} and summarize the top 3 results.",
    "I need up-to-date information on {query}. Search for it.",
    "Find and compare different sources about {query}.",
]

# Filesystem: user tasks involving local files
fs_task_templates = [
    "List all the files in the {path} directory.",
    "Read the contents of {path} and show me what's in it.",
    "Find all files matching *.py in {path}.",
    "Create a new file at {path} with some default content.",
    "Search for the string 'TODO' in all files under {path}.",
    "Get the file size and modification date of {path}.",
    "Read the first 50 lines of {path}.",
    "Check if {path} exists and tell me its type (file or directory).",
    "Find all files larger than 1MB in {path}.",
    "Create the directory structure at {path} if it doesn't exist.",
    "Replace all occurrences of 'old_value' with 'new_value' in {path}.",
    "Count the number of lines in {path}.",
    "Copy {path} to a backup location.",
    "Read {path} and extract all JSON keys.",
    "List all files modified in the last 24 hours under {path}.",
    "Append a new entry to {path}.",
    "Search recursively in {path} for files containing 'error'.",
    "Show me the directory tree structure of {path}.",
]

# --- Generate tasks ---

tasks = []

def add_tasks(category, templates, artifacts, count, artifact_key="url"):
    """Generate tasks by combining templates with artifacts."""
    for _ in range(count):
        template = random.choice(templates)
        artifact = random.choice(artifacts)
        task = template.format(**{artifact_key: artifact})
        tasks.append({"category": category, "task": task, "artifact": artifact})

# Fetch: 250 tasks
# Static HTML
add_tasks("fetch", fetch_task_templates, fetch_static, 50)
# JS-rendered
add_tasks("fetch", fetch_task_templates, fetch_js, 60)
# Readability
add_tasks("fetch", fetch_task_templates, fetch_readability, 40)
# YouTube
add_tasks("fetch", youtube_task_templates, fetch_youtube, 50)
# Anti-bot
add_tasks("fetch", fetch_task_templates, fetch_antibot, 30)
# Mixed fetch (any URL)
all_fetch_urls = fetch_static + fetch_js + fetch_readability + fetch_antibot
add_tasks("fetch", fetch_task_templates, all_fetch_urls, 20)

# PDF: 250 tasks
# Text-based at URLs
add_tasks("pdf", pdf_text_templates, pdf_url_text, 40, "path")
# Form PDFs at URLs
add_tasks("pdf", pdf_form_templates, pdf_url_form, 20, "path")
# Scanned PDFs at URLs
add_tasks("pdf", pdf_scanned_templates, pdf_url_scanned, 15, "path")
# Simple PDFs at URLs
add_tasks("pdf", pdf_text_templates, pdf_url_simple, 15, "path")
# Large PDFs at URLs
add_tasks("pdf", pdf_text_templates + pdf_table_templates, pdf_url_large, 25, "path")
# Local text PDFs
add_tasks("pdf", pdf_text_templates, pdf_local_text, 30, "path")
# Local form PDFs
add_tasks("pdf", pdf_form_templates, pdf_local_forms, 35, "path")
# Local scanned PDFs
add_tasks("pdf", pdf_scanned_templates, pdf_local_scanned, 15, "path")
# Local table extraction
add_tasks("pdf", pdf_table_templates, ["data/pdfs/attention.pdf"], 15, "path")
# Mixed local
add_tasks("pdf", pdf_text_templates, pdf_local, 40, "path")

# Search: 250 tasks
add_tasks("search", search_task_templates, search_precise, 65, "query")
add_tasks("search", search_task_templates, search_recent, 65, "query")
add_tasks("search", search_task_templates, search_broad, 60, "query")
add_tasks("search", search_task_templates, search_niche, 60, "query")

# Filesystem: 250 tasks
all_fs_paths = fs_simple + fs_nested + fs_deep + fs_large + fs_special
add_tasks("filesystem", fs_task_templates, all_fs_paths, 250, "path")

# Shuffle
random.shuffle(tasks)

# Save
with open("../data/tasks.json", "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Generated {len(tasks)} tasks.")
for cat in ["fetch", "pdf", "search", "filesystem"]:
    count = sum(1 for t in tasks if t["category"] == cat)
    print(f"  {cat}: {count}")

# Show a few examples per category
print("\n--- Sample tasks ---")
for cat in ["fetch", "pdf", "search", "filesystem"]:
    cat_tasks = [t for t in tasks if t["category"] == cat]
    print(f"\n{cat.upper()}:")
    for t in random.sample(cat_tasks, min(3, len(cat_tasks))):
        print(f"  {t['task'][:100]}")

# --- Generate held-out test tasks (different seed, same sandbox) ---
random.seed(99)
test_tasks = []

def add_test_tasks(category, templates, artifacts, count, artifact_key="url"):
    for _ in range(count):
        template = random.choice(templates)
        artifact = random.choice(artifacts)
        task = template.format(**{artifact_key: artifact})
        test_tasks.append({"category": category, "task": task, "artifact": artifact})

# 50 per category = 200 test tasks
add_test_tasks("fetch", fetch_task_templates, fetch_static + fetch_js + fetch_readability + fetch_antibot, 50)
add_test_tasks("pdf", pdf_text_templates + pdf_form_templates + pdf_table_templates, pdf_url_text + pdf_url_form + pdf_local, 50, "path")
add_test_tasks("search", search_task_templates, search_precise + search_recent + search_broad + search_niche, 50, "query")
all_fs_paths = fs_simple + fs_nested + fs_deep + fs_large + fs_special
add_test_tasks("filesystem", fs_task_templates, all_fs_paths, 50, "path")

random.shuffle(test_tasks)

with open("../data/test_tasks.json", "w") as f:
    json.dump(test_tasks, f, indent=2)

print(f"\nGenerated {len(test_tasks)} test tasks.")
for cat in ["fetch", "pdf", "search", "filesystem"]:
    count = sum(1 for t in test_tasks if t["category"] == cat)
    print(f"  {cat}: {count}")
