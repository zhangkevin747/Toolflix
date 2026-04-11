"""
Generate tasks where at least one tool in our pool can solve them.
Tasks are designed around known tool capabilities, not hypothetical ones.
"""
import json
import random

random.seed(42)

# === Artifacts that our tools CAN handle ===

# Fetch: static pages verified to return real content
fetch_urls = [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://www.paulgraham.com/greatwork.html",
    "https://www.gnu.org/philosophy/free-sw.html",
    "https://motherfuckingwebsite.com",
    "https://react.dev/learn",
    "https://news.ycombinator.com",
    "https://medium.com/@karpathy/software-2-0-a64152b37c35",
]

# PDF: local files that our tools can read
pdf_local = [
    "data/pdfs/attention.pdf",
    "data/pdfs/w9.pdf",
    "data/pdfs/fillable_sample.pdf",
    "data/pdfs/form_example.pdf",
    "data/pdfs/dummy.pdf",
]

# No PDF URLs — local files only (verified to work with our tools)
pdf_urls = []

# Search: queries that web search can answer
search_queries = [
    "Stripe PaymentIntents API documentation",
    "Python requests library timeout parameter",
    "RFC 7231 HTTP semantics",
    "OpenAI latest model release 2024",
    "latest CVE for log4j",
    "best open source alternatives to Figma",
    "how to implement RAG with LangChain",
    "comparison of vector databases 2024",
    "mcp model context protocol server specification",
    "FAISS IVF vs HNSW index tradeoffs",
    "NVIDIA earnings report 2024",
    "rust vs go performance comparison",
]

# Filesystem: paths that exist in our fixtures
FIXTURES = "data/fixtures"
fs_files = [
    f"{FIXTURES}/config.json",
    f"{FIXTURES}/README.md",
    f"{FIXTURES}/data.csv",
    f"{FIXTURES}/large_log.txt",
    f"{FIXTURES}/big_data.json",
    f"{FIXTURES}/deeply/nested/path/to/file.txt",
    f"{FIXTURES}/project/src/main.py",
    f"{FIXTURES}/project/tests/test_main.py",
    f"{FIXTURES}/project/docs/index.md",
]
fs_dirs = [
    f"{FIXTURES}/",
    f"{FIXTURES}/project/",
    f"{FIXTURES}/project/src/",
    f"{FIXTURES}/project/tests/",
    f"{FIXTURES}/project/docs/",
]

# === Task templates matched to tool capabilities ===

# Fetch: tools can fetch HTML, text, markdown from URLs
fetch_templates = [
    "Get the content of {url} and tell me what it's about.",
    "Read {url} and give me a bullet-point summary.",
    "Retrieve the page at {url} and convert it to markdown.",
    "Pull the main article from {url}, ignoring navigation and ads.",
    "Get the full text content from {url} so I can search through it.",
    "I need a clean, readable version of the article at {url}.",
    "Scrape {url} and tell me the page title and first paragraph.",
    "Go to {url} and extract all headings and subheadings.",
    "Fetch the page at {url} and tell me how many words are on it.",
    "I want the raw HTML source code of {url}.",
    "Summarize the main content of this page: {url}",
    "Go to {url} and extract the key points.",
]

# PDF: tools can extract text, form fields, search text, read metadata
pdf_text_templates = [
    "Read this PDF and summarize its key findings: {path}",
    "Extract all the text from {path}.",
    "I need the full text of {path} for indexing.",
    "Read {path} and extract the abstract and conclusion.",
    "Pull out the key statistics and figures mentioned in {path}.",
    "Read this research paper and explain the methodology: {path}",
]

pdf_form_templates = [
    "What form fields are in this PDF: {path}",
    "Extract the filled-in values from all form fields in {path}.",
    "List all the fillable fields and their current values in {path}.",
    "Read the form at {path} and tell me what information it requires.",
]

pdf_search_templates = [
    "Search for the word 'attention' in {path}.",
    "Read {path} and find any mentions of specific dates or deadlines.",
    "Extract all the text from {path} and search for mentions of 'model'.",
]

pdf_meta_templates = [
    "How many pages does {path} have?",
    "Get the metadata (author, title, page count) from {path}.",
]

# Search: tools can web search and return results
search_templates = [
    "Search the web and find {query}.",
    "Look up {query} and give me the top results.",
    "Find me the most authoritative source for {query}.",
    "I need you to research {query} and give me a summary.",
    "Search online for {query} and return the URLs of relevant pages.",
    "Find recent information about {query}.",
    "Do a web search for {query} and compile what you find.",
    "Search for {query} and return structured results I can work with.",
    "Find the official source for {query}.",
    "Research {query} and give me links to the best resources.",
    "Search the web for {query} and summarize the top 3 results.",
]

# Filesystem: tools can read files, list dirs, create files, get info
fs_read_templates = [
    "Read the contents of {path}.",
    "Show me what's in {path}.",
    "Read {path} and tell me what it contains.",
]

fs_list_templates = [
    "List all the files in {path}.",
    "Show me the directory tree structure of {path}.",
    "What files are in {path}?",
]

fs_info_templates = [
    "Get the file size and modification date of {path}.",
    "Check if {path} exists and tell me its type (file or directory).",
    "Count the number of lines in {path}.",
]

fs_write_templates = [
    "Create a new file at {path}/test_output.txt with the content 'Hello, World!'.",
    "Create nested directories at {path}/a/b/c/.",
]

fs_search_templates = [
    "Search for the string 'TODO' in all files under {path}.",
    "Find all .py files under {path}.",
    "Search recursively in {path} for files containing 'test'.",
]

# === Excel: tools can create, read, write, format, chart ===
EXCEL_FIXTURES = "data/fixtures"
# (path, sheet_name) pairs so tasks include the sheet name
excel_files_with_sheets = [
    (f"{EXCEL_FIXTURES}/sales_data.xlsx", "Sales"),
    (f"{EXCEL_FIXTURES}/employee_records.xlsx", "Employees"),
    (f"{EXCEL_FIXTURES}/inventory.xlsx", "Current Stock"),
    (f"{EXCEL_FIXTURES}/budget.xlsx", "Budget 2025"),
    (f"{EXCEL_FIXTURES}/grades.xlsx", "Grades"),
]
excel_files = [f for f, _ in excel_files_with_sheets]

excel_read_templates = [
    "Read the data from {path} (sheet: {sheet}) and tell me what it contains.",
    "Read {path} (sheet: {sheet}) and summarize the data.",
    "How many rows and columns are in {path} (sheet: {sheet})?",
    "Read the first 10 rows from {path} (sheet: {sheet}).",
    "Read all data from {path} (sheet: {sheet}).",
    "Get all cell values from {path} (sheet: {sheet}).",
]

excel_write_templates = [
    "Create a new Excel workbook at {path}/output_report.xlsx with a sheet called Summary.",
    "Add a new worksheet called Analysis to {path}.",
    "Copy the sheet {sheet} in {path} to a new sheet called Backup.",
    "Insert 3 new rows at row 5 in {path} (sheet: {sheet}).",
    "Write the data [[\"Name\", \"Score\"], [\"Alice\", 95], [\"Bob\", 87]] to a new workbook at {path}/test_output.xlsx.",
]

excel_formula_templates = [
    "Add a SUM formula to the bottom of the first numeric column in {path} (sheet: {sheet}).",
    "Add an AVERAGE formula for all numeric columns in {path} (sheet: {sheet}).",
    "Add a formula to calculate the total across all rows in {path} (sheet: {sheet}).",
]

excel_chart_templates = [
    "Create a bar chart in {path} (sheet: {sheet}) from the data.",
    "Create a pie chart in {path} (sheet: {sheet}) showing the distribution of values in the first numeric column.",
    "Create a line chart in {path} (sheet: {sheet}) comparing all numeric columns.",
]

excel_format_templates = [
    "Format the header row of {path} (sheet: {sheet}) to be bold with a blue background.",
    "Convert the data range in {path} (sheet: {sheet}) into a formatted Excel table.",
    "Create a pivot table from {path} (sheet: {sheet}) summarizing the data.",
]

# === Wikipedia: tools can search, get articles, summaries, sections ===
wiki_topics = [
    "machine learning",
    "quantum computing",
    "CRISPR gene editing",
    "climate change",
    "blockchain technology",
    "neural network",
    "photosynthesis",
    "general relativity",
    "Renaissance",
    "artificial intelligence",
    "mRNA vaccine",
    "dark matter",
]

wiki_articles = [
    "Python (programming language)",
    "Large language model",
    "Transformer (deep learning model)",
    "Attention (machine learning)",
    "Reinforcement learning",
    "Natural language processing",
    "Convolutional neural network",
    "Backpropagation",
    "Generative adversarial network",
    "Transfer learning",
]

wiki_search_templates = [
    "Search Wikipedia for articles about {topic}.",
    "Find Wikipedia articles related to {topic}.",
    "Look up {topic} on Wikipedia and give me the top results.",
    "Search Wikipedia for {topic} and list the most relevant articles.",
]

wiki_article_templates = [
    "Get the full Wikipedia article on {article}.",
    "Read the Wikipedia page for {article} and summarize it.",
    "Get the summary of the Wikipedia article on {article}.",
    "What does Wikipedia say about {article}?",
    "Pull the key facts from the Wikipedia page on {article}.",
    "Get the sections and headings of the Wikipedia article on {article}.",
    "Find related topics linked from the Wikipedia page on {article}.",
]

# === ArXiv: tools can search papers, download, read, get LaTeX ===
arxiv_queries = [
    "tool retrieval large language models",
    "wide and deep learning recommender systems",
    "model context protocol",
    "retrieval augmented generation",
    "chain of thought prompting",
    "diffusion models image generation",
    "reinforcement learning from human feedback",
    "sparse mixture of experts",
    "vision language models",
    "graph neural networks",
    "federated learning privacy",
    "attention mechanism transformers",
]

arxiv_ids = [
    "2405.16089",  # COLT / ToolRet
    "1706.03762",  # Attention Is All You Need
    "1606.07792",  # Wide & Deep Learning
    "2005.14165",  # GPT-3
    "2302.13971",  # LLaMA
    "2203.15556",  # Chinchilla
    "2307.09288",  # Llama 2
    "2312.10997",  # Mixtral
]

arxiv_search_templates = [
    "Search arXiv for recent papers about {query}.",
    "Find papers on arXiv about {query}.",
    "Look up {query} on arXiv and show me the top results.",
    "Search arXiv for {query} and give me paper titles and abstracts.",
]

arxiv_paper_templates = [
    "Get the abstract of arXiv paper {arxiv_id}.",
    "Download and read the full text of arXiv paper {arxiv_id}.",
    "What are the sections of arXiv paper {arxiv_id}?",
    "Get the LaTeX source of arXiv paper {arxiv_id}.",
    "Read arXiv paper {arxiv_id} and summarize the methodology.",
    "Extract the introduction from arXiv paper {arxiv_id}.",
]

# === Generate tasks ===

tasks = []

def add(category, templates, artifacts, count, key="url"):
    for _ in range(count):
        template = random.choice(templates)
        artifact = random.choice(artifacts)
        task = template.format(**{key: artifact})
        tasks.append({"category": category, "task": task, "artifact": artifact})

# Fetch: 500
add("fetch", fetch_templates, fetch_urls, 500)

# PDF: 500 (mix of text, form, search, metadata)
add("pdf", pdf_text_templates, pdf_local, 200, "path")
add("pdf", pdf_form_templates, ["data/pdfs/w9.pdf", "data/pdfs/fillable_sample.pdf", "data/pdfs/form_example.pdf"], 120, "path")
add("pdf", pdf_search_templates, pdf_local, 100, "path")
add("pdf", pdf_meta_templates, pdf_local, 80, "path")

# Search: 500
add("search", search_templates, search_queries, 500, "query")

# Filesystem: 500
add("filesystem", fs_read_templates, fs_files, 120, "path")
add("filesystem", fs_list_templates, fs_dirs, 100, "path")
add("filesystem", fs_info_templates, fs_files, 100, "path")
add("filesystem", fs_write_templates, fs_dirs, 80, "path")
add("filesystem", fs_search_templates, fs_dirs, 100, "path")

# Excel: 500 (with sheet names)
def add_excel(templates, count):
    for _ in range(count):
        template = random.choice(templates)
        path, sheet = random.choice(excel_files_with_sheets)
        task = template.format(path=path, sheet=sheet)
        tasks.append({"category": "excel", "task": task, "artifact": path})

add_excel(excel_read_templates, 150)
add_excel(excel_write_templates, 100)
add_excel(excel_formula_templates, 80)
add_excel(excel_chart_templates, 80)
add_excel(excel_format_templates, 90)

# Wikipedia: 500
add("wikipedia", wiki_search_templates, wiki_topics, 200, "topic")
add("wikipedia", wiki_article_templates, wiki_articles, 300, "article")

# ArXiv: 500
add("arxiv", arxiv_search_templates, arxiv_queries, 250, "query")
add("arxiv", arxiv_paper_templates, arxiv_ids, 250, "arxiv_id")

# Stratified interleave
by_cat = {}
for t in tasks:
    cat = t["category"]
    if cat not in by_cat:
        by_cat[cat] = []
    by_cat[cat].append(t)

# Shuffle within each category
for cat in by_cat:
    random.shuffle(by_cat[cat])

cats = ["fetch", "pdf", "search", "filesystem", "excel", "wikipedia", "arxiv"]
stratified = []
max_len = max(len(by_cat[c]) for c in cats)
for i in range(max_len):
    for cat in cats:
        if i < len(by_cat[cat]):
            stratified.append(by_cat[cat][i])

with open("../data/tasks_solvable_stratified.json", "w") as f:
    json.dump(stratified, f, indent=2)

print(f"Generated {len(stratified)} stratified solvable tasks")
for cat in cats:
    print(f"  {cat}: {len(by_cat[cat])}")

# === Generate held-out test tasks (different seed, same artifacts) ===
random.seed(99)
test_tasks = []

def add_test(category, templates, artifacts, count, key="url"):
    for _ in range(count):
        template = random.choice(templates)
        artifact = random.choice(artifacts)
        task = template.format(**{key: artifact})
        test_tasks.append({"category": category, "task": task, "artifact": artifact})

# 50 per category = 350 test tasks
add_test("fetch", fetch_templates, fetch_urls, 50)
add_test("pdf", pdf_text_templates + pdf_form_templates + pdf_search_templates + pdf_meta_templates,
         pdf_local, 50, "path")
add_test("search", search_templates, search_queries, 50, "query")

all_fs_templates = fs_read_templates + fs_list_templates + fs_info_templates + fs_search_templates
add_test("filesystem", all_fs_templates, fs_files + fs_dirs, 50, "path")

all_excel_templates = excel_read_templates + excel_formula_templates + excel_chart_templates + excel_format_templates
for _ in range(50):
    template = random.choice(all_excel_templates)
    path, sheet = random.choice(excel_files_with_sheets)
    task = template.format(path=path, sheet=sheet)
    test_tasks.append({"category": "excel", "task": task, "artifact": path})

add_test("wikipedia", wiki_search_templates, wiki_topics, 25, "topic")
add_test("wikipedia", wiki_article_templates, wiki_articles, 25, "article")

add_test("arxiv", arxiv_search_templates, arxiv_queries, 25, "query")
add_test("arxiv", arxiv_paper_templates, arxiv_ids, 25, "arxiv_id")

random.shuffle(test_tasks)

with open("../data/test_tasks.json", "w") as f:
    json.dump(test_tasks, f, indent=2)

print(f"\nGenerated {len(test_tasks)} solvable test tasks")
for cat in cats:
    count = sum(1 for t in test_tasks if t["category"] == cat)
    print(f"  {cat}: {count}")
