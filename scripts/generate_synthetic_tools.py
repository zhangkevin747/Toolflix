"""
Generate synthetic tool variants to 5x the tool pool.

For each real tool, create variants with:
  - Different descriptions (misleading, underselling, overpromising)
  - Different failure rates (0%, 30%, 50%, 70%)
  - Category-specific quirks (works for some inputs, fails for others)

This creates controlled quality variance — exactly the underdifferentiation
and description-reality gap the reranker should learn to resolve.
"""
import json
import random
import copy

random.seed(42)

with open("../data/tools.json") as f:
    servers = json.load(f)

# === Variant templates ===

# Description modifiers: (prefix, suffix, quality_label)
# Paraphrase templates — each rewrites a tool description to say the same thing
# differently. Similar enough that the retriever ranks them close, but not
# identical. Each variant also has a different failure mode.
PARAPHRASE_TEMPLATES = {
    "fetch": {
        "clone-reliable": [
            "Retrieves content from web URLs and returns it in a clean, readable format.",
            "Downloads web page content and provides extracted text for processing.",
            "Accesses remote web resources and delivers their textual content.",
            "Pulls web content from the given URL and returns structured output.",
        ],
        "clone-flaky": [
            "Connects to web servers and extracts page content for further analysis.",
            "Grabs website data and converts it into a usable text format.",
            "Loads web pages and parses their content into a readable form.",
            "Obtains online content from URLs and delivers it as processed text.",
        ],
        "clone-truncator": [
            "Fetches web resources and provides a summary of the page content.",
            "Captures web page text and returns a condensed version of the content.",
            "Retrieves website content and outputs a brief text representation.",
            "Accesses URL content and returns key extracted information.",
        ],
        "clone-stale": [
            "Interfaces with web endpoints to retrieve and return page content.",
            "Processes web URLs and extracts their primary textual information.",
            "Resolves web addresses and provides the resulting document content.",
            "Handles HTTP requests to fetch and return website text content.",
        ],
    },
    "pdf": {
        "clone-reliable": [
            "Reads PDF documents and extracts their text content for processing.",
            "Parses PDF files to retrieve text, metadata, and structural information.",
            "Opens PDF documents and returns their readable text representation.",
            "Processes PDF files and outputs extracted text and document data.",
        ],
        "clone-flaky": [
            "Handles PDF file processing to extract textual and form data.",
            "Analyzes PDF documents and retrieves their embedded text content.",
            "Works with PDF files to pull out readable text and field values.",
            "Manages PDF document reading and content extraction tasks.",
        ],
        "clone-truncator": [
            "Extracts a preview of text content from PDF documents.",
            "Reads PDF files and provides a brief summary of their content.",
            "Scans PDF documents and returns key text excerpts.",
            "Processes PDFs to generate a compact text representation.",
        ],
        "clone-stale": [
            "Interfaces with PDF documents to extract and return text data.",
            "Provides PDF reading capabilities for text and metadata extraction.",
            "Enables PDF content access for text retrieval and document analysis.",
            "Supports PDF file operations including text extraction and search.",
        ],
    },
    "search": {
        "clone-reliable": [
            "Searches the web using multiple engines and returns relevant results.",
            "Performs web searches and delivers ranked results with snippets.",
            "Queries search engines to find relevant web pages and information.",
            "Executes web searches and provides structured result listings.",
        ],
        "clone-flaky": [
            "Conducts online searches across multiple sources for relevant content.",
            "Looks up information on the web and returns matching results.",
            "Runs search queries against web indexes to find relevant pages.",
            "Performs internet searches and compiles matching results.",
        ],
        "clone-truncator": [
            "Searches the web and returns brief result summaries.",
            "Queries search engines and provides compact result listings.",
            "Finds web content matching your query with short descriptions.",
            "Delivers quick search results with titles and brief snippets.",
        ],
        "clone-stale": [
            "Interfaces with web search APIs to retrieve query results.",
            "Provides search functionality across multiple web sources.",
            "Enables web searching with configurable result formatting.",
            "Supports multi-engine web search with result aggregation.",
        ],
    },
    "filesystem": {
        "clone-reliable": [
            "Reads files and directories from the local filesystem.",
            "Accesses local files to read content, list directories, and get metadata.",
            "Provides filesystem operations including reading, listing, and file info.",
            "Manages local file access for reading content and directory browsing.",
        ],
        "clone-flaky": [
            "Handles local filesystem operations for file and directory access.",
            "Works with the local filesystem to read files and explore directories.",
            "Performs file system tasks including content reading and file lookup.",
            "Interacts with local storage to retrieve file contents and metadata.",
        ],
        "clone-truncator": [
            "Reads files and returns a brief preview of their content.",
            "Accesses local files and provides compact content summaries.",
            "Retrieves file content with condensed output for quick review.",
            "Reads filesystem content and returns abbreviated results.",
        ],
        "clone-stale": [
            "Interfaces with the local filesystem for file operations.",
            "Provides file access capabilities for reading and directory listing.",
            "Enables local file operations including read, list, and search.",
            "Supports filesystem interactions for content retrieval and browsing.",
        ],
    },
    "excel": {
        "clone-reliable": [
            "Opens and reads Excel workbooks, extracting cell data and sheet metadata.",
            "Processes Excel files to retrieve spreadsheet data, formulas, and formatting.",
            "Provides full Excel file manipulation including read, write, and chart creation.",
            "Handles Excel spreadsheet operations for data extraction and analysis.",
        ],
        "clone-flaky": [
            "Works with Excel files to extract data, formulas, and sheet structure.",
            "Manages Excel workbook operations for reading and writing spreadsheet data.",
            "Interacts with Excel documents to pull cell values and worksheet info.",
            "Performs Excel file processing including data retrieval and sheet management.",
        ],
        "clone-truncator": [
            "Reads Excel files and returns a preview of the spreadsheet contents.",
            "Extracts a compact summary of data from Excel workbooks.",
            "Opens Excel files and provides abbreviated cell data output.",
            "Processes Excel spreadsheets and returns condensed data extracts.",
        ],
        "clone-stale": [
            "Interfaces with Excel file formats for spreadsheet data operations.",
            "Provides Excel workbook access for data reading and sheet inspection.",
            "Enables Excel file interactions including cell reading and metadata.",
            "Supports Excel document processing for data extraction workflows.",
        ],
    },
    "wikipedia": {
        "clone-reliable": [
            "Searches Wikipedia and retrieves full article content and summaries.",
            "Queries Wikipedia's knowledge base to find and return article text.",
            "Accesses Wikipedia articles, summaries, sections, and related topics.",
            "Provides Wikipedia lookup for article content and factual information.",
        ],
        "clone-flaky": [
            "Connects to Wikipedia to search articles and retrieve page content.",
            "Looks up topics on Wikipedia and returns article text and metadata.",
            "Fetches Wikipedia entries and delivers article content for processing.",
            "Performs Wikipedia queries to find and return encyclopedia content.",
        ],
        "clone-truncator": [
            "Searches Wikipedia and returns brief article summaries.",
            "Retrieves compact Wikipedia article excerpts for quick reference.",
            "Queries Wikipedia and provides condensed article overviews.",
            "Looks up Wikipedia content and returns shortened article text.",
        ],
        "clone-stale": [
            "Interfaces with Wikipedia's API for article search and retrieval.",
            "Provides Wikipedia access for searching and reading encyclopedia articles.",
            "Enables Wikipedia queries for article content and topic exploration.",
            "Supports Wikipedia lookups including search, summaries, and full articles.",
        ],
    },
    "arxiv": {
        "clone-reliable": [
            "Searches arXiv for academic papers and retrieves full text and abstracts.",
            "Queries the arXiv repository to find and download research papers.",
            "Accesses arXiv papers by ID or search query, returning text and metadata.",
            "Provides arXiv paper lookup including abstracts, sections, and citations.",
        ],
        "clone-flaky": [
            "Connects to arXiv to search for and retrieve academic paper content.",
            "Looks up research papers on arXiv and returns their text and metadata.",
            "Fetches arXiv paper content including abstracts and full document text.",
            "Performs arXiv searches to find and return scientific paper data.",
        ],
        "clone-truncator": [
            "Searches arXiv and returns brief paper abstracts and titles.",
            "Retrieves compact summaries of arXiv papers for quick review.",
            "Queries arXiv and provides condensed paper information.",
            "Looks up arXiv papers and returns shortened content excerpts.",
        ],
        "clone-stale": [
            "Interfaces with arXiv's API for paper search and retrieval.",
            "Provides arXiv access for searching and reading research papers.",
            "Enables arXiv queries for paper abstracts and full text access.",
            "Supports arXiv lookups including search, download, and reading.",
        ],
    },
}

DESC_VARIANTS = {
    "clone-reliable": {
        "synth_type": "random_failure",
        "fail_rate": 0.0,
    },
    "clone-flaky": {
        "synth_type": "random_failure",
        "fail_rate": 0.35,
    },
    "clone-truncator": {
        "synth_type": "truncate",
        "max_chars": 80,
    },
    "clone-stale": {
        "synth_type": "stale",
    },
}

# Per-category niche variants: only work for specific artifact patterns
NICHE_VARIANTS = {
    "fetch": [
        {
            "name_suffix": "-static-only",
            "desc": "Optimized for static HTML pages. Extracts clean text content from simple websites.",
            "works_for": ["paulgraham.com", "gnu.org", "motherfuckingwebsite"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.85,
        },
        {
            "name_suffix": "-js-renderer",
            "desc": "Full JavaScript rendering engine for dynamic web applications. Handles SPAs and client-side content.",
            "works_for": ["react.dev", "news.ycombinator"],
            "fail_rate_match": 0.1,
            "fail_rate_mismatch": 0.7,
        },
    ],
    "pdf": [
        {
            "name_suffix": "-forms-specialist",
            "desc": "Specialized PDF form field extractor. Reads fillable form data with high accuracy.",
            "works_for": ["w9.pdf", "fillable_sample.pdf", "form_example.pdf"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.8,
        },
        {
            "name_suffix": "-text-extractor",
            "desc": "High-quality PDF text extraction with layout preservation.",
            "works_for": ["attention.pdf", "dummy.pdf"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.75,
        },
    ],
    "search": [
        {
            "name_suffix": "-tech-focused",
            "desc": "Search engine optimized for technical documentation and programming resources.",
            "works_for": ["Stripe", "Python requests", "RFC", "FAISS", "mcp model context"],
            "fail_rate_match": 0.1,
            "fail_rate_mismatch": 0.7,
        },
        {
            "name_suffix": "-news-focused",
            "desc": "Real-time news and current events search engine with fresh results.",
            "works_for": ["OpenAI", "NVIDIA", "log4j", "CVE"],
            "fail_rate_match": 0.1,
            "fail_rate_mismatch": 0.7,
        },
    ],
    "filesystem": [
        {
            "name_suffix": "-read-only",
            "desc": "Fast read-only filesystem access. Reads files and lists directories efficiently.",
            "works_for": ["read", "list", "show", "contents"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.8,
        },
        {
            "name_suffix": "-write-specialist",
            "desc": "File creation and modification toolkit. Creates files, directories, and edits content.",
            "works_for": ["create", "write", "mkdir", "new file"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.8,
        },
    ],
    "excel": [
        {
            "name_suffix": "-read-only",
            "desc": "Read-only Excel file access. Extracts data, metadata, and cell values from spreadsheets.",
            "works_for": ["read", "get", "how many", "what is", "list"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.85,
        },
        {
            "name_suffix": "-chart-specialist",
            "desc": "Excel chart and visualization creation tool. Creates bar, pie, and line charts from data.",
            "works_for": ["chart", "pie", "bar", "line", "visualization"],
            "fail_rate_match": 0.1,
            "fail_rate_mismatch": 0.8,
        },
    ],
    "wikipedia": [
        {
            "name_suffix": "-science-focused",
            "desc": "Wikipedia lookup specialized for science and technology topics.",
            "works_for": ["machine learning", "quantum", "CRISPR", "neural", "photosynthesis", "dark matter"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.75,
        },
        {
            "name_suffix": "-history-focused",
            "desc": "Wikipedia search optimized for historical and cultural topics.",
            "works_for": ["Renaissance", "history", "general relativity", "climate"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.75,
        },
    ],
    "arxiv": [
        {
            "name_suffix": "-nlp-focused",
            "desc": "arXiv paper search specialized for NLP and language model research.",
            "works_for": ["language model", "retrieval", "generation", "prompting", "attention", "transformer"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.8,
        },
        {
            "name_suffix": "-cv-focused",
            "desc": "arXiv paper search optimized for computer vision and image generation.",
            "works_for": ["diffusion", "image", "vision", "visual"],
            "fail_rate_match": 0.05,
            "fail_rate_mismatch": 0.8,
        },
    ],
}

# === Generate variants ===

synthetic_tools = []  # List of (server_entry, synthetic_config)
synthetic_configs = {}  # synthetic_id -> config for failure injection

# Pick representative tools from each category to clone
TOOLS_TO_CLONE = {
    "fetch": [
        ("zcaceres-fetch-mcp", "fetch_readable"),
        ("fetcher-mcp", "fetch_url"),
        ("duckduckgo-mcp", "fetch_content"),
        ("mcp-server-fetch", "fetch"),
    ],
    "pdf": [
        ("fabriqa-pdf-reader", "read-pdf"),
        ("mcp-pdf-forms", "extract_form_fields"),
        ("mcp-pdf-forms", "extract_text"),
        ("pdf-reader-mcp", "read_pdf"),
    ],
    "search": [
        ("open-websearch", "search"),
        ("duckduckgo-mcp", "search"),
    ],
    "filesystem": [
        ("mcp-server-filesystem", "read_file"),
        ("mcp-server-filesystem", "list_directory"),
        ("mcp-server-filesystem", "get_file_info"),
        ("mcp-server-filesystem", "search_files"),
    ],
    "excel": [
        ("excel-mcp-haris", "read_data_from_excel"),
        ("excel-mcp-haris", "create_chart"),
        ("excel-mcp-guillehr2", "open_workbook_tool"),
        ("excel-mcp-guillehr2", "write_sheet_data_tool"),
    ],
    "wikipedia": [
        ("wikipedia-mcp-rudra", "search_wikipedia"),
        ("wikipedia-mcp-rudra", "get_summary"),
        ("wikipedia-mcp-azzar", "search"),
        ("wikipedia-mcp-azzar", "getPageSummary"),
    ],
    "arxiv": [
        ("arxiv-mcp-blazick", "search_papers"),
        ("arxiv-mcp-blazick", "download_paper"),
        ("arxiv-latex-mcp", "get_paper_abstract"),
        ("arxiv-latex-mcp", "list_paper_sections"),
    ],
}

# Build index of existing tools
tool_lookup = {}
for server in servers:
    for tool in server.get("tools", []):
        key = (server["id"], tool["name"])
        tool_lookup[key] = (server, tool)


_synth_counter = 0

def make_synthetic_server(base_server, base_tool, variant_id, description,
                          tool_name_suffix="", extra_tools=None):
    """Create a synthetic server entry based on a real one.

    IDs are opaque (v001, v002...) so the agent can't infer quality from the name.
    """
    global _synth_counter
    _synth_counter += 1
    # Opaque ID: category + base tool hint + counter. No quality info leaked.
    synth_id = f"{base_server['category']}-{base_tool['name']}-v{_synth_counter:03d}"
    synth_id = synth_id.replace("/", "-")[:60]

    synth_tool = copy.deepcopy(base_tool)
    synth_tool["name"] = base_tool["name"]
    synth_tool["description"] = description

    # Generate a plausible author name
    authors = ["devtools-io", "mcp-contrib", "toolsmith", "opentools",
               "api-utils", "serverkit", "mcpworks", "databridge",
               "toolhub", "coreutils", "webtools-ai", "fileops"]
    author = random.choice(authors)

    synth_server = {
        "id": synth_id,
        "name": f"{author}/{base_tool['name']}",
        "category": base_server["category"],
        "github": f"https://github.com/{author}/{synth_id}",
        "install": copy.deepcopy(base_server["install"]),
        "server_info": {"name": synth_id, "version": "0.1.0"},
        "marketplace": {
            "author": author,
            "description": description,
        },
        "tools": [synth_tool] + (extra_tools or []),
        "_synthetic": {
            "real_server_id": base_server["id"],
            "real_tool_name": base_tool["name"],
        },
    }
    return synth_server


count = 0

STALE_RESPONSES = {
    "fetch": "Error: Connection reset by peer. The server closed the connection unexpectedly.",
    "pdf": "Error: Unable to parse document. The file may be corrupted or in an unsupported format.",
    "search": '{"results": [], "message": "No results found for your query. Try broadening your search terms."}',
    "filesystem": "Error: Permission denied. The process does not have sufficient privileges to access this resource.",
    "excel": "Error: Unable to open workbook. The file format may be unsupported or the file is corrupted.",
    "wikipedia": '{"error": "Article not found. The requested page does not exist on Wikipedia."}',
    "arxiv": '{"error": "Paper not found. The requested arXiv ID does not exist or has been withdrawn."}',
}

# 1. Clone variants — identical descriptions, different failure modes
for category, tool_keys in TOOLS_TO_CLONE.items():
    for server_id, tool_name in tool_keys:
        if (server_id, tool_name) not in tool_lookup:
            continue
        base_server, base_tool = tool_lookup[(server_id, tool_name)]

        for variant_name, variant_cfg in DESC_VARIANTS.items():
            # Pick a paraphrased description for this category+variant
            cat_paraphrases = PARAPHRASE_TEMPLATES.get(category, {}).get(variant_name, [])
            if cat_paraphrases:
                desc = random.choice(cat_paraphrases)
            else:
                desc = base_tool["description"]  # fallback to original
            synth = make_synthetic_server(
                base_server, base_tool, variant_name, desc,
            )
            synth_id = synth["id"]

            synth_type = variant_cfg["synth_type"]
            if synth_type == "random_failure":
                synthetic_configs[synth_id] = {
                    "type": "random_failure",
                    "real_server_id": server_id,
                    "real_tool_name": tool_name,
                    "fail_rate": variant_cfg["fail_rate"],
                }
            elif synth_type == "truncate":
                synthetic_configs[synth_id] = {
                    "type": "truncate",
                    "real_server_id": server_id,
                    "real_tool_name": tool_name,
                    "max_chars": variant_cfg["max_chars"],
                }
            elif synth_type == "stale":
                synthetic_configs[synth_id] = {
                    "type": "stale",
                    "real_server_id": server_id,
                    "real_tool_name": tool_name,
                    "stale_response": STALE_RESPONSES.get(category, "Error: Service unavailable."),
                    "stale_rate": 0.6,  # 60% of the time returns stale, 40% works
                }

            synthetic_tools.append(synth)
            count += 1

# 2. Niche variants (work for specific inputs, fail for others)
for category, niches in NICHE_VARIANTS.items():
    tool_keys = TOOLS_TO_CLONE.get(category, [])
    for server_id, tool_name in tool_keys[:2]:
        if (server_id, tool_name) not in tool_lookup:
            continue
        base_server, base_tool = tool_lookup[(server_id, tool_name)]

        for niche in niches:
            # Niche tools use the SAME description as the original
            synth = make_synthetic_server(
                base_server, base_tool, "niche",
                base_tool["description"],  # identical description
                tool_name_suffix=niche["name_suffix"],
            )

            synth_id = synth["id"]
            synthetic_configs[synth_id] = {
                "type": "niche",
                "real_server_id": server_id,
                "real_tool_name": tool_name,
                "works_for": niche["works_for"],
                "fail_rate_match": niche["fail_rate_match"],
                "fail_rate_mismatch": niche["fail_rate_mismatch"],
            }
            synthetic_tools.append(synth)
            count += 1

# 3. Quality tiers — same description, different reliability
TIER_CONFIGS = [
    ("gold", "random_failure", {"fail_rate": 0.05}),
    ("silver", "random_failure", {"fail_rate": 0.20}),
    ("bronze", "truncate", {"max_chars": 120}),
    ("free", "stale", {}),
]

for category, tool_keys in TOOLS_TO_CLONE.items():
    for server_id, tool_name in tool_keys[:2]:
        if (server_id, tool_name) not in tool_lookup:
            continue
        base_server, base_tool = tool_lookup[(server_id, tool_name)]

        for tier_name, synth_type, tier_cfg in TIER_CONFIGS:
            # Map tiers to variant types for paraphrase lookup
            tier_variant_map = {"gold": "clone-reliable", "silver": "clone-flaky",
                                "bronze": "clone-truncator", "free": "clone-stale"}
            variant_key = tier_variant_map.get(tier_name, "clone-reliable")
            cat_paraphrases = PARAPHRASE_TEMPLATES.get(category, {}).get(variant_key, [])
            desc = random.choice(cat_paraphrases) if cat_paraphrases else base_tool["description"]
            synth = make_synthetic_server(
                base_server, base_tool, tier_name, desc,
            )

            synth_id = synth["id"]
            cfg = {
                "type": synth_type,
                "real_server_id": server_id,
                "real_tool_name": tool_name,
            }
            if synth_type == "random_failure":
                cfg["fail_rate"] = tier_cfg["fail_rate"]
            elif synth_type == "truncate":
                cfg["max_chars"] = tier_cfg["max_chars"]
            elif synth_type == "stale":
                cfg["stale_response"] = STALE_RESPONSES.get(category, "Error: Service unavailable.")
                cfg["stale_rate"] = 0.65
            synthetic_configs[synth_id] = cfg
            synthetic_tools.append(synth)
            count += 1

# === Save ===

# Add synthetic tools to servers list
all_servers = servers + synthetic_tools

with open("../data/tools.json", "w") as f:
    json.dump(all_servers, f, indent=2)

with open("../data/synthetic_configs.json", "w") as f:
    json.dump(synthetic_configs, f, indent=2)

print(f"Original servers: {len(servers)}")
print(f"Synthetic variants: {count}")
print(f"Total servers: {len(all_servers)}")
total_tools = sum(len(s.get("tools", [])) for s in all_servers)
print(f"Total endpoints: {total_tools}")

# Category breakdown
from collections import Counter
cats = Counter(s["category"] for s in all_servers)
for cat, n in sorted(cats.items()):
    print(f"  {cat}: {n} servers")
