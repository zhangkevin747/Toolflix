"""
Verify that each task in our solvable set can actually be completed by at least one tool.
Bypass the agent — call tools directly with correct arguments and check results.
"""
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mcp_client import MCPClient

# Load tools
with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'tools.json')) as f:
    servers = json.load(f)

tool_index = {}
for server in servers:
    for tool in server.get('tools', []):
        key = f"{server['id']}/{tool['name']}"
        tool_index[key] = {
            'server_id': server['id'],
            'tool_name': tool['name'],
            'install': server.get('install', {}),
        }

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
client = MCPClient(timeout=30)


def resolve_path(p):
    if p.startswith('data/'):
        return os.path.join(PROJECT_ROOT, p)
    return p


def try_tool(tool_key, arguments):
    """Call a tool and return (success, preview)."""
    info = tool_index[tool_key]
    result = client.call_tool(
        server_id=info['server_id'],
        tool_name=info['tool_name'],
        arguments=arguments,
        install=info['install'],
    )
    preview = str(result)[:300]
    is_error = 'error' in preview.lower() or result.get('isError', False)
    has_content = len(str(result)) > 50 and not is_error
    return has_content, preview


# Define verifiable task-tool pairs per category
VERIFY_PAIRS = {
    'fetch': [
        # (artifact_pattern, tool_key, make_args_fn)
        ('paulgraham.com', 'zcaceres-fetch-mcp/fetch_readable', lambda url: {'url': url, 'max_length': 5000}),
        ('paulgraham.com', 'fetcher-mcp/fetch_url', lambda url: {'url': url, 'timeout': 15000, 'extractContent': True, 'maxLength': 5000}),
        ('wikipedia.org', 'zcaceres-fetch-mcp/fetch_readable', lambda url: {'url': url, 'max_length': 5000}),
        ('wikipedia.org', 'fetcher-mcp/fetch_url', lambda url: {'url': url, 'timeout': 15000, 'extractContent': True, 'maxLength': 5000}),
        ('gnu.org', 'zcaceres-fetch-mcp/fetch_readable', lambda url: {'url': url, 'max_length': 5000}),
        ('motherfuckingwebsite', 'zcaceres-fetch-mcp/fetch_txt', lambda url: {'url': url, 'max_length': 5000}),
        ('react.dev', 'duckduckgo-mcp/fetch_content', lambda url: {'url': url, 'max_length': 5000}),
        ('news.ycombinator', 'zcaceres-fetch-mcp/fetch_html', lambda url: {'url': url, 'max_length': 10000}),
        ('arstechnica.com', 'zcaceres-fetch-mcp/fetch_readable', lambda url: {'url': url, 'max_length': 5000}),
        ('medium.com', 'zcaceres-fetch-mcp/fetch_readable', lambda url: {'url': url, 'max_length': 5000}),
    ],
    'pdf': [
        ('data/pdfs/attention.pdf', 'fabriqa-pdf-reader/read-pdf', lambda p: {'file': resolve_path(p), 'pages': '1-3', 'clean_text': True}),
        ('data/pdfs/attention.pdf', 'mcp-pdf-forms/extract_text', lambda p: {'pdf_path': resolve_path(p), 'start_page': 0, 'end_page': 3}),
        ('data/pdfs/w9.pdf', 'mcp-pdf-forms/extract_form_fields', lambda p: {'pdf_path': resolve_path(p)}),
        ('data/pdfs/fillable_sample.pdf', 'mcp-pdf-forms/extract_form_fields', lambda p: {'pdf_path': resolve_path(p)}),
        ('data/pdfs/form_example.pdf', 'mcp-pdf-forms/extract_form_fields', lambda p: {'pdf_path': resolve_path(p)}),
        ('data/pdfs/dummy.pdf', 'fabriqa-pdf-reader/read-pdf', lambda p: {'file': resolve_path(p), 'pages': 'all', 'clean_text': True}),
        ('data/pdfs/dummy.pdf', 'mcp-pdf-forms/extract_text', lambda p: {'pdf_path': resolve_path(p), 'start_page': 0, 'end_page': 1}),
        ('data/pdfs/attention.pdf', 'mcp-pdf-forms/search_text', lambda p: {'pdf_path': resolve_path(p), 'pattern': 'attention'}),
        ('data/pdfs/attention.pdf', 'fabriqa-pdf-reader/pdf-metadata', lambda p: {'file': resolve_path(p)}),
        ('data/pdfs/w9.pdf', 'fabriqa-pdf-reader/pdf-metadata', lambda p: {'file': resolve_path(p)}),
        ('https://arxiv.org/pdf/1706.03762', 'mcp-pdf-forms/extract_text', lambda p: {'pdf_path': p, 'start_page': 0, 'end_page': 3}),
    ],
    'search': [
        ('Stripe', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
        ('Python requests', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
        ('RFC 7231', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
        ('OpenAI', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
        ('log4j', 'duckduckgo-mcp/search', lambda q: {'query': q, 'max_results': 5}),
        ('Figma', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
        ('RAG', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
        ('vector databases', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
        ('mcp model context', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
        ('FAISS', 'open-websearch/search', lambda q: {'query': q, 'limit': 5}),
    ],
    'filesystem': [
        ('data/fixtures/config.json', 'mcp-server-filesystem/read_file', lambda p: {'path': resolve_path(p)}),
        ('data/fixtures/README.md', 'mcp-server-filesystem/read_file', lambda p: {'path': resolve_path(p)}),
        ('data/fixtures/data.csv', 'mcp-server-filesystem/read_file', lambda p: {'path': resolve_path(p)}),
        ('data/fixtures/large_log.txt', 'mcp-server-filesystem/read_text_file', lambda p: {'path': resolve_path(p), 'head': 50}),
        ('data/fixtures/big_data.json', 'mcp-server-filesystem/read_file', lambda p: {'path': resolve_path(p)}),
        ('data/fixtures/deeply/nested/path/to/file.txt', 'mcp-server-filesystem/read_file', lambda p: {'path': resolve_path(p)}),
        ('data/fixtures/project/src/main.py', 'mcp-server-filesystem/read_file', lambda p: {'path': resolve_path(p)}),
        ('data/fixtures/project/', 'mcp-server-filesystem/list_directory', lambda p: {'path': resolve_path(p.rstrip('/'))}),
        ('data/fixtures/project/src/', 'mcp-server-filesystem/list_directory', lambda p: {'path': resolve_path(p.rstrip('/'))}),
        ('data/fixtures/project/tests/', 'mcp-server-filesystem/list_directory', lambda p: {'path': resolve_path(p.rstrip('/'))}),
        ('data/fixtures/config.json', 'mcp-server-filesystem/get_file_info', lambda p: {'path': resolve_path(p)}),
        ('data/fixtures/', 'mcp-server-filesystem/search_files', lambda p: {'path': resolve_path(p.rstrip('/')), 'pattern': '*.py'}),
        ('data/fixtures/', 'mcp-server-filesystem/create_directory', lambda p: {'path': resolve_path(p) + 'a/b/c'}),
    ],
}


def main():
    results = {'pass': 0, 'fail': 0, 'details': []}

    # Map artifact patterns to actual full artifacts
    fetch_url_map = {
        'paulgraham.com': 'https://www.paulgraham.com/greatwork.html',
        'wikipedia.org': 'https://en.wikipedia.org/wiki/Large_language_model',
        'gnu.org': 'https://www.gnu.org/philosophy/free-sw.html',
        'motherfuckingwebsite': 'https://motherfuckingwebsite.com',
        'react.dev': 'https://react.dev/learn',
        'news.ycombinator': 'https://news.ycombinator.com',
        'arstechnica.com': 'https://arstechnica.com/science/2024/01/the-year-in-science-2023/',
        'medium.com': 'https://medium.com/@karpathy/software-2-0-a64152b37c35',
    }

    for cat, pairs in VERIFY_PAIRS.items():
        print(f"\n=== {cat.upper()} ===")
        for artifact_pattern, tool_key, make_args in pairs:
            # Resolve artifact pattern to actual value
            if cat == 'fetch':
                actual_artifact = fetch_url_map.get(artifact_pattern, artifact_pattern)
            else:
                actual_artifact = artifact_pattern
            args = make_args(actual_artifact)
            try:
                success, preview = try_tool(tool_key, args)
            except Exception as e:
                success = False
                preview = str(e)[:200]

            status = "PASS" if success else "FAIL"
            results['pass' if success else 'fail'] += 1
            print(f"  {status}: {tool_key} on '{artifact_pattern}'")
            if not success:
                print(f"    {preview[:150]}")

            results['details'].append({
                'category': cat,
                'artifact': artifact_pattern,
                'tool': tool_key,
                'success': success,
            })

            time.sleep(0.5)  # Don't hammer servers

    print(f"\n{'='*50}")
    print(f"PASS: {results['pass']}, FAIL: {results['fail']}")
    print(f"Success rate: {results['pass']/(results['pass']+results['fail'])*100:.0f}%")

    # Save results
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'verified_pairs.json'), 'w') as f:
        json.dump(results['details'], f, indent=2)


if __name__ == '__main__':
    main()
