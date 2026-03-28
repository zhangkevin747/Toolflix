import json
from sentence_transformers import SentenceTransformer

# Use sentence-transformers to generate dense semantic embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load tools
with open("../data/tools.json", "r") as f:
    servers = json.load(f)

endpoints = []

for server in servers:
    server_description = server.get("marketplace", {}).get("description", "")

    for tool in server.get("tools", []):
        # For each tool endpoint, concatenate the tool name + description + server-level description into one string
        text = f"{tool['name']} {tool.get('description', '')} {server_description}"

        endpoints.append({
            "server_id": server["id"],
            "tool_name": tool["name"],
            "category": server.get("category", ""),
            "text": text,
        })

# Generate embeddings
texts = [ep["text"] for ep in endpoints]
embeddings = model.encode(texts, show_progress_bar=True)

# Create a new json file with the embeddings
output = []
for ep, emb in zip(endpoints, embeddings):
    output.append({
        "server_id": ep["server_id"],
        "tool_name": ep["tool_name"],
        "category": ep["category"],
        "text": ep["text"],
        "embedding": emb.tolist(),
    })

with open("../data/embeddings.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated embeddings for {len(output)} tool endpoints.")
