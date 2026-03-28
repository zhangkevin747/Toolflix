"""
Phase 1: Semantic Retrieval

Always use OOP paradigm.
The retriever takes as input the query, and outputs top k tool recommendations to the agent.
"""
import json
import threading
import numpy as np
from sentence_transformers import SentenceTransformer


BLOCKED_ENDPOINTS = {
    ("scrapling-fetch-mcp", "s_fetch_page"),
    ("scrapling-fetch-mcp", "s_fetch_pattern"),
    ("fetch-mcp", "fetch_youtube_transcript"),
    ("zcaceres-fetch-mcp", "fetch_youtube_transcript"),
}


class Retriever:
    """The retriever takes as input the query, and outputs top k tool recommendations to the agent."""

    def __init__(self, embeddings_path: str, tools_path: str):
        # Sentence transformer embeddings on endpoint descriptions
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._encode_lock = threading.Lock()

        # Load precomputed embeddings
        with open(embeddings_path, "r") as f:
            self.endpoints = json.load(f)

        self.endpoint_embeddings = np.array([ep["embedding"] for ep in self.endpoints])
        self.endpoint_norms = np.linalg.norm(self.endpoint_embeddings, axis=1)

        # Load full tool data for returning schemas
        with open(tools_path, "r") as f:
            servers = json.load(f)

        # Index at the tool endpoint level (server_id + tool_name)
        self.tool_index = {}
        for server in servers:
            for tool in server.get("tools", []):
                key = (server["id"], tool["name"])
                self.tool_index[key] = {
                    "server_id": server["id"],
                    "server_name": server["name"],
                    "tool_name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {}),
                    "install": server.get("install", {}),
                    "category": server.get("category", ""),
                }

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-N candidate endpoints for a given query."""
        with self._encode_lock:
            query_emb = self.model.encode([query])[0]
        query_norm = np.linalg.norm(query_emb)

        # Cosine similarity
        sims = np.dot(self.endpoint_embeddings, query_emb) / (self.endpoint_norms * query_norm)

        top_indices = np.argsort(sims)[::-1]

        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            ep = self.endpoints[idx]
            key = (ep["server_id"], ep["tool_name"])
            if key in BLOCKED_ENDPOINTS:
                continue
            tool_info = self.tool_index.get(key, {})
            results.append({
                **tool_info,
                "similarity": float(sims[idx]),
            })

        return results
