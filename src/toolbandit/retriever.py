"""Stage 1: retrieval. Narrow ~482 listings down to a handful of candidates.

This is plain TF-IDF text matching (no neural net, no external library). Given a
short query like "weather forecast", it returns the listings whose text is most
similar, by cosine similarity of TF-IDF vectors.

Its only job is *recall*: make sure the right tool is somewhere in the candidate
set. It cannot tell good tools from broken ones, because broken tools have
good-looking descriptions. Telling quality apart is the bandit's job (stage 2).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, NamedTuple

from .data import searchable_text, words


class Candidate(NamedTuple):
    listing_id: str
    score: float  # cosine similarity to the query, in [0, 1]


class Retriever:
    def __init__(self, listings: list[dict[str, Any]]) -> None:
        self.ids = [listing["listing_id"] for listing in listings]

        # Count words in each listing's text.
        docs = [Counter(words(searchable_text(listing))) for listing in listings]

        # IDF: rare words are more informative than common ones.
        doc_count = max(len(docs), 1)
        seen_in = Counter()
        for doc in docs:
            seen_in.update(doc.keys())
        self.idf = {
            word: math.log((doc_count + 1) / (n + 1)) + 1.0
            for word, n in seen_in.items()
        }

        # Pre-compute each listing's weighted vector and its length.
        self.vectors = [self._weigh(doc) for doc in docs]
        self.lengths = [self._length(vec) for vec in self.vectors]

    def search(self, query: str, limit: int) -> list[Candidate]:
        """Return up to `limit` listings most similar to the query."""
        q_vec = self._weigh(Counter(words(query)))
        q_len = self._length(q_vec)
        if not q_vec or q_len == 0:
            return []

        results: list[Candidate] = []
        for listing_id, vec, length in zip(self.ids, self.vectors, self.lengths):
            if length == 0:
                continue
            dot = sum(weight * vec.get(word, 0.0) for word, weight in q_vec.items())
            if dot <= 0:
                continue
            results.append(Candidate(listing_id, dot / (q_len * length)))

        results.sort(key=lambda c: c.score, reverse=True)
        return results[:limit]

    def _weigh(self, counts: Counter) -> dict[str, float]:
        """Turn raw word counts into TF-IDF weights."""
        return {
            word: (1.0 + math.log(count)) * self.idf.get(word, 1.0)
            for word, count in counts.items()
        }

    @staticmethod
    def _length(vec: dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in vec.values()))
