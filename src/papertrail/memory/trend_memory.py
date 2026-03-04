"""
Trend memory – tracks topic/keyword frequency across ingested papers.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from papertrail.schemas.schema import Paper

# Common English stop-words we skip when counting topics
_STOP = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "on", "for", "with",
        "to", "is", "are", "was", "were", "be", "been", "by", "from",
        "at", "as", "we", "our", "that", "this", "these", "those",
        "using", "via", "based", "new", "novel", "approach", "method",
        "paper", "work", "study", "model", "models", "result", "results",
    }
)


class TrendMemory:
    """
    Accumulates keyword frequencies from paper titles and abstracts so
    the agent can surface trending research topics.
    """

    def __init__(self) -> None:
        self._keyword_counter: Counter = Counter()
        self._category_counter: Counter = Counter()
        self._paper_count: int = 0

    # ── Ingestion ──────────────────────────────────────────────

    def update(self, papers: List[Paper]) -> None:
        """Add keywords from *papers* to the trend counters."""
        for paper in papers:
            self._paper_count += 1
            for cat in paper.categories:
                self._category_counter[cat] += 1
            words = self._extract_keywords(paper.title + " " + paper.abstract)
            self._keyword_counter.update(words)

    # ── Queries ────────────────────────────────────────────────

    def top_keywords(self, n: int = 20) -> List[Tuple[str, int]]:
        """Return the *n* most frequent keywords."""
        return self._keyword_counter.most_common(n)

    def top_categories(self, n: int = 10) -> List[Tuple[str, int]]:
        """Return the *n* most frequent arXiv categories."""
        return self._category_counter.most_common(n)

    def keyword_cloud(self, n: int = 30) -> Dict[str, int]:
        return dict(self._keyword_counter.most_common(n))

    def summary(self) -> str:
        """Return a human-readable trend summary."""
        if self._paper_count == 0:
            return "No papers ingested yet."
        kw = ", ".join(k for k, _ in self.top_keywords(10))
        cats = ", ".join(c for c, _ in self.top_categories(5))
        return (
            f"Trends across {self._paper_count} papers:\n"
            f"  Top keywords : {kw}\n"
            f"  Top categories: {cats}"
        )

    # ── Internals ──────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        words = re.findall(r"\b[a-z][a-z\-]{2,}\b", text.lower())
        return [w for w in words if w not in _STOP]

    @property
    def paper_count(self) -> int:
        return self._paper_count

    def clear(self) -> None:
        self._keyword_counter.clear()
        self._category_counter.clear()
        self._paper_count = 0
