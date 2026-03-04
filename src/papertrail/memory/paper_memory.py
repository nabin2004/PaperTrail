"""
In-session paper memory.

Keeps a lightweight dict of Paper objects in RAM so chains and agents
can reference paper metadata without hitting the filesystem repeatedly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from papertrail.schemas.schema import Paper


class PaperMemory:
    """Thread-unsafe single-session paper registry."""

    def __init__(self) -> None:
        self._store: Dict[str, Paper] = {}

    # ── Write ──────────────────────────────────────────────────

    def add(self, paper: Paper) -> None:
        self._store[paper.arxiv_id] = paper

    def add_many(self, papers: List[Paper]) -> None:
        for p in papers:
            self.add(p)

    # ── Read ───────────────────────────────────────────────────

    def get(self, arxiv_id: str) -> Optional[Paper]:
        return self._store.get(arxiv_id)

    def all(self) -> List[Paper]:
        return list(self._store.values())

    def ids(self) -> List[str]:
        return list(self._store.keys())

    # ── Search ─────────────────────────────────────────────────

    def search_by_category(self, category: str) -> List[Paper]:
        return [p for p in self._store.values() if category in p.categories]

    def search_by_author(self, author: str) -> List[Paper]:
        author_lower = author.lower()
        return [
            p for p in self._store.values()
            if any(author_lower in a.lower() for a in p.authors)
        ]

    # ── Misc ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, arxiv_id: str) -> bool:
        return arxiv_id in self._store

    def clear(self) -> None:
        self._store.clear()
