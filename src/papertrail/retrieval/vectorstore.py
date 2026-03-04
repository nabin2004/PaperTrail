"""
FAISS-backed vector store for paper chunks.

The index is persisted to data/indices/<name>.faiss + <name>.meta.json.
Reloaded automatically from disk on next instantiation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = _DATA_DIR / "indices"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP (inner-product / cosine when
    embeddings are L2-normalised).

    Parameters
    ----------
    index_name : str
        File-system name for the saved index (no extension).
    dim : int
        Embedding dimension; must match the model used for indexing.
    """

    def __init__(self, index_name: str = "papers", dim: int = 384) -> None:
        self.index_name = index_name
        self.dim = dim
        self._index_path = INDEX_DIR / f"{index_name}.faiss"
        self._meta_path = INDEX_DIR / f"{index_name}.meta.json"

        if self._index_path.exists() and self._meta_path.exists():
            self._load()
        else:
            self._init_empty()

    # ──────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────

    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """
        Add embeddings + their metadata dicts.

        Parameters
        ----------
        embeddings : np.ndarray (N, dim) float32
        chunks     : list of dicts with at least {paper_id, chunk_id, text, source}
        """
        import faiss

        if embeddings.shape[0] == 0:
            return
        assert embeddings.shape[1] == self.dim, (
            f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}"
        )
        self._index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)
        self._save()

    # ──────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[float, Dict]]:
        """
        Return up to *k* (score, chunk_dict) pairs sorted by similarity.
        Score is the inner-product (higher = more similar for normalised vecs).
        """
        if self._index.ntotal == 0:
            return []
        q = query_embedding.reshape(1, -1).astype(np.float32)
        k_actual = min(k, self._index.ntotal)
        scores, indices = self._index.search(q, k_actual)
        results: List[Tuple[float, Dict]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((float(score), self._chunks[idx]))
        return results

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal

    def indexed_paper_ids(self) -> List[str]:
        """Return unique paper IDs that have been indexed."""
        return list({c["paper_id"] for c in self._chunks})

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(
            json.dumps(self._chunks, ensure_ascii=False), encoding="utf-8"
        )

    def _load(self) -> None:
        import faiss
        self._index = faiss.read_index(str(self._index_path))
        self._chunks = json.loads(self._meta_path.read_text(encoding="utf-8"))

    def _init_empty(self) -> None:
        import faiss
        self._index = faiss.IndexFlatIP(self.dim)
        self._chunks: List[Dict] = []

    def reset(self) -> None:
        """Wipe the index and all stored chunks from disk."""
        self._init_empty()
        for p in (self._index_path, self._meta_path):
            if p.exists():
                p.unlink()
