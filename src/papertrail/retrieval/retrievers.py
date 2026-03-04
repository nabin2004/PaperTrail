"""
High-level retriever that combines embedding + FAISS search,
then enriches results with paper metadata.
"""
from __future__ import annotations

from typing import List, Optional

from papertrail.ingestion.metadata import load_metadata
from papertrail.processing.embeddings import embed_query
from papertrail.retrieval.vectorstore import VectorStore
from papertrail.schemas.schema import SearchResult


class PaperRetriever:
    """
    Semantic retriever over the FAISS index.

    Usage::

        retriever = PaperRetriever()
        results = retriever.retrieve("attention mechanism transformers", k=10)
    """

    def __init__(self, index_name: str = "papers") -> None:
        self.store = VectorStore(index_name=index_name)

    def retrieve(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Retrieve the top-k most relevant chunks for *query*.

        Returns a list of SearchResult objects sorted by descending similarity.
        """
        if self.store.total_chunks == 0:
            return []

        q_vec = embed_query(query)
        raw = self.store.search(q_vec, k=k)  # List[(score, chunk_dict)]

        results: List[SearchResult] = []
        for score, chunk in raw:
            # Enrich with paper-level metadata when available
            paper = load_metadata(chunk["paper_id"])
            results.append(
                SearchResult(
                    paper_id=chunk["paper_id"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=score,
                    title=paper.title if paper else None,
                    authors=paper.authors if paper else None,
                    published=str(paper.published.date()) if paper else None,
                    abstract=paper.abstract if paper else None,
                )
            )
        return results

    def is_empty(self) -> bool:
        return self.store.total_chunks == 0

    def stats(self) -> dict:
        return {
            "total_chunks": self.store.total_chunks,
            "indexed_papers": len(self.store.indexed_paper_ids()),
        }
