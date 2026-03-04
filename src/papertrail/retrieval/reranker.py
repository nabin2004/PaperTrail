"""
Lightweight cross-encoder reranker.

For MVP we use a pure-Python cosine reranker based on the same embedding
model (no extra model download required).  Drop-in cross-encoder can be
swapped in by replacing `_score_pair`.
"""
from __future__ import annotations

from typing import List

import numpy as np

from papertrail.processing.embeddings import embed_query, embed_texts
from papertrail.schemas.schema import SearchResult


def rerank(
    query: str,
    results: List[SearchResult],
    top_k: int = 5,
) -> List[SearchResult]:
    """
    Rerank *results* by cosine similarity between the query embedding
    and each chunk embedding.  Returns the top *top_k* results.

    Parameters
    ----------
    query   : the original search query
    results : list of SearchResult from PaperRetriever
    top_k   : number of results to keep
    """
    if not results:
        return []

    q_vec = embed_query(query)                          # (dim,)
    texts = [r.text for r in results]
    chunk_vecs = embed_texts(texts)                     # (N, dim)

    # Cosine similarity (embeddings are already L2-normalised → dot product = cos)
    scores = chunk_vecs @ q_vec                         # (N,)

    ranked = sorted(
        zip(scores.tolist(), results),
        key=lambda x: x[0],
        reverse=True,
    )

    reranked: List[SearchResult] = []
    for new_score, result in ranked[:top_k]:
        reranked.append(result.model_copy(update={"score": float(new_score)}))
    return reranked
