"""
Reranking utilities – cosine reranker (MVP) and cross-encoder reranker (upgrade).

For MVP we use a pure-Python cosine reranker based on the same embedding
model (no extra model download required).  Drop-in cross-encoder can be
used for higher accuracy at the cost of additional model download.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Literal, Optional

import numpy as np

from papertrail.processing.embeddings import embed_query, embed_texts
from papertrail.schemas.schema import SearchResult

# Default reranker type (can be overridden via env var)
RERANKER_TYPE = os.getenv("RERANKER_TYPE", "cosine")  # "cosine" or "cross-encoder"
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# ──────────────────────────────────────────────────────────────
# Main rerank function
# ──────────────────────────────────────────────────────────────

def rerank(
    query: str,
    results: List[SearchResult],
    top_k: int = 5,
    method: Optional[Literal["cosine", "cross-encoder"]] = None,
) -> List[SearchResult]:
    """
    Rerank *results* and return the top *top_k*.

    Parameters
    ----------
    query   : the original search query
    results : list of SearchResult from PaperRetriever
    top_k   : number of results to keep
    method  : reranking method ("cosine" or "cross-encoder")
              Defaults to RERANKER_TYPE env var or "cosine"

    Returns
    -------
    List[SearchResult]
        Reranked results sorted by descending relevance score.
    """
    if not results:
        return []

    method = method or RERANKER_TYPE

    if method == "cross-encoder":
        return _rerank_cross_encoder(query, results, top_k)
    else:
        return _rerank_cosine(query, results, top_k)


# ──────────────────────────────────────────────────────────────
# Cosine reranker (MVP – no extra model needed)
# ──────────────────────────────────────────────────────────────

def _rerank_cosine(
    query: str,
    results: List[SearchResult],
    top_k: int,
) -> List[SearchResult]:
    """
    Rerank using cosine similarity between query and chunk embeddings.
    Fast but less accurate than cross-encoder.
    """
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


# ──────────────────────────────────────────────────────────────
# Cross-encoder reranker (upgrade – higher accuracy)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _get_cross_encoder(model_name: str):
    """
    Load a cross-encoder model (cached).

    TODO
    ----
    - Implement with sentence-transformers CrossEncoder
    - Support model selection via env var
    - Add GPU support option
    """
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name, max_length=512)
    except ImportError:
        raise ImportError(
            "Cross-encoder reranking requires sentence-transformers.\n"
            "Install with: pip install sentence-transformers"
        )


def _rerank_cross_encoder(
    query: str,
    results: List[SearchResult],
    top_k: int,
    model_name: str = CROSS_ENCODER_MODEL,
) -> List[SearchResult]:
    """
    Rerank using a cross-encoder model for higher accuracy.

    Cross-encoders jointly encode query-document pairs, providing
    more accurate relevance scores than bi-encoder cosine similarity.

    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, good quality)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better quality)
    - BAAI/bge-reranker-base                (excellent quality)
    - BAAI/bge-reranker-large               (best quality, slowest)

    TODO
    ----
    - Batch scoring for efficiency
    - Score normalization
    - Hybrid scoring (combine with cosine)
    """
    model = _get_cross_encoder(model_name)

    # Create query-document pairs
    pairs = [(query, r.text) for r in results]

    # Score all pairs
    scores = model.predict(pairs)

    # Sort by score descending
    ranked = sorted(
        zip(scores.tolist(), results),
        key=lambda x: x[0],
        reverse=True,
    )

    reranked: List[SearchResult] = []
    for new_score, result in ranked[:top_k]:
        reranked.append(result.model_copy(update={"score": float(new_score)}))
    return reranked


# ──────────────────────────────────────────────────────────────
# Hybrid reranker (future – combine multiple signals)
# ──────────────────────────────────────────────────────────────

def rerank_hybrid(
    query: str,
    results: List[SearchResult],
    top_k: int = 5,
    cosine_weight: float = 0.3,
    cross_encoder_weight: float = 0.7,
) -> List[SearchResult]:
    """
    Combine cosine and cross-encoder scores for hybrid reranking.

    TODO
    ----
    - Implement score fusion
    - Score normalization (min-max or z-score)
    - Reciprocal Rank Fusion (RRF) alternative
    - BM25 signal integration
    """
    raise NotImplementedError(
        "Hybrid reranking not yet implemented.\n"
        "Use rerank() with method='cosine' or method='cross-encoder'."
    )


# ──────────────────────────────────────────────────────────────
# Cohere reranker (cloud alternative)
# ──────────────────────────────────────────────────────────────

def rerank_cohere(
    query: str,
    results: List[SearchResult],
    top_k: int = 5,
    model: str = "rerank-english-v3.0",
) -> List[SearchResult]:
    """
    Rerank using Cohere's rerank API (cloud-based).

    Requires COHERE_API_KEY environment variable.

    TODO
    ----
    - Implement Cohere API integration
    - Handle rate limiting
    - Cost tracking
    """
    raise NotImplementedError(
        "Cohere reranking not yet implemented.\n"
        "Install with: pip install cohere\n"
        "Set COHERE_API_KEY environment variable."
    )
