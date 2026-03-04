"""
Embedding utilities using HuggingFace sentence-transformers via LangChain.

Default model : all-MiniLM-L6-v2  (384-dim, ~90 MB, downloads on first use)
Override      : set EMBEDDING_MODEL env var
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import numpy as np

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # dimension of all-MiniLM-L6-v2


# ──────────────────────────────────────────────────────────────
# Model singleton
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _get_model(model_name: str):
    """Cache the HuggingFaceEmbeddings instance to avoid repeated loads."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a list of strings.

    Returns
    -------
    np.ndarray of shape (N, dim), dtype float32, L2-normalised
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model(model_name)
    vecs = model.embed_documents(texts)
    return np.array(vecs, dtype=np.float32)


def embed_query(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a single query string.

    Returns
    -------
    np.ndarray of shape (dim,), dtype float32, L2-normalised
    """
    model = _get_model(model_name)
    vec = model.embed_query(text)
    return np.array(vec, dtype=np.float32)


def embedding_dim() -> int:
    """Return the dimension of the configured embedding model."""
    return EMBEDDING_DIM