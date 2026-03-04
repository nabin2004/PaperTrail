"""
PaperTrail unit tests.

Run with:  pytest tests/
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────

def test_paper_schema():
    from papertrail.schemas.schema import Paper
    p = Paper(
        arxiv_id="1234.5678",
        title="Test Paper",
        abstract="Abstract text here.",
        authors=["Alice", "Bob"],
        primary_category="cs.LG",
        categories=["cs.LG", "cs.AI"],
        published="2024-01-01T00:00:00",
        updated="2024-01-01T00:00:00",
    )
    assert p.arxiv_id == "1234.5678"
    assert p.indexed is False


def test_search_result_schema():
    from papertrail.schemas.schema import SearchResult
    r = SearchResult(paper_id="1234.5678", chunk_id=0, text="some text", score=0.9)
    assert r.score == 0.9


# ──────────────────────────────────────────────────────────────
# Cleaners
# ──────────────────────────────────────────────────────────────

def test_clean_text_unicode():
    from papertrail.processing.cleaners import clean_text
    result = clean_text("Hello\u2013World \u201cquoted\u201d")
    assert "-" in result
    assert '"' in result


def test_clean_text_whitespace():
    from papertrail.processing.cleaners import clean_text
    result = clean_text("line1\n\n\n\nline2")
    assert "\n\n\n" not in result


def test_clean_file(tmp_path):
    from papertrail.processing.cleaners import clean_file
    f = tmp_path / "test.txt"
    f.write_text("Hello  World\u2013test", encoding="utf-8")
    result = clean_file(f)
    assert "Hello" in result


# ──────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────

def test_embed_texts_shape():
    from papertrail.processing.embeddings import embed_texts
    vecs = embed_texts(["hello world", "attention mechanism"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32


def test_embed_query_shape():
    from papertrail.processing.embeddings import embed_query
    vec = embed_query("test query")
    assert vec.shape == (384,)


def test_embed_texts_empty():
    from papertrail.processing.embeddings import embed_texts
    vecs = embed_texts([])
    assert vecs.shape[0] == 0


# ──────────────────────────────────────────────────────────────
# VectorStore
# ──────────────────────────────────────────────────────────────

def test_vectorstore_add_and_search(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Need to re-init to pick up the env var
    import importlib
    import papertrail.retrieval.vectorstore as vs_mod
    importlib.reload(vs_mod)

    from papertrail.processing.embeddings import embed_texts, embed_query
    store = vs_mod.VectorStore(index_name="test_idx", dim=384)

    texts = ["attention mechanism", "diffusion model", "reinforcement learning"]
    vecs = embed_texts(texts)
    chunks = [{"paper_id": f"p{i}", "chunk_id": 0, "text": t, "source": ""} for i, t in enumerate(texts)]
    store.add_chunks(vecs, chunks)

    assert store.total_chunks == 3

    q = embed_query("neural attention")
    results = store.search(q, k=2)
    assert len(results) == 2
    assert results[0][0] >= results[1][0]   # sorted by score desc


def test_vectorstore_empty_search(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import importlib
    import papertrail.retrieval.vectorstore as vs_mod
    importlib.reload(vs_mod)

    store = vs_mod.VectorStore(index_name="empty_idx", dim=384)
    from papertrail.processing.embeddings import embed_query
    results = store.search(embed_query("test"), k=5)
    assert results == []


# ──────────────────────────────────────────────────────────────
# Splitters
# ──────────────────────────────────────────────────────────────

def test_split_and_save(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import importlib
    import papertrail.processing.splitters as spl
    importlib.reload(spl)

    txt = tmp_path / "test_paper.txt"
    content = "This is a test paper. " * 200           # ~4400 chars
    txt.write_text(content, encoding="utf-8")

    out = spl.split_and_save(txt)
    assert out.exists()
    chunks = spl.load_chunks("test_paper")
    assert len(chunks) > 1
    assert all("text" in c for c in chunks)
    assert all("paper_id" in c for c in chunks)


# ──────────────────────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────────────────────

def test_rerank_ordering():
    from papertrail.retrieval.reranker import rerank
    from papertrail.schemas.schema import SearchResult

    results = [
        SearchResult(paper_id="p0", chunk_id=0, text="diffusion model generation", score=0.0),
        SearchResult(paper_id="p1", chunk_id=0, text="attention mechanism transformer", score=0.0),
        SearchResult(paper_id="p2", chunk_id=0, text="reinforcement learning reward", score=0.0),
    ]
    reranked = rerank("attention transformer neural", results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].text == "attention mechanism transformer"


def test_rerank_empty():
    from papertrail.retrieval.reranker import rerank
    assert rerank("query", [], top_k=5) == []


# ──────────────────────────────────────────────────────────────
# Memory
# ──────────────────────────────────────────────────────────────

def test_paper_memory():
    from papertrail.memory.paper_memory import PaperMemory
    from papertrail.schemas.schema import Paper

    mem = PaperMemory()
    p = Paper(
        arxiv_id="test.001",
        title="Attention Is All You Need",
        abstract="Transformer paper.",
        authors=["Vaswani"],
        primary_category="cs.CL",
        categories=["cs.CL"],
        published="2017-06-12",
        updated="2017-06-12",
    )
    mem.add(p)
    assert len(mem) == 1
    assert mem.get("test.001") is not None
    assert mem.get("nonexistent") is None
    found = mem.search_by_author("Vaswani")
    assert len(found) == 1


def test_trend_memory():
    from papertrail.memory.trend_memory import TrendMemory
    from papertrail.schemas.schema import Paper

    papers = [
        Paper(
            arxiv_id=f"t.00{i}",
            title=f"Transformer attention paper {i}",
            abstract="We propose a new attention mechanism for transformers.",
            authors=[f"Author{i}"],
            primary_category="cs.LG",
            categories=["cs.LG"],
            published="2024-01-01",
            updated="2024-01-01",
        )
        for i in range(3)
    ]
    mem = TrendMemory()
    mem.update(papers)
    assert mem.paper_count == 3
    keywords = dict(mem.top_keywords(50))
    assert "transformer" in keywords or "attention" in keywords


# ──────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────

def test_metadata_save_load(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import importlib
    import papertrail.ingestion.metadata as meta
    importlib.reload(meta)

    from papertrail.schemas.schema import Paper
    p = Paper(
        arxiv_id="meta.001",
        title="Metadata Test Paper",
        abstract="Testing metadata persistence.",
        authors=["Test Author"],
        primary_category="cs.AI",
        categories=["cs.AI"],
        published="2024-01-01",
        updated="2024-01-01",
    )
    meta.save_metadata(p)
    loaded = meta.load_metadata("meta.001")
    assert loaded is not None
    assert loaded.title == "Metadata Test Paper"
    assert meta.metadata_exists("meta.001")
    assert not meta.metadata_exists("nonexistent")


# ──────────────────────────────────────────────────────────────
# Evaluation (heuristic only – no LLM needed)
# ──────────────────────────────────────────────────────────────

def test_faithfulness_no_llm():
    from papertrail.evaluation.faithfulness import score_faithfulness
    from papertrail.schemas.schema import SearchResult

    synthesis = "The attention mechanism improves transformer performance."
    sources = [
        SearchResult(paper_id="p0", chunk_id=0, text="Attention mechanisms are key to transformer models.", score=0.9),
    ]
    score = score_faithfulness(synthesis, sources)
    assert 0.0 <= score <= 1.0


def test_coverage_no_llm():
    from papertrail.evaluation.coverage import score_coverage
    from papertrail.schemas.schema import SearchResult

    question = "How do transformers use attention?"
    results = [
        SearchResult(paper_id="p0", chunk_id=0, text="Transformers rely on self-attention layers.", score=0.9),
    ]
    score = score_coverage(question, results)
    assert 0.0 <= score <= 1.0
    assert score > 0.3   # "transformers" and "attention" are present
