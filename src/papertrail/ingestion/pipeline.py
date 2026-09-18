"""
Ingestion pipeline – handles downloading, processing, splitting, embedding,
and indexing papers into the vector store.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import logfire
from papertrail.utils.observability import papers_indexed_counter
from papertrail.ingestion.metadata import save_metadata
from papertrail.ingestion.pdf_loader import download_pdf, save_processed_text
from papertrail.processing.embeddings import embed_texts
from papertrail.processing.splitters import split_and_save
from papertrail.retrieval.vectorstore import VectorStore
from papertrail.schemas.schema import Paper


def load_chunks_from_file(chunks_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL chunk dictionaries from disk."""
    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def index_single_paper(
    paper: Paper,
    store: Optional[VectorStore] = None,
    clean: bool = True,
    skip_if_indexed: bool = True,
) -> Tuple[bool, str, int]:
    """
    Downloads, extracts text, chunks, embeds, and indexes a single Paper.

    Parameters
    ----------
    paper : Paper
        The paper metadata object to ingest.
    store : Optional[VectorStore]
        The vector store to save chunks into (defaults to standard store).
    clean : bool
        Whether to clean extracted text before chunking.
    skip_if_indexed : bool
        Whether to skip if paper is already in the vector store.

    Returns
    -------
    Tuple[bool, str, int]
        (success: bool, status_message: str, num_chunks: int)
    """
    with logfire.span("ingestion.index_paper", arxiv_id=paper.arxiv_id, title=paper.title[:80]):
        if store is None:
            store = VectorStore()

        if skip_if_indexed and paper.arxiv_id in store.indexed_paper_ids():
            return False, "already indexed", 0

        pdf_path = download_pdf(paper)
        if pdf_path is None:
            return False, "no PDF available", 0

        text_path = save_processed_text(pdf_path, clean=clean)
        chunks_path = split_and_save(text_path)

        chunk_dicts = load_chunks_from_file(chunks_path)
        if not chunk_dicts:
            return False, "no text extracted", 0

        texts = [c["text"] for c in chunk_dicts]
        embeddings = embed_texts(texts)
        store.add_chunks(embeddings, chunk_dicts)

        paper.indexed = True
        save_metadata(paper)
        papers_indexed_counter.add(1, {"category": paper.primary_category or "cs.AI"})
        logfire.info(
            "Indexed paper {arxiv_id} ('{title}') with {count} chunks",
            arxiv_id=paper.arxiv_id,
            title=paper.title[:60],
            count=len(chunk_dicts),
        )
        return True, f"indexed ({len(chunk_dicts)} chunks)", len(chunk_dicts)

