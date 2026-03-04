"""
PaperTrail – entry point.

Run via the CLI:
    papertrail --help

Or run the demo pipeline directly:
    python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure the src layout is importable when running `python main.py`
sys.path.insert(0, str(Path(__file__).parent / "src"))


def demo() -> None:
    """
    Quick end-to-end demo:
      1. Fetch 3 arXiv papers
      2. Process and index them
      3. Search for 'attention mechanism'
    """
    from papertrail.ingestion.arxiv_client import ArxivClient
    from papertrail.ingestion.metadata import save_metadata
    from papertrail.ingestion.pdf_loader import download_pdf, save_processed_text
    from papertrail.processing.embeddings import embed_texts
    from papertrail.processing.splitters import split_and_save
    from papertrail.retrieval.retrievers import PaperRetriever
    from papertrail.retrieval.vectorstore import VectorStore
    import json

    print("=" * 60)
    print("PaperTrail Demo")
    print("=" * 60)

    # ── Ingest ────────────────────────────────────────────────
    client = ArxivClient(categories=["cs.LG"], max_results=3)
    print("\n[1/3] Fetching papers from arXiv…")
    papers = client.fetch_papers()
    print(f"  Found {len(papers)} papers.")

    store = VectorStore()

    for paper in papers:
        print(f"\n  → {paper.title[:70]}")
        if paper.arxiv_id in store.indexed_paper_ids():
            print("    already indexed.")
            continue
        try:
            pdf = download_pdf(paper)
            if pdf is None:
                print("    no PDF URL – skip.")
                continue
            txt = save_processed_text(pdf, clean=True)
            chunks_path = split_and_save(txt)
            chunks = []
            with chunks_path.open() as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
            if not chunks:
                continue
            vecs = embed_texts([c["text"] for c in chunks])
            store.add_chunks(vecs, chunks)
            save_metadata(paper)
            print(f"    indexed {len(chunks)} chunks.")
        except Exception as e:
            print(f"    error: {e}")

    # ── Search ────────────────────────────────────────────────
    print("\n[2/3] Semantic search: 'attention mechanism'")
    retriever = PaperRetriever()
    if retriever.is_empty():
        print("  Nothing indexed yet.")
        return
    results = retriever.retrieve("attention mechanism", k=3)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.title or r.paper_id} (score={r.score:.3f})")
        print(f"      {r.text[:120]}…")

    print("\n[3/3] Done! Run `papertrail --help` for the full CLI.")


if __name__ == "__main__":
    demo()
