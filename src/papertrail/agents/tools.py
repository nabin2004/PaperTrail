"""
Research tools for PaperTrail agents.

Provides typed, documented tool functions for PydanticAI agents,
as well as backward-compatible LangChain BaseTool wrappers.
"""
from __future__ import annotations

from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def search_local_papers(query: str, top_k: int = 5) -> str:
    """
    Search locally indexed papers using semantic vector search and Cross-Encoder reranking.

    Use this when answering questions about concepts, architectures, or empirical findings
    present in the user's indexed library.

    Args:
        query: The semantic search query (e.g. 'transformer attention mechanism').
        top_k: Number of relevant excerpts to return (default 5).
    """
    from papertrail.retrieval.reranker import rerank
    from papertrail.retrieval.retrievers import PaperRetriever

    retriever = PaperRetriever()
    if retriever.is_empty():
        return "No papers indexed locally yet. You can use search_arxiv and download_and_index_paper to add papers."

    results = retriever.retrieve(query, k=top_k * 3)
    reranked = rerank(query, results, top_k=top_k)

    if not reranked:
        return f"No relevant chunks found in the local index for '{query}'."

    lines = []
    for i, r in enumerate(reranked, 1):
        authors_str = ", ".join(r.authors[:3]) + (" et al." if r.authors and len(r.authors) > 3 else "")
        date_str = f"({r.published})" if r.published else ""
        header = f"[{i}] {r.title or r.paper_id} – {authors_str} {date_str} (score: {r.score:.3f})"
        lines.append(f"{header}\n{r.text.strip()}\n")

    return "\n".join(lines)


def search_arxiv(
    query: str,
    limit: int = 5,
    categories: Optional[str] = None,
    sort: str = "relevance",
) -> str:
    """
    Search the online arXiv repository for papers matching a topic or keyword query.

    Use this when the local index does not have enough information, or when looking for
    the latest state-of-the-art literature.

    Args:
        query: Search keywords or research question (e.g. 'state space models vision').
        limit: Maximum number of papers to retrieve (default 5).
        categories: Optional comma-separated categories (e.g. 'cs.AI,cs.LG,cs.CV').
        sort: Sort order ('relevance', 'submittedDate', or 'lastUpdatedDate').
    """
    from papertrail.ingestion.arxiv_client import ArxivClient
    from papertrail.retrieval.vectorstore import VectorStore

    cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else []
    client = ArxivClient(query=query, categories=cats, max_results=limit, sort_by=sort)

    try:
        papers = client.fetch_papers()
    except Exception as exc:
        return f"Error querying arXiv API: {exc}"

    if not papers:
        return f"No papers found on arXiv for query '{query}'."

    store = VectorStore()
    indexed_ids = set(store.indexed_paper_ids())

    lines = [f"Found {len(papers)} papers on arXiv for '{query}':\n"]
    for i, p in enumerate(papers, 1):
        status = "[Already Indexed]" if p.arxiv_id in indexed_ids else "[Not Indexed]"
        authors_str = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        date_str = str(p.published.date()) if hasattr(p.published, "date") else str(p.published)[:10]
        abstract_snippet = (p.abstract[:200] + "...") if len(p.abstract) > 200 else p.abstract
        lines.append(
            f"{i}. {status} **{p.title}**\n"
            f"   arXiv ID: `{p.arxiv_id}` | Authors: {authors_str} | Date: {date_str} | Primary Cat: {p.primary_category}\n"
            f"   Abstract: {abstract_snippet}\n"
        )
    return "\n".join(lines)


def download_and_index_paper(arxiv_id: str) -> str:
    """
    Download a paper from arXiv by its ID, extract text, split into chunks, compute embeddings,
    and save it into the local FAISS vector store and metadata catalog.

    Use this after discovering relevant papers with `search_arxiv` that you want to ingest.

    Args:
        arxiv_id: The arXiv paper identifier (e.g. '1706.03762' or '2206.03003v2').
    """
    from papertrail.ingestion.arxiv_client import ArxivClient
    from papertrail.ingestion.pipeline import index_single_paper
    from papertrail.retrieval.vectorstore import VectorStore

    clean_id = arxiv_id.strip()
    client = ArxivClient(query=f"id:{clean_id}", max_results=1)

    try:
        papers = client.fetch_papers()
    except Exception as exc:
        return f"Failed to fetch metadata for arXiv ID '{clean_id}': {exc}"

    if not papers:
        return f"Could not find arXiv paper with ID '{clean_id}'."

    paper = papers[0]
    store = VectorStore()
    success, msg, n_chunks = index_single_paper(paper, store=store, clean=True)

    if success:
        return (
            f"Successfully downloaded and indexed '{paper.title}' (`{paper.arxiv_id}`). "
            f"Created {n_chunks} text chunks in local vector store."
        )
    else:
        return f"Could not index paper `{paper.arxiv_id}`: {msg}"


def list_indexed_papers() -> str:
    """
    List all papers currently stored and indexed in the local knowledge base.

    Returns the count, arXiv IDs, titles, primary categories, and dates.
    """
    from papertrail.ingestion.metadata import load_all_metadata

    papers = load_all_metadata()
    if not papers:
        return "No papers are currently indexed in the local store."

    lines = [f"Total indexed papers: {len(papers)}\n"]
    for i, p in enumerate(papers, 1):
        authors_str = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        date_str = str(p.published.date()) if hasattr(p.published, "date") else str(p.published)[:10]
        lines.append(f"{i}. [{p.arxiv_id}] **{p.title}** ({authors_str}, {date_str}) [{p.primary_category}]")

    return "\n".join(lines)


def get_trending_topics(top_n: int = 15) -> str:
    """
    Extract top trending keywords and category distributions across all locally indexed papers.

    Use this to identify recurring research themes, paradigms, or popular subjects in the corpus.
    """
    from papertrail.ingestion.metadata import load_all_metadata
    from papertrail.memory.trend_memory import TrendMemory

    papers = load_all_metadata()
    if not papers:
        return "No papers indexed yet to compute trends."

    mem = TrendMemory()
    mem.update(papers)
    return mem.summary()


def read_paper_abstract(paper_id: str) -> str:
    """
    Retrieve the full abstract and metadata for a paper by its arXiv ID or exact title.

    Args:
        paper_id: The arXiv ID (e.g. '1706.03762') or paper ID.
    """
    from papertrail.ingestion.metadata import load_metadata

    clean_id = paper_id.strip()
    paper = load_metadata(clean_id)

    if not paper:
        # Fallback search arXiv
        from papertrail.ingestion.arxiv_client import ArxivClient
        client = ArxivClient(query=f"id:{clean_id}", max_results=1)
        try:
            papers = client.fetch_papers()
            if papers:
                paper = papers[0]
        except Exception:
            pass

    if not paper:
        return f"Paper '{clean_id}' not found locally or on arXiv."

    authors_str = ", ".join(paper.authors)
    date_str = str(paper.published.date()) if hasattr(paper.published, "date") else str(paper.published)
    return (
        f"**Title:** {paper.title}\n"
        f"**arXiv ID:** {paper.arxiv_id}\n"
        f"**Authors:** {authors_str}\n"
        f"**Published:** {date_str}\n"
        f"**Categories:** {', '.join(paper.categories)}\n\n"
        f"**Abstract:**\n{paper.abstract}"
    )


def get_pydantic_ai_tools() -> list:
    """Return all core tool functions for PydanticAI agents."""
    return [
        search_local_papers,
        search_arxiv,
        download_and_index_paper,
        list_indexed_papers,
        get_trending_topics,
        read_paper_abstract,
    ]


# ──────────────────────────────────────────────────────────────
# Backward-Compatible LangChain BaseTool Classes
# ──────────────────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(..., description="The semantic search query for papers.")
    top_k: int = Field(default=5, description="Number of results to return.")


class SearchPapersTool(BaseTool):
    """Search the indexed paper corpus for relevant chunks."""
    name: str = "search_papers"
    description: str = (
        "Search the locally indexed arXiv papers for chunks of text relevant to a query. "
        "Use this to find evidence for a research question."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, top_k: int = 5) -> str:  # type: ignore[override]
        return search_local_papers(query=query, top_k=top_k)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async not supported")


class GetTrendsTool(BaseTool):
    """Return current trend keywords from indexed papers."""
    name: str = "get_trends"
    description: str = (
        "Returns the top trending keywords and arXiv categories from all indexed papers."
    )

    def _run(self, _: str = "") -> str:  # type: ignore[override]
        return get_trending_topics()

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async not supported")


class ListPapersTool(BaseTool):
    """List all indexed papers."""
    name: str = "list_papers"
    description: str = "List all papers currently indexed in PaperTrail."

    def _run(self, _: str = "") -> str:  # type: ignore[override]
        return list_indexed_papers()

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async not supported")


def get_all_tools() -> list:
    """Return LangChain-compatible tools list."""
    return [SearchPapersTool(), GetTrendsTool(), ListPapersTool()]
