"""
LangChain-compatible tools for the research agent.

Each tool wraps a PaperTrail capability behind the langchain BaseTool interface.
The agent can call these tools when formulating a research answer.
"""
from __future__ import annotations

from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Input schemas
# ──────────────────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(..., description="The semantic search query for papers.")
    top_k: int = Field(default=5, description="Number of results to return.")


class SynthesisInput(BaseModel):
    question: str = Field(..., description="The research question to answer.")
    top_k: int = Field(default=5, description="How many paper chunks to use.")


# ──────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────

class SearchPapersTool(BaseTool):
    """Search the indexed paper corpus for relevant chunks."""

    name: str = "search_papers"
    description: str = (
        "Search the locally indexed arXiv papers for chunks of text relevant to a query. "
        "Use this to find evidence for a research question."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, top_k: int = 5) -> str:  # type: ignore[override]
        from papertrail.retrieval.retrievers import PaperRetriever
        from papertrail.retrieval.reranker import rerank

        retriever = PaperRetriever()
        if retriever.is_empty():
            return "No papers indexed yet. Run `papertrail ingest` first."
        results = retriever.retrieve(query, k=top_k * 2)
        reranked = rerank(query, results, top_k=top_k)
        lines = []
        for i, r in enumerate(reranked, 1):
            lines.append(f"[{i}] {r.title or r.paper_id} (score={r.score:.3f})\n{r.text[:300]}\n")
        return "\n".join(lines) if lines else "No relevant papers found."

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async not supported")


class GetTrendsTool(BaseTool):
    """Return current trend keywords from indexed papers."""

    name: str = "get_trends"
    description: str = (
        "Returns the top trending keywords and arXiv categories from all indexed papers."
    )

    def _run(self, _: str = "") -> str:  # type: ignore[override]
        from papertrail.ingestion.metadata import load_all_metadata
        from papertrail.memory.trend_memory import TrendMemory

        papers = load_all_metadata()
        if not papers:
            return "No papers indexed yet."
        mem = TrendMemory()
        mem.update(papers)
        return mem.summary()

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async not supported")


class ListPapersTool(BaseTool):
    """List all indexed papers."""

    name: str = "list_papers"
    description: str = "List all papers currently indexed in PaperTrail."

    def _run(self, _: str = "") -> str:  # type: ignore[override]
        from papertrail.ingestion.metadata import load_all_metadata

        papers = load_all_metadata()
        if not papers:
            return "No papers indexed yet."
        lines = [
            f"- {p.arxiv_id}: {p.title} ({', '.join(p.authors[:2])}{'...' if len(p.authors) > 2 else ''})"
            for p in papers
        ]
        return f"{len(papers)} papers indexed:\n" + "\n".join(lines)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async not supported")


def get_all_tools() -> list:
    return [SearchPapersTool(), GetTrendsTool(), ListPapersTool()]
