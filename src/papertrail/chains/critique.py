"""
Critique chain – identifies weaknesses in a synthesis and suggests improvements.
"""
from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from papertrail.schemas.schema import SearchResult
from papertrail.utils.llm import get_llm

_CRITIQUE_PROMPT = """\
You are a critical peer reviewer for AI research papers.

Research question: {question}

Synthesis to critique:
{synthesis}

Available source excerpts:
{context}

Provide a concise critique covering:
1. Claims in the synthesis NOT supported by the source excerpts (hallucinations).
2. Important aspects of the question left unanswered.
3. Any logical inconsistencies or over-generalisations.
4. One concrete suggestion to improve the synthesis.

Be specific and constructive. Keep the critique under 200 words."""


def critique(synthesis: str, question: str, results: List[SearchResult]) -> str:
    """
    Critique *synthesis* against *results*.

    Falls back to a simple heuristic check when no LLM is available.
    """
    if not synthesis or not results:
        return "Nothing to critique."

    context = _format_context(results)

    try:
        llm = get_llm(temperature=0.2)
        prompt = ChatPromptTemplate.from_template(_CRITIQUE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(
            {"question": question, "synthesis": synthesis, "context": context}
        )
    except ValueError:
        # Heuristic fallback: count sentences and keywords
        word_count = len(synthesis.split())
        n_sources = len(results)
        return (
            f"[LLM not configured – heuristic critique]\n"
            f"Synthesis has {word_count} words and draws on {n_sources} source chunks.\n"
            f"Manual review recommended to verify factual accuracy."
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _format_context(results: List[SearchResult]) -> str:
    return "\n\n".join(
        f"[Source {i}] {r.title or r.paper_id}: {r.text[:300]}..."
        for i, r in enumerate(results, 1)
    )
