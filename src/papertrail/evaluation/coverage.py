"""
Coverage evaluation – how well the retrieved chunks cover the research question.

Score: 0.0 (poor coverage) → 1.0 (full coverage)
"""
from __future__ import annotations

import re
from typing import List

from papertrail.schemas.schema import SearchResult


def score_coverage(question: str, results: List[SearchResult]) -> float:
    """
    Return a coverage score in [0, 1].

    Uses keyword overlap between the question and the retrieved chunks.
    """
    if not question or not results:
        return 0.0

    # Try LLM-based scoring first
    try:
        return _llm_coverage(question, results)
    except Exception:
        pass

    return _keyword_coverage(question, results)


# ──────────────────────────────────────────────────────────────
# LLM-based scorer
# ──────────────────────────────────────────────────────────────

def _llm_coverage(question: str, results: List[SearchResult]) -> float:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from papertrail.utils.llm import get_llm

    context = "\n\n".join(r.text[:300] for r in results[:8])
    prompt_text = (
        "Rate how well the provided paper excerpts cover the research question. "
        "Reply with a single decimal between 0.0 (no coverage) and 1.0 (full coverage). Nothing else.\n\n"
        f"Research question: {question}\n\nExcerpts:\n{context}"
    )
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_template("{text}")
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"text": prompt_text}).strip()
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if match:
        score = float(match.group())
        return min(max(score, 0.0), 1.0)
    return 0.5


# ──────────────────────────────────────────────────────────────
# Heuristic fallback
# ──────────────────────────────────────────────────────────────

def _keyword_coverage(question: str, results: List[SearchResult]) -> float:
    """Fraction of question keywords found in retrieved chunks."""
    _STOP = {"the", "a", "an", "and", "or", "of", "in", "is", "are", "how", "what",
             "why", "when", "which", "who", "do", "does", "with", "for", "to"}

    q_words = {w for w in re.findall(r"\b\w{3,}\b", question.lower()) if w not in _STOP}
    if not q_words:
        return 1.0

    combined = " ".join(r.text.lower() for r in results)
    found = {w for w in q_words if w in combined}
    return len(found) / len(q_words)
