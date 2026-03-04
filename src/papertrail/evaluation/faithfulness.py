"""
Faithfulness evaluation – estimates how grounded a synthesis is in its sources.

Score: 0.0 (hallucinated) → 1.0 (fully grounded)

MVP implementation uses token-overlap (no extra model required).
An LLM-based scorer is used when available for higher accuracy.
"""
from __future__ import annotations

import re
from typing import List

from papertrail.schemas.schema import SearchResult


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def score_faithfulness(synthesis: str, sources: List[SearchResult]) -> float:
    """
    Return a faithfulness score in [0, 1].

    Strategy
    --------
    1. Try LLM-based scoring (more accurate, needs API key).
    2. Fall back to token-overlap heuristic.
    """
    if not synthesis or not sources:
        return 0.0

    # Try LLM-based scoring
    try:
        return _llm_faithfulness(synthesis, sources)
    except Exception:
        pass

    # Heuristic fallback
    return _token_overlap_faithfulness(synthesis, sources)


# ──────────────────────────────────────────────────────────────
# LLM-based scorer
# ──────────────────────────────────────────────────────────────

def _llm_faithfulness(synthesis: str, sources: List[SearchResult]) -> float:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from papertrail.utils.llm import get_llm

    context = "\n\n".join(r.text[:400] for r in sources[:6])
    prompt_text = (
        "Rate how faithfully the synthesis below is grounded in the provided sources. "
        "Reply with a single decimal number between 0.0 and 1.0. Nothing else.\n\n"
        f"Sources:\n{context}\n\nSynthesis:\n{synthesis[:1000]}"
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

def _token_overlap_faithfulness(synthesis: str, sources: List[SearchResult]) -> float:
    """Fraction of synthesis bigrams that appear in at least one source."""
    synth_bigrams = _bigrams(synthesis.lower())
    if not synth_bigrams:
        return 0.0

    source_text = " ".join(r.text.lower() for r in sources)
    source_bigrams = _bigrams(source_text)

    overlap = synth_bigrams & source_bigrams
    return len(overlap) / len(synth_bigrams)


def _bigrams(text: str) -> set:
    tokens = re.findall(r"\b\w+\b", text)
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
