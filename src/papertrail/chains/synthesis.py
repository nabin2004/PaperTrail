"""
Synthesis chain – distils retrieved paper chunks into a cohesive answer.
"""
from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from papertrail.schemas.schema import SearchResult
from papertrail.utils.llm import get_llm

_SYNTHESIS_PROMPT = """\
You are a research synthesis expert.  Using ONLY the paper excerpts below, write a comprehensive answer to the research question.

Research question: {question}

--- Paper excerpts ---
{context}
--- end of excerpts ---

Your synthesis must:
1. Directly answer the question with specific evidence from the excerpts.
2. Cite papers inline as (Author et al., YEAR) or (arxiv:<id>) when available.
3. Highlight consensus findings, disagreements, and open questions.
4. Be factual – do NOT introduce information not present in the excerpts.
5. Be structured: use short paragraphs or bullet points for readability.

Synthesis:"""


def synthesize(question: str, results: List[SearchResult]) -> str:
    """
    Synthesize paper chunks into an answer for *question*.

    If no LLM is configured a plain text compilation is returned instead.
    """
    if not results:
        return "No relevant papers found for the given question."

    context = _format_context(results)

    try:
        llm = get_llm(temperature=0.3)
        prompt = ChatPromptTemplate.from_template(_SYNTHESIS_PROMPT)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})
    except ValueError as exc:
        # Graceful degradation: return a formatted compilation
        header = f"[LLM not configured – raw excerpts for: {question}]\n\n"
        return header + context


# ── helpers ──────────────────────────────────────────────────────────────────

def _format_context(results: List[SearchResult]) -> str:
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        title = r.title or r.paper_id
        authors = ", ".join(r.authors[:3]) + " et al." if r.authors else ""
        published = f"({r.published})" if r.published else ""
        citation = f"{authors} {published}".strip()
        lines.append(f"[{i}] {title} – {citation}\n{r.text}\n")
    return "\n".join(lines)
