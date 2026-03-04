"""
Planning chain – given a research question, produce a structured search plan.
"""
from __future__ import annotations

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from papertrail.schemas.schema import ResearchPlan
from papertrail.utils.llm import get_llm

_PLANNING_PROMPT = """\
You are a research planning assistant.  Given a research question, create a concise structured search plan.

Research question: {question}

Respond with ONLY valid JSON (no markdown, no commentary) matching this schema:
{{
  "concepts":    ["<key concept>", ...],
  "queries":     ["<search query 1>", "<search query 2>", ...],
  "categories":  ["<arXiv category e.g. cs.LG>", ...],
  "paper_types": ["survey", "empirical", "theoretical", ...]
}}

Include 3-5 queries, 1-4 arXiv categories, and 2-4 paper types."""


def plan_research(question: str) -> ResearchPlan:
    """
    Generate a ResearchPlan for *question*.

    Falls back to a simple keyword plan if the LLM is not configured.
    """
    try:
        llm = get_llm(temperature=0.2)
        prompt = ChatPromptTemplate.from_template(_PLANNING_PROMPT)
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"question": question})
        data = _parse_json(raw)
        return ResearchPlan(question=question, raw=raw, **data)
    except ValueError:
        # No LLM configured – build a minimal plan from the question words
        words = [w for w in question.lower().split() if len(w) > 3]
        return ResearchPlan(
            question=question,
            concepts=words[:5],
            queries=[question],
            categories=["cs.AI", "cs.LG"],
            paper_types=["survey", "empirical"],
            raw="(LLM not configured – using keyword fallback)",
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Extract the first JSON object from *text*."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"concepts": [], "queries": [text], "categories": [], "paper_types": []}
