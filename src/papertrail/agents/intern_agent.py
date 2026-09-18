"""
Research Intern Agent – powered by PydanticAI.

An autonomous research assistant that interacts with the PaperTrail
toolkit: searching indexed papers, discovering papers on arXiv,
indexing new literature, exploring trends, and synthesizing evidence.
"""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from papertrail.agents.tools import (
    search_local_papers,
    search_arxiv,
    download_and_index_paper,
    list_indexed_papers,
    get_trending_topics,
    read_paper_abstract,
)
from papertrail.schemas.schema import ResearchReport, SearchResult, ResearchPlan
from papertrail.utils.llm import get_pydantic_ai_model


INTERN_SYSTEM_PROMPT = """\
You are the PaperTrail Research Intern — an autonomous, diligent, and scientifically rigorous AI research assistant.

Your mission is to help researchers discover, analyze, synthesize, and organize academic literature.

You have access to a suite of specialized tools:
1. `search_local_papers(query, top_k)`: Search the user's locally indexed papers with vector similarity and reranking.
2. `search_arxiv(query, limit, categories, sort)`: Search arXiv for the latest papers and literature.
3. `download_and_index_paper(arxiv_id)`: Download a PDF from arXiv, parse text, chunk, embed, and index it into the local store.
4. `list_indexed_papers()`: View what papers are currently in the user's local corpus.
5. `get_trending_topics(top_n)`: Inspect trending keywords and research topics across indexed papers.
6. `read_paper_abstract(paper_id)`: Read the abstract and detailed metadata for a specific paper.

Guidelines for your workflow:
- **Be thorough & grounded:** When asked about a research topic, check local literature first. If local results are sparse or outdated, search arXiv.
- **Proactive & helpful:** If finding an important new paper on arXiv that directly answers the user's inquiry, download and index it to expand the user's research database.
- **Cite accurately:** Attribute claims to authors, years, and paper IDs (e.g. `Vaswani et al. (2017) [arxiv:1706.03762]`).
- **Structured output:** Present evidence clearly with key takeaways, methodology comparisons, trade-offs, and open questions.
"""


def create_intern_agent(model: Model | str | None = None) -> Agent[None, str]:
    """
    Factory to construct a PydanticAI Research Intern Agent.

    If model is None, it resolves from the environment (OpenAI/Groq/Ollama).
    """
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            # Fallback for environments without API keys (e.g. tests or raw use)
            resolved_model = "test"
    else:
        resolved_model = model

    agent = Agent(
        model=resolved_model,
        name="research_intern",
        instructions=INTERN_SYSTEM_PROMPT,
        tools=[
            search_local_papers,
            search_arxiv,
            download_and_index_paper,
            list_indexed_papers,
            get_trending_topics,
            read_paper_abstract,
        ],
    )
    return agent


class ResearchIntern:
    """
    High-level interface for the PaperTrail Research Intern.

    Provides synchronous and asynchronous invocation, interactive chat,
    and structured research synthesis.
    """

    def __init__(self, model: Model | str | None = None) -> None:
        self.model = model
        self.agent = create_intern_agent(model=model)

    def run_sync(self, prompt: str) -> str:
        """Run the research intern synchronously and return response text."""
        result = self.agent.run_sync(prompt)
        return str(result.output)

    async def run(self, prompt: str) -> str:
        """Run the research intern asynchronously and return response text."""
        result = await self.agent.run(prompt)
        return str(result.output)

    def investigate(self, topic: str) -> str:
        """
        Run an end-to-end literature investigation on *topic*.
        The intern will search local literature, check arXiv if needed,
        and synthesize a comprehensive brief.
        """
        prompt = (
            f"Conduct a thorough research investigation on: '{topic}'.\n"
            f"1. Check if we have relevant papers indexed locally.\n"
            f"2. Search arXiv for relevant or recent papers on this topic.\n"
            f"3. Highlight consensus findings, technical approaches, and open challenges.\n"
            f"4. Provide citations for all cited works."
        )
        return self.run_sync(prompt)

    def collect_data(self, topic: str, limit: int = 5) -> str:
        """
        Search and collect papers on a topic, downloading and indexing them into the repository.
        """
        prompt = (
            f"Data Collection Task: Search arXiv for up to {limit} key papers on '{topic}'.\n"
            f"Download and index each promising paper that is not already in our local store.\n"
            f"Summarize the newly collected papers with their paper IDs, titles, and key contributions."
        )
        return self.run_sync(prompt)

    def reproduce_study(self, paper_id_or_topic: str) -> str:
        """
        Evaluate reproduction feasibility, datasets, open-source codebases, and run setup.
        """
        prompt = (
            f"Reproduction Feasibility Check: Investigate '{paper_id_or_topic}'.\n"
            f"1. Retrieve the paper abstract and methodology details.\n"
            f"2. Identify required benchmarks, dataset splits, and hyperparameters.\n"
            f"3. Outline step-by-step reproduction instructions and potential pitfalls."
        )
        return self.run_sync(prompt)

    def run_baseline_experiment(self, topic: str) -> str:
        """
        Formulate a simple exploratory baseline run to establish initial performance benchmarks.
        """
        prompt = (
            f"Simple Baseline Experiment: For '{topic}', identify standard baseline models\n"
            f"and established evaluation metrics. Propose a quick-start pilot test protocol."
        )
        return self.run_sync(prompt)

