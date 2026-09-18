"""
ResearchAssistant – Senior AI Research Assistant powered by PydanticAI.

Positions between the junior Research Intern (search, discovery, indexing)
and the Principal Scientist (hypothesis formulation, experimental design).
Responsible for comparative literature reviews, gap analysis, and structured dossiers.
"""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from papertrail.agents.intern_agent import ResearchIntern
from papertrail.agents.tools import (
    search_local_papers,
    search_arxiv,
    download_and_index_paper,
    list_indexed_papers,
    get_trending_topics,
    read_paper_abstract,
)
from papertrail.schemas.schema import (
    ResearchDossier,
    GapAnalysis,
    PaperSummary,
    PaperComparisonDimension,
    BenchmarkResult,
)
from papertrail.utils.llm import get_pydantic_ai_model


ASSISTANT_SYSTEM_PROMPT = """\
You are the PaperTrail Senior Research Assistant — an analytical, rigorous AI researcher.

You operate directly between the junior Research Intern (who discovers and indexes raw literature) \
and the Principal AI Scientist (who formulates scientific hypotheses and experimental designs).

Your primary responsibilities are:
1. **Comparative Synthesis:** Analyze multiple papers side-by-side. Compare methodologies, benchmark performance, theoretical bounds, computational complexity, strengths, and limitations.
2. **Gap Analysis:** Critically evaluate the state-of-the-art to identify missing baselines, untested assumptions, dataset biases, scalability bottlenecks, and open scientific questions.
3. **Structured Research Dossiers:** Synthesize disorganized paper findings into coherent, structured literature reviews and dossiers ready for the Principal Scientist.
4. **Supervise Discovery:** When literature in the local repository or search results lacks necessary depth, formulate targeted search strategies or delegate to the Research Intern.

Guidelines:
- Ground all comparisons in empirical evidence from tools.
- Never make speculative claims without citing the relevant papers.
- Highlight contrasting viewpoints and contradictions across literature.
"""


def create_assistant_agent(model: Model | str | None = None) -> Agent[None, str]:
    """
    Factory to construct a PydanticAI Research Assistant Agent.
    """
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="research_assistant",
        instructions=ASSISTANT_SYSTEM_PROMPT,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
            list_indexed_papers,
            get_trending_topics,
            download_and_index_paper,
        ],
    )


def create_dossier_agent(model: Model | str | None = None) -> Agent[None, ResearchDossier]:
    """Factory to construct a structured Research Dossier generator using PydanticAI."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="research_dossier_generator",
        instructions=(
            ASSISTANT_SYSTEM_PROMPT
            + "\nYou must return a structured ResearchDossier containing executive summary, "
            "paper summaries, comparative dimensions, and gap analysis."
        ),
        output_type=ResearchDossier,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
            list_indexed_papers,
        ],
    )


def create_gap_agent(model: Model | str | None = None) -> Agent[None, GapAnalysis]:
    """Factory to construct a structured Gap Analysis generator using PydanticAI."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="gap_analysis_generator",
        instructions=(
            ASSISTANT_SYSTEM_PROMPT
            + "\nYou must return a structured GapAnalysis identifying deficiencies, "
            "untested assumptions, open questions, and promising future directions."
        ),
        output_type=GapAnalysis,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
        ],
    )


def create_benchmark_agent(model: Model | str | None = None) -> Agent[None, BenchmarkResult]:
    """Factory to construct a structured Benchmark Result generator using PydanticAI."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="benchmark_generator",
        instructions=(
            ASSISTANT_SYSTEM_PROMPT
            + "\nYou must return a structured BenchmarkResult detailing task type, "
            "primary metrics, baseline model comparisons, and compute requirements."
        ),
        output_type=BenchmarkResult,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
        ],
    )


class ResearchAssistant:
    """
    High-level interface for the PaperTrail Senior Research Assistant.

    Provides comparative paper analysis, literature gap identification,
    and structured research dossier preparation.
    """

    def __init__(self, model: Model | str | None = None) -> None:
        self.model = model
        self.agent = create_assistant_agent(model=model)
        self._dossier_agent: Optional[Agent[None, ResearchDossier]] = None
        self._gap_agent: Optional[Agent[None, GapAnalysis]] = None
        self._benchmark_agent: Optional[Agent[None, BenchmarkResult]] = None
        self._intern: Optional[ResearchIntern] = None

    @property
    def dossier_agent(self) -> Agent[None, ResearchDossier]:
        if self._dossier_agent is None:
            self._dossier_agent = create_dossier_agent(model=self.model)
        return self._dossier_agent

    @property
    def gap_agent(self) -> Agent[None, GapAnalysis]:
        if self._gap_agent is None:
            self._gap_agent = create_gap_agent(model=self.model)
        return self._gap_agent

    @property
    def benchmark_agent(self) -> Agent[None, BenchmarkResult]:
        if self._benchmark_agent is None:
            self._benchmark_agent = create_benchmark_agent(model=self.model)
        return self._benchmark_agent

    @property
    def intern(self) -> ResearchIntern:
        if self._intern is None:
            self._intern = ResearchIntern(model=self.model)
        return self._intern

    def run_sync(self, prompt: str) -> str:
        """Run the research assistant synchronously and return markdown text."""
        result = self.agent.run_sync(prompt)
        return str(result.output)

    async def run(self, prompt: str) -> str:
        """Run the research assistant asynchronously and return markdown text."""
        result = await self.agent.run(prompt)
        return str(result.output)

    def compare_papers(self, topic_or_papers: str) -> str:
        """
        Conduct an in-depth comparative analysis across papers for a given topic or paper set.
        """
        prompt = (
            f"Conduct a rigorous comparative analysis for: '{topic_or_papers}'.\n"
            f"1. Search local literature and arXiv to identify the most important competing methods.\n"
            f"2. Build a comparative breakdown: Methodology, Benchmark results, Computational efficiency, Strengths & Limitations.\n"
            f"3. Highlight direct trade-offs between approaches."
        )
        return self.run_sync(prompt)

    def analyze_gaps(self, topic: str) -> GapAnalysis:
        """
        Analyze current research literature and generate a structured GapAnalysis model.
        """
        prompt = (
            f"Analyze the research landscape for '{topic}'. Identify critical gaps, untested assumptions, "
            f"unresolved scientific questions, and the most promising directions."
        )
        result = self.gap_agent.run_sync(prompt)
        return result.output

    def prepare_dossier(self, topic: str) -> ResearchDossier:
        """
        Compile a full structured ResearchDossier for the Principal Scientist.
        """
        prompt = (
            f"Compile a comprehensive scientific research dossier on '{topic}'.\n"
            f"Gather literature, extract methodology and benchmark details, perform dimensional comparisons, "
            f"and identify research gaps."
        )
        result = self.dossier_agent.run_sync(prompt)
        return result.output

    def run_benchmarks(self, topic_or_models: str) -> BenchmarkResult:
        """
        Compile empirical benchmark evaluations and comparative model scores for a topic or set of models.
        """
        prompt = (
            f"Extract empirical benchmark evaluation results for: '{topic_or_models}'.\n"
            f"Search relevant papers to identify standardized benchmarks (e.g. GLUE, MMLU, LongBench, ImageNet),\n"
            f"quantitative metric values across models, baseline gains, and compute requirements."
        )
        result = self.benchmark_agent.run_sync(prompt)
        return result.output

    def plan_implementation(self, method_or_model: str) -> str:
        """
        Synthesize detailed implementation specs for a proposed model or method:
        architecture components, layer configurations, training objectives, and hardware footprint.
        """
        prompt = (
            f"Implementation Blueprint: For '{method_or_model}', generate a concrete engineering plan:\n"
            f"1. Model architecture: Layers, normalization, activation functions, state dimensions.\n"
            f"2. Training configuration: Loss functions, optimizers, learning rate schedule, context length.\n"
            f"3. Infrastructure & hardware requirements: VRAM, precision (FP16/BF16), compute estimates."
        )
        return self.run_sync(prompt)

    def delegate_to_intern(self, request: str) -> str:
        """
        Delegate a literature search or indexing request to the junior Research Intern.
        """
        return self.intern.run_sync(request)
