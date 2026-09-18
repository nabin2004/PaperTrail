"""
LeadResearcher – Senior Research Lead / Principal Investigator powered by PydanticAI.

The primary point of contact for the user. Steers research direction, validates
methodology, conducts peer-review quality assessments, coordinates cross-domain
synergies, and delegates tasks down to Researcher, Research Assistant, and Intern.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from papertrail.agents.intern_agent import ResearchIntern
from papertrail.agents.assistant_agent import ResearchAssistant
from papertrail.agents.scientist_agent import ScientistAgent, ResearcherAgent
from papertrail.agents.tools import (
    search_local_papers,
    search_arxiv,
    read_paper_abstract,
    list_indexed_papers,
    get_trending_topics,
)
from papertrail.schemas.schema import (
    PaperReviewScorecard,
    ResearchRoadmap,
    ResearchDossier,
    HypothesisSpec,
    ExperimentSpec,
    BenchmarkResult,
    MatrixProject,
    CrossLabSynthesis,
)
from papertrail.utils.llm import get_pydantic_ai_model


LEAD_SYSTEM_PROMPT = """\
You are the PaperTrail Senior Research Lead & Principal Investigator (PI).

You are the PRIMARY INTERFACE for the user. Your role is to guide the overall research vision, \
ensure scientific rigor, assess paper quality, coordinate across domains, and orchestrate \
your research team:
- **Researcher:** Formulates novel hypotheses, designs controlled experiments, interprets empirical results.
- **Research Assistant:** Executes benchmarks, analyzes implementations, builds comparative trade-off matrices.
- **Research Intern:** Performs broad literature searches, downloads & indexes papers, verifies reproduction steps.

Your key directives:
1. **Strategic Leadership:** Define clear research directions, spot high-impact open problems, and connect ideas across domains.
2. **Quality & Rigor:** Demand rigorous baselines, fair benchmarking, statistical significance, and clear falsification criteria.
3. **Multi-Agent Orchestration:** Use your delegation tools to assign tasks to the right tier rather than guessing:
   - Need hypotheses or experimental design? -> Task the Researcher.
   - Need benchmarks, implementation specs, or comparative matrices? -> Task the Research Assistant.
   - Need paper search, PDF downloads, or reproduction checks? -> Task the Research Intern.
4. **Executive Synthesis:** Synthesize technical findings into clear, actionable, and scientifically sound conclusions for the user.
"""


def create_lead_agent(
    model: Model | str | None = None,
    researcher: Optional[ResearcherAgent] = None,
    assistant: Optional[ResearchAssistant] = None,
    intern: Optional[ResearchIntern] = None,
) -> Agent[None, str]:
    """Factory to construct the Senior Research Lead Agent with delegation tools."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    # Instantiations for delegation
    _researcher = researcher or ResearcherAgent(model=resolved_model)
    _assistant = assistant or ResearchAssistant(model=resolved_model)
    _intern = intern or ResearchIntern(model=resolved_model)

    agent = Agent(
        model=resolved_model,
        name="senior_research_lead",
        instructions=LEAD_SYSTEM_PROMPT,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
            get_trending_topics,
            list_indexed_papers,
        ],
    )

    # Delegation tools registered on the lead agent
    @agent.tool_plain
    def task_researcher(task: str) -> str:
        """
        Delegate to the Researcher to formulate testable hypotheses, design experimental protocols,
        or interpret empirical outcomes.
        """
        return _researcher.agent.run_sync(task).output

    @agent.tool_plain
    def task_research_assistant(task: str) -> str:
        """
        Delegate to the Research Assistant to compile comparative paper matrices,
        benchmark evaluations, or engineering implementation plans.
        """
        return _assistant.run_sync(task)

    @agent.tool_plain
    def task_research_intern(task: str) -> str:
        """
        Delegate to the Research Intern to search literature, collect & index papers,
        or check reproduction feasibility.
        """
        return _intern.run_sync(task)

    @agent.tool_plain
    def consult_specialized_lab(lab_code: str, task: str) -> str:
        """
        Consult one of the 10 specialized research laboratories:
        - Core Domains: LMI, AI, VI, RCI, SIKG, GI, RLDI
        - Cross-Cutting Methodologies: FMPT, ISAI
        - Applications: AISL
        """
        from papertrail.labs.lab_agent import ResearchLabAgent
        lab_agent = ResearchLabAgent(lab_code=lab_code, model=resolved_model)
        return lab_agent.run_sync(task)

    return agent


def create_scorecard_agent(model: Model | str | None = None) -> Agent[None, PaperReviewScorecard]:
    """Factory to construct a PaperReviewScorecard generator using PydanticAI."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="paper_review_scorer",
        instructions=(
            LEAD_SYSTEM_PROMPT
            + "\nYou must return a structured PaperReviewScorecard evaluating soundness, "
            "novelty, empirical rigor, clarity, overall verdict, strengths, weaknesses, and recommendations."
        ),
        output_type=PaperReviewScorecard,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
        ],
    )


def create_roadmap_agent(model: Model | str | None = None) -> Agent[None, ResearchRoadmap]:
    """Factory to construct a ResearchRoadmap generator using PydanticAI."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="research_roadmap_generator",
        instructions=(
            LEAD_SYSTEM_PROMPT
            + "\nYou must return a structured ResearchRoadmap detailing initiative name, "
            "strategic objective, phased timeline, cross-domain connections, critical risks, and delegation plan."
        ),
        output_type=ResearchRoadmap,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
            get_trending_topics,
        ],
    )


class LeadResearcher:
    """
    High-level interface for the Senior Research Lead.

    This is the primary conversational and orchestrating interface in PaperTrail.
    Coordinates across Researcher, Research Assistant, and Intern tiers.
    """

    def __init__(self, model: Model | str | None = None) -> None:
        self.model = model
        self.researcher = ResearcherAgent(model=model)
        self.assistant = ResearchAssistant(model=model)
        self.intern = ResearchIntern(model=model)
        self.agent = create_lead_agent(
            model=model,
            researcher=self.researcher,
            assistant=self.assistant,
            intern=self.intern,
        )
        self._scorecard_agent: Optional[Agent[None, PaperReviewScorecard]] = None
        self._roadmap_agent: Optional[Agent[None, ResearchRoadmap]] = None

    @property
    def scorecard_agent(self) -> Agent[None, PaperReviewScorecard]:
        if self._scorecard_agent is None:
            self._scorecard_agent = create_scorecard_agent(model=self.model)
        return self._scorecard_agent

    @property
    def roadmap_agent(self) -> Agent[None, ResearchRoadmap]:
        if self._roadmap_agent is None:
            self._roadmap_agent = create_roadmap_agent(model=self.model)
        return self._roadmap_agent

    def run_sync(self, prompt: str) -> str:
        """Run the Senior Research Lead synchronously and return markdown text."""
        result = self.agent.run_sync(prompt)
        return str(result.output)

    async def run(self, prompt: str) -> str:
        """Run the Senior Research Lead asynchronously and return markdown text."""
        result = await self.agent.run(prompt)
        return str(result.output)

    def review_paper_quality(self, paper_title_or_id: str) -> PaperReviewScorecard:
        """
        Conduct a peer-review evaluation of a paper or research proposal.
        Returns a structured PaperReviewScorecard.
        """
        prompt = (
            f"Conduct an expert peer-review evaluation of: '{paper_title_or_id}'.\n"
            f"Check abstract/text, evaluate soundness (1-5), novelty (1-5), empirical rigor (1-5), and clarity (1-5).\n"
            f"Provide overall verdict, strengths, weaknesses, and actionable recommendations."
        )
        result = self.scorecard_agent.run_sync(prompt)
        return result.output

    def plan_roadmap(self, initiative: str) -> ResearchRoadmap:
        """
        Develop a strategic multi-phase research roadmap with delegation assignments.
        """
        prompt = (
            f"Develop a comprehensive research roadmap for: '{initiative}'.\n"
            f"Define strategic objectives, phased milestones, cross-domain connections, "
            f"technical risks, and delegation responsibilities across Researcher, Assistant, and Intern."
        )
        result = self.roadmap_agent.run_sync(prompt)
        return result.output

    def orchestrate_campaign(self, topic: str) -> str:
        """
        Orchestrate a multi-tier research campaign across all 4 levels:
        1. Intern: Literature discovery & collection
        2. Assistant: Comparative breakdown & gap analysis
        3. Researcher: Hypothesis formulation & experiment design
        4. Lead: Strategic synthesis, methodology critique & roadmap
        """
        prompt = (
            f"Orchestrate a complete research campaign on: '{topic}'.\n"
            f"1. Task the Research Intern to inspect indexed and arXiv literature.\n"
            f"2. Task the Research Assistant to analyze comparative trade-offs and identify literature gaps.\n"
            f"3. Task the Researcher to formulate a testable hypothesis and design an empirical experiment.\n"
            f"4. As Lead, synthesize the final strategic direction, evaluating scientific viability and risks."
        )
        return self.run_sync(prompt)

    def consult_lab(self, lab_code: str, task: str) -> str:
        """
        Directly consult one of the 10 specialized research laboratories.
        """
        from papertrail.labs.lab_agent import ResearchLabAgent
        lab_agent = ResearchLabAgent(lab_code=lab_code, model=self.model)
        return lab_agent.run_sync(task)

    def orchestrate_matrix_collaboration(self, project: MatrixProject) -> CrossLabSynthesis:
        """
        Orchestrate a multi-lab cross-cutting matrix project across domains,
        methodologies, and applications.
        """
        from papertrail.labs.collaboration import MatrixProjectCoordinator
        coordinator = MatrixProjectCoordinator(model=self.model)
        return coordinator.execute_matrix_project(project)

