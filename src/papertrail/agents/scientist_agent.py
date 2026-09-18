"""
ScientistAgent – AI Scientist agent powered by PydanticAI.

Capable of literature review, hypothesis generation, experiment design,
and research planning grounded in PaperTrail's tools.
"""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic_ai import Agent
from pydantic_ai.models import Model

from papertrail.agents.tools import (
    search_local_papers,
    search_arxiv,
    get_trending_topics,
    read_paper_abstract,
)
from papertrail.schemas.schema import (
    ResearchDossier,
    HypothesisSpec,
    ExperimentSpec,
)
from papertrail.utils.llm import get_pydantic_ai_model
from papertrail.utils.observability import extract_model_name, agent_run_counter
import logfire


SCIENTIST_SYSTEM_PROMPT = """\
You are the PaperTrail AI Scientist — a senior research scientist in AI and Computer Science.

Your purpose is to assist in scientific discovery by:
1. Synthesizing existing scientific paradigms and identifying research gaps.
2. Formulating novel, testable, and falsifiable research hypotheses.
3. Designing robust, controlled experiments with baselines, datasets, and evaluation metrics.
4. Structuring literature review trees and future research directions.

Always ground your reasoning in actual empirical literature found via search tools.
"""


def create_scientist_agent(model: Model | str | None = None) -> Agent[None, str]:
    """Factory to construct a PydanticAI Scientist Agent."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="ai_scientist",
        instructions=SCIENTIST_SYSTEM_PROMPT,
        tools=[
            search_local_papers,
            search_arxiv,
            get_trending_topics,
            read_paper_abstract,
        ],
    )


def create_hypothesis_spec_agent(model: Model | str | None = None) -> Agent[None, HypothesisSpec]:
    """Factory to construct an agent producing structured HypothesisSpec outputs."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="hypothesis_spec_generator",
        instructions=(
            SCIENTIST_SYSTEM_PROMPT
            + "\nYou must return a structured HypothesisSpec containing the hypothesis statement, "
            "theoretical basis, independent and dependent variables, and specific falsification criteria."
        ),
        output_type=HypothesisSpec,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
        ],
    )


def create_experiment_spec_agent(model: Model | str | None = None) -> Agent[None, ExperimentSpec]:
    """Factory to construct an agent producing structured ExperimentSpec outputs."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="experiment_spec_generator",
        instructions=(
            SCIENTIST_SYSTEM_PROMPT
            + "\nYou must return a structured ExperimentSpec containing title, hypothesis, "
            "datasets, baselines, ablation conditions, quantitative metrics, and expected outcomes."
        ),
        output_type=ExperimentSpec,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
        ],
    )


class ScientistAgent:
    """
    AI Scientist Agent supporting hypothesis generation, experimental design,
    and research roadmap planning.
    """

    def __init__(self, model: Model | str | None = None) -> None:
        self.prompt = ""
        self.memory: List[tuple[str, str]] = []
        self.plan: Optional[str] = None
        self.model = model
        self.model_name = extract_model_name(model)
        self.agent = create_scientist_agent(model=model)

    def generate_plan(self, topic: str, dossier: Optional[ResearchDossier] = None) -> str:
        """Generate a structured research plan for a topic, optionally grounded in a ResearchDossier."""
        with logfire.span(
            "researcher.generate_plan",
            agent="ai_scientist",
            model=self.model_name,
            topic=topic,
        ):
            logfire.info(
                "AI Researcher generating research plan for {topic} with model {model}",
                topic=topic,
                model=self.model_name,
            )
            agent_run_counter.add(
                1, {"agent": "ai_scientist", "model": self.model_name, "action": "generate_plan"}
            )
            dossier_context = ""
            if dossier:
                dossier_context = (
                    f"\n\nContext from Research Assistant Dossier:\n"
                    f"- Executive Summary: {dossier.executive_summary}\n"
                    f"- Papers Analyzed: {len(dossier.papers_analyzed)}\n"
                    f"- Identified Gaps: {dossier.gap_analysis.identified_gaps if dossier.gap_analysis else 'N/A'}\n"
                )
            prompt = (
                f"Formulate a structured research plan for: '{topic}'.\n"
                f"Use search tools to inspect relevant papers.{dossier_context}\n"
                f"Cover: background literature, key questions, methodology, and expected timeline."
            )
            result = self.agent.run_sync(prompt)
            self.plan = str(result.output)
            return self.plan

    def generate_hypothesis(self, topic: str, dossier: Optional[ResearchDossier] = None) -> str:
        """Formulate a novel, falsifiable research hypothesis for a topic, optionally grounded in a ResearchDossier."""
        with logfire.span(
            "researcher.generate_hypothesis",
            agent="ai_scientist",
            model=self.model_name,
            topic=topic,
        ):
            logfire.info(
                "AI Researcher formulating hypothesis for {topic} with model {model}",
                topic=topic,
                model=self.model_name,
            )
            agent_run_counter.add(
                1, {"agent": "ai_scientist", "model": self.model_name, "action": "generate_hypothesis"}
            )
            dossier_context = ""
            if dossier:
                dossier_context = (
                    f"\n\nEvidence from Research Assistant Dossier:\n"
                    f"- Gaps: {dossier.gap_analysis.identified_gaps if dossier.gap_analysis else 'N/A'}\n"
                    f"- Untested Assumptions: {dossier.gap_analysis.untested_assumptions if dossier.gap_analysis else 'N/A'}\n"
                    f"- Open Questions: {dossier.gap_analysis.open_questions if dossier.gap_analysis else 'N/A'}\n"
                )
            prompt = (
                f"Based on current literature in '{topic}', identify major gaps or limitations "
                f"{dossier_context}\n"
                f"and propose 2-3 novel, falsifiable scientific hypotheses with theoretical justification."
            )
            result = self.agent.run_sync(prompt)
            return str(result.output)

    def design_experiment(self, hypothesis: str) -> str:
        """Design an empirical experiment to evaluate a hypothesis."""
        with logfire.span(
            "researcher.design_experiment",
            agent="ai_scientist",
            model=self.model_name,
            hypothesis_preview=hypothesis[:80],
        ):
            logfire.info(
                "AI Researcher designing experiment with model {model}",
                model=self.model_name,
                hypothesis_preview=hypothesis[:80],
            )
            agent_run_counter.add(
                1, {"agent": "ai_scientist", "model": self.model_name, "action": "design_experiment"}
            )
            prompt = (
                f"Design an empirical evaluation for this hypothesis:\n'{hypothesis}'\n\n"
                f"Include:\n"
                f"1. Datasets & Benchmarks\n"
                f"2. Baseline Models to compare against\n"
                f"3. Quantitative Metrics\n"
                f"4. Ablation Studies\n"
                f"5. Failure modes & verification criteria"
            )
            result = self.agent.run_sync(prompt)
            return str(result.output)

    def formulate_hypothesis_spec(self, topic: str) -> HypothesisSpec:
        """Formulate a strongly typed, testable HypothesisSpec for a topic."""
        with logfire.span(
            "researcher.formulate_hypothesis_spec",
            agent="ai_scientist",
            model=self.model_name,
            topic=topic,
        ):
            logfire.info(
                "AI Researcher generating structured HypothesisSpec for {topic} with model {model}",
                topic=topic,
                model=self.model_name,
            )
            hypo_agent = create_hypothesis_spec_agent(model=self.agent.model)
            prompt = (
                f"Formulate a precise, testable scientific hypothesis on: '{topic}'.\n"
                f"Ground it in current theoretical limits and specify concrete falsification criteria."
            )
            result = hypo_agent.run_sync(prompt)
            return result.output

    def design_experiment_spec(self, hypothesis: str) -> ExperimentSpec:
        """Design a strongly typed, controlled ExperimentSpec evaluating a hypothesis."""
        with logfire.span(
            "researcher.design_experiment_spec",
            agent="ai_scientist",
            model=self.model_name,
            hypothesis_preview=hypothesis[:80],
        ):
            logfire.info(
                "AI Researcher generating structured ExperimentSpec with model {model}",
                model=self.model_name,
                hypothesis_preview=hypothesis[:80],
            )
            exp_agent = create_experiment_spec_agent(model=self.agent.model)
            prompt = (
                f"Design a rigorous empirical ExperimentSpec to validate or falsify:\n'{hypothesis}'\n"
                f"Specify datasets, baselines, ablation conditions, metrics, and expected outcomes."
            )
            result = exp_agent.run_sync(prompt)
            return result.output

    def interpret_results(self, results_summary: str, hypothesis: Optional[str] = None) -> str:
        """
        Interpret empirical benchmark results against theoretical claims or a hypothesis.
        Determine if the findings corroborate, falsify, or complicate the hypothesis.
        """
        with logfire.span(
            "researcher.interpret_results",
            agent="ai_scientist",
            model=self.model_name,
            has_hypothesis=bool(hypothesis),
        ):
            logfire.info(
                "AI Researcher interpreting empirical results with model {model}",
                model=self.model_name,
                has_hypothesis=bool(hypothesis),
            )
            hypo_text = f"\nHypothesis: '{hypothesis}'\n" if hypothesis else ""
            prompt = (
                f"Result Interpretation Analysis:{hypo_text}\n"
                f"Empirical Results:\n{results_summary}\n\n"
                f"1. Empirical Validation: Do the data support, contradict, or leave the hypothesis inconclusive?\n"
                f"2. Theoretical Ramifications: Why did these outcomes occur?\n"
                f"3. Failure Mode & Anomaly Diagnosis: Were there unexpected degradations or baseline anomalies?\n"
                f"4. Follow-up Recommendations: Next experiments to run."
            )
            result = self.agent.run_sync(prompt)
            return str(result.output)

    # ── Backward compatibility ───────────────────────────────────────────────

    def observe(self, input_data: Any) -> str:
        observation = f"Observation: {input_data}"
        self.memory.append(("Observe", observation))
        return observation

    def think(self, observation: str) -> str:
        thought = f"Thought: Based on '{observation}', I will analyze the research direction."
        self.memory.append(("think", thought))
        return thought

    def act(self, thought: str) -> str:
        if not self.plan:
            self.plan = (
                "Research Plan:\n"
                "1. Identify key papers on the topic.\n"
                "2. Summarize core approaches.\n"
                "3. Highlight current limitations and open challenges.\n"
                "4. Propose a novel direction based on the findings."
            )
        action = f"Action: Generated research plan.\n{self.plan}"
        self.memory.append(("act", action))
        return action


# Alias for Tier 3 Researcher
ResearcherAgent = ScientistAgent
create_researcher_agent = create_scientist_agent