"""
Matrix Project Collaboration Coordinator.

Executes multi-lab cross-cutting projects combining:
- Core Research Domains (e.g. Agentic Intelligence)
- Cross-Cutting Methodologies (e.g. Foundation Model Post-Training)
- Applications & Translation (e.g. AI for Science & Learning)
"""
from __future__ import annotations

from typing import List, Optional
from pydantic_ai import Agent
from pydantic_ai.models import Model

from papertrail.labs.registry import get_lab
from papertrail.labs.lab_agent import ResearchLabAgent
from papertrail.schemas.schema import (
    MatrixProject,
    CrossLabSynthesis,
    InterLabRequest,
)
from papertrail.utils.llm import get_pydantic_ai_model
from papertrail.utils.observability import extract_model_name, agent_run_counter
import logfire


def create_matrix_synthesis_agent(model: Model | str | None = None) -> Agent[None, CrossLabSynthesis]:
    """Factory to construct an agent producing structured CrossLabSynthesis reports."""
    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name="matrix_project_synthesizer",
        instructions=(
            "You are the PaperTrail Cross-Lab Matrix Project Synthesizer. "
            "You synthesize multi-laboratory contributions (domain theory, cross-cutting methodologies, "
            "and downstream applications) into a cohesive CrossLabSynthesis report."
        ),
        output_type=CrossLabSynthesis,
    )


class MatrixProjectCoordinator:
    """
    Coordinates multi-lab matrix research initiatives across domains and methodologies.
    """

    def __init__(self, model: Model | str | None = None) -> None:
        self.model = model
        self.model_name = extract_model_name(model)
        self._synthesis_agent: Optional[Agent[None, CrossLabSynthesis]] = None

    @property
    def synthesis_agent(self) -> Agent[None, CrossLabSynthesis]:
        if self._synthesis_agent is None:
            self._synthesis_agent = create_matrix_synthesis_agent(model=self.model)
        return self._synthesis_agent

    def execute_matrix_project(self, project: MatrixProject) -> CrossLabSynthesis:
        """
        Execute a cross-lab matrix project by querying the domain lab, collaborating methodology labs,
        and application labs, then synthesizing the results into a CrossLabSynthesis.
        """
        participating = [project.primary_domain] + project.collaborating_methodologies + project.collaborating_applications
        unique_labs = list(dict.fromkeys(participating))

        with logfire.span(
            "matrix_collaboration.execute",
            project_name=project.project_name,
            primary_domain=project.primary_domain,
            participating_labs=unique_labs,
            model=self.model_name,
        ):
            logfire.info(
                "Coordinating cross-lab matrix project {project} across {labs} with model {model}",
                project=project.project_name,
                labs=", ".join(unique_labs),
                model=self.model_name,
            )
            agent_run_counter.add(
                1,
                {"agent": "matrix_coordinator", "model": self.model_name, "project": project.project_name},
            )

            # 1. Gather Domain Lab Insights
            with logfire.span("matrix_collaboration.domain_insights", domain=project.primary_domain):
                domain_agent = ResearchLabAgent(lab_code=project.primary_domain, model=self.model)
                domain_prompt = (
                    f"Matrix Project: '{project.project_name}'\n"
                    f"Mission: {project.mission_statement}\n"
                    f"As the primary domain lab ({project.primary_domain}), articulate the core theoretical mechanisms, "
                    f"foundational models, algorithms, and key challenges for this initiative."
                )
                domain_output = domain_agent.run_sync(domain_prompt)

            # 2. Gather Methodology Lab Specifications
            methodology_outputs: List[str] = []
            for m_code in project.collaborating_methodologies:
                with logfire.span("matrix_collaboration.methodology_insights", methodology=m_code):
                    m_agent = ResearchLabAgent(lab_code=m_code, model=self.model)
                    m_prompt = (
                        f"Matrix Project: '{project.project_name}'\n"
                        f"Domain Context from {project.primary_domain}: {domain_output[:400]}\n"
                        f"As the cross-cutting methodology lab ({m_code}), specify the exact adaptation, training, "
                        f"loss formulations, optimization algorithms, and infrastructure requirements."
                    )
                    m_resp = m_agent.run_sync(m_prompt)
                    methodology_outputs.append(f"[{m_code} Lab]: {m_resp}")

            # 3. Gather Application Lab Translation (if any)
            application_outputs: List[str] = []
            for a_code in project.collaborating_applications:
                with logfire.span("matrix_collaboration.application_insights", application=a_code):
                    a_agent = ResearchLabAgent(lab_code=a_code, model=self.model)
                    a_prompt = (
                        f"Matrix Project: '{project.project_name}'\n"
                        f"Mission: {project.mission_statement}\n"
                        f"As the application & translation lab ({a_code}), outline real-world deployment scenarios, "
                        f"evaluation benchmarks, user/learner impact, and practical considerations."
                    )
                    a_resp = a_agent.run_sync(a_prompt)
                    application_outputs.append(f"[{a_code} Lab]: {a_resp}")

            # 4. Synthesize via Structured Pydantic Model
            with logfire.span("matrix_collaboration.synthesis", model=self.model_name):
                synth_prompt = (
                    f"Synthesize this multi-lab matrix project into a CrossLabSynthesis report:\n"
                    f"Project Name: {project.project_name}\n"
                    f"Participating Labs: {', '.join(unique_labs)}\n"
                    f"Primary Domain Input: {domain_output}\n"
                    f"Methodology Contributions: {' | '.join(methodology_outputs)}\n"
                    f"Application Contributions: {' | '.join(application_outputs) if application_outputs else 'None'}\n"
                )
                result = self.synthesis_agent.run_sync(synth_prompt)
                synthesis = result.output
                synthesis.project_name = project.project_name
                synthesis.participating_labs = unique_labs
                return synthesis

