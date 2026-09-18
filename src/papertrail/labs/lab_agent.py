"""
ResearchLabAgent – Autonomous specialized research laboratory agent powered by PydanticAI.

Each laboratory operates as an expert scientific division with domain-specific
guidance, access to literature tools, and inter-lab communication protocols.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from papertrail.labs.registry import get_lab, LAB_CATALOG
from papertrail.schemas.schema import (
    LabInfo,
    InterLabRequest,
    InterLabResponse,
)
from papertrail.agents.tools import (
    search_local_papers,
    search_arxiv,
    read_paper_abstract,
    get_trending_topics,
    list_indexed_papers,
)
from papertrail.utils.llm import get_pydantic_ai_model


def build_lab_system_prompt(lab: LabInfo) -> str:
    """Construct a tailored system prompt for a specific research lab."""
    cat_descriptor = {
        "domain": "Core Research Domain",
        "methodology": "Cross-Cutting Methodology Lab",
        "application": "Application & Translation Lab",
    }[lab.category]

    topics_list = "\n".join(f"- {t}" for t in lab.key_topics)

    return f"""\
You are the **{lab.full_name}** ({lab.short_name}), a world-class {cat_descriptor} within the PaperTrail AI Research Organization.

### Your Core Mission:
{lab.focus}

### Key Scientific & Technical Topics in Your Mandate:
{topics_list}

### Operating Directives:
1. **Scientific Rigor:** Anchor all answers, recommendations, and methodologies in empirical research literature.
2. **Distinct Competence:** Provide deep specialization in your field. Do not give generic machine learning answers—deliver the specialized algorithms, state-of-the-art benchmarks, mathematical formulations, and engineering designs expected from the {lab.code} Lab.
3. **Cross-Lab Synergy:** If an inquiry requires assistance from another lab (e.g. asking a domain lab for post-training recommendations or infrastructure optimization), recognize cross-cutting boundaries and formulate explicit inter-lab collaboration interfaces.
4. **Actionable Outputs:** Provide concrete technical methodologies, baseline comparisons, dataset recommendations, and next experimental steps.
"""


def create_lab_pydantic_agent(lab_code: str, model: Model | str | None = None) -> Agent[None, str]:
    """Factory to construct a PydanticAI Agent tailored to a specific research lab."""
    lab = get_lab(lab_code)
    if lab is None:
        raise ValueError(f"Unknown research lab code: '{lab_code}'. Valid codes: {list(LAB_CATALOG.keys())}")

    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    agent = Agent(
        model=resolved_model,
        name=f"lab_{lab.code.lower()}",
        instructions=build_lab_system_prompt(lab),
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
            get_trending_topics,
            list_indexed_papers,
        ],
    )
    return agent


def create_interlab_response_agent(lab_code: str, model: Model | str | None = None) -> Agent[None, InterLabResponse]:
    """Factory to construct an agent producing structured InterLabResponse payloads."""
    lab = get_lab(lab_code)
    if lab is None:
        raise ValueError(f"Unknown research lab code: '{lab_code}'")

    if model is None:
        try:
            resolved_model = get_pydantic_ai_model()
        except Exception:
            resolved_model = "test"
    else:
        resolved_model = model

    return Agent(
        model=resolved_model,
        name=f"interlab_responder_{lab.code.lower()}",
        instructions=(
            build_lab_system_prompt(lab)
            + "\nYou are responding to an InterLab consultation request from another laboratory. "
            "You must return a structured InterLabResponse detailing your specialized analysis, "
            "concrete methodology recommendations, and proposed experiments."
        ),
        output_type=InterLabResponse,
        tools=[
            search_local_papers,
            search_arxiv,
            read_paper_abstract,
        ],
    )


class ResearchLabAgent:
    """
    High-level interface for a specialized PaperTrail research laboratory.
    """

    def __init__(self, lab_code: str, model: Model | str | None = None) -> None:
        self.lab_info = get_lab(lab_code)
        if self.lab_info is None:
            raise ValueError(f"Unknown research lab code: '{lab_code}'")
        self.code = self.lab_info.code
        self.model = model
        self.agent = create_lab_pydantic_agent(lab_code=self.code, model=model)
        self._responder_agent: Optional[Agent[None, InterLabResponse]] = None

    @property
    def responder_agent(self) -> Agent[None, InterLabResponse]:
        if self._responder_agent is None:
            self._responder_agent = create_interlab_response_agent(lab_code=self.code, model=self.model)
        return self._responder_agent

    def run_sync(self, prompt: str) -> str:
        """Run the laboratory agent synchronously."""
        result = self.agent.run_sync(prompt)
        return str(result.output)

    async def run(self, prompt: str) -> str:
        """Run the laboratory agent asynchronously."""
        result = await self.agent.run(prompt)
        return str(result.output)

    def handle_interlab_request(self, request: InterLabRequest) -> InterLabResponse:
        """
        Process a structured consultation request from another laboratory.
        """
        prompt = (
            f"InterLab Consultation Request from {request.from_lab} Lab to {self.code} Lab:\n"
            f"Task/Objective: {request.task_description}\n"
            f"Context: {request.context}\n\n"
            f"Provide your specialized {self.lab_info.full_name} analysis, concrete methodology recommendations, "
            f"and proposed validation experiments."
        )
        result = self.responder_agent.run_sync(prompt)
        # Ensure correct lab routing metadata
        resp = result.output
        resp.from_lab = self.code
        resp.to_lab = request.from_lab
        return resp

    def consult_lab(self, target_lab_code: str, task: str, context: Optional[Dict[str, Any]] = None) -> InterLabResponse:
        """
        Send a consultation request to a peer laboratory.
        """
        req = InterLabRequest(
            from_lab=self.code,
            to_lab=target_lab_code.upper(),
            task_description=task,
            context=context or {},
        )
        target_agent = ResearchLabAgent(lab_code=target_lab_code, model=self.model)
        return target_agent.handle_interlab_request(req)
