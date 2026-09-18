"""
PaperTrail agents – research, scientific planning, and PydanticAI research intern.
"""
from papertrail.agents.lead_agent import (
    LeadResearcher,
    create_lead_agent,
    create_scorecard_agent,
    create_roadmap_agent,
)
from papertrail.agents.assistant_agent import (
    ResearchAssistant,
    create_assistant_agent,
    create_dossier_agent,
    create_gap_agent,
    create_benchmark_agent,
)
from papertrail.agents.intern_agent import ResearchIntern, create_intern_agent
from papertrail.agents.research_agent import ResearchAgent
from papertrail.agents.scientist_agent import (
    ScientistAgent,
    create_scientist_agent,
    ResearcherAgent,
    create_researcher_agent,
    create_hypothesis_spec_agent,
    create_experiment_spec_agent,
)
from papertrail.agents.tools import (
    download_and_index_paper,
    get_pydantic_ai_tools,
    get_trending_topics,
    list_indexed_papers,
    read_paper_abstract,
    search_arxiv,
    search_local_papers,
)

__all__ = [
    "LeadResearcher",
    "create_lead_agent",
    "create_scorecard_agent",
    "create_roadmap_agent",
    "ResearcherAgent",
    "create_researcher_agent",
    "ScientistAgent",
    "create_scientist_agent",
    "create_hypothesis_spec_agent",
    "create_experiment_spec_agent",
    "ResearchAssistant",
    "create_assistant_agent",
    "create_dossier_agent",
    "create_gap_agent",
    "create_benchmark_agent",
    "ResearchIntern",
    "create_intern_agent",
    "ResearchAgent",
    "search_local_papers",
    "search_arxiv",
    "download_and_index_paper",
    "list_indexed_papers",
    "get_trending_topics",
    "read_paper_abstract",
    "get_pydantic_ai_tools",
]
