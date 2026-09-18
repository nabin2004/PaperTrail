"""
Unit tests for PydanticAI-powered agents (ResearchIntern, ScientistAgent, tools).
"""
import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from pydantic_ai.models.test import TestModel

from papertrail.agents.assistant_agent import (
    ResearchAssistant,
    create_assistant_agent,
    create_dossier_agent,
    create_gap_agent,
)
from papertrail.agents.intern_agent import ResearchIntern, create_intern_agent
from papertrail.agents.scientist_agent import ScientistAgent, create_scientist_agent
from papertrail.agents.tools import (
    download_and_index_paper,
    get_pydantic_ai_tools,
    get_trending_topics,
    list_indexed_papers,
    read_paper_abstract,
    search_arxiv,
    search_local_papers,
)
from papertrail.cli import cli
from papertrail.schemas.schema import (
    Paper,
    ResearchDossier,
    GapAnalysis,
    PaperSummary,
    PaperComparisonDimension,
)


# ── Tool Tests ───────────────────────────────────────────────────────────────

def test_pydantic_ai_tools_list():
    tools = get_pydantic_ai_tools()
    assert len(tools) == 6
    for t in tools:
        assert callable(t)


def test_list_indexed_papers(monkeypatch):
    monkeypatch.setattr(
        "papertrail.ingestion.metadata.load_all_metadata",
        lambda: [
            Paper(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                abstract="Summary",
                authors=["Ashish Vaswani", "Noam Shazeer"],
                primary_category="cs.CL",
                categories=["cs.CL"],
                published="2017-06-12T00:00:00Z",
                updated="2017-06-12T00:00:00Z",
            )
        ],
    )
    res = list_indexed_papers()
    assert "Attention Is All You Need" in res
    assert "1706.03762" in res


def test_get_trending_topics(monkeypatch):
    monkeypatch.setattr(
        "papertrail.ingestion.metadata.load_all_metadata",
        lambda: [
            Paper(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                abstract="Summary with attention mechanism and transformer",
                authors=["Ashish Vaswani"],
                primary_category="cs.CL",
                categories=["cs.CL"],
                published="2017-06-12T00:00:00Z",
                updated="2017-06-12T00:00:00Z",
            )
        ],
    )
    res = get_trending_topics()
    assert "Top Keywords" in res or "cs.CL" in res


def test_search_local_papers(monkeypatch):
    res = search_local_papers("transformer", top_k=2)
    assert isinstance(res, str)
    assert len(res) > 0


def test_search_arxiv_tool():
    with patch("papertrail.ingestion.arxiv_client.ArxivClient.fetch_papers") as mock_fetch:
        mock_fetch.return_value = [
            Paper(
                arxiv_id="2301.00001",
                title="Recent Advance in Machine Learning",
                abstract="This paper discusses new architectures.",
                authors=["Alice Researcher"],
                primary_category="cs.AI",
                categories=["cs.AI"],
                published="2023-01-01T00:00:00Z",
                updated="2023-01-01T00:00:00Z",
            )
        ]
        res = search_arxiv("machine learning", limit=1)
        assert "Recent Advance in Machine Learning" in res
        assert "2301.00001" in res


def test_download_and_index_paper_tool():
    with patch("papertrail.ingestion.arxiv_client.ArxivClient.fetch_papers") as mock_fetch:
        mock_fetch.return_value = [
            Paper(
                arxiv_id="2301.00001",
                title="Recent Advance in Machine Learning",
                abstract="This paper discusses new architectures.",
                authors=["Alice Researcher"],
                primary_category="cs.AI",
                categories=["cs.AI"],
                published="2023-01-01T00:00:00Z",
                updated="2023-01-01T00:00:00Z",
            )
        ]
        with patch("papertrail.ingestion.pipeline.index_single_paper") as mock_index:
            mock_index.return_value = (True, "indexed (15 chunks)", 15)
            res = download_and_index_paper("2301.00001")
            assert "Successfully downloaded and indexed" in res
            assert "15 text chunks" in res


# ── ResearchIntern Agent Tests ───────────────────────────────────────────────

def test_research_intern_agent_creation():
    model = TestModel(call_tools=[])
    intern = ResearchIntern(model=model)
    assert intern.agent.name == "research_intern"

    res = intern.run_sync("What are trending topics?")
    assert isinstance(res, str)


def test_research_intern_investigate():
    model = TestModel(call_tools=[])
    intern = ResearchIntern(model=model)
    res = intern.investigate("graph neural networks")
    assert isinstance(res, str)


# ── ScientistAgent Tests ─────────────────────────────────────────────────────

def test_scientist_agent_plan():
    model = TestModel(call_tools=[])
    scientist = ScientistAgent(model=model)
    plan = scientist.generate_plan("quantum computing algorithms")
    assert isinstance(plan, str)


def test_scientist_agent_hypothesis():
    model = TestModel(call_tools=[])
    scientist = ScientistAgent(model=model)
    hypo = scientist.generate_hypothesis("diffusion models for speech")
    assert isinstance(hypo, str)


def test_scientist_agent_experiment():
    model = TestModel(call_tools=[])
    scientist = ScientistAgent(model=model)
    exp = scientist.design_experiment("State space models achieve linear attention scaling")
    assert isinstance(exp, str)


def test_scientist_agent_grounded_in_dossier():
    model = TestModel(call_tools=[])
    scientist = ScientistAgent(model=model)
    dossier = ResearchDossier(
        topic="state space models",
        executive_summary="SSMs provide linear complexity.",
        papers_analyzed=[
            PaperSummary(
                paper_id="2312.00752",
                title="Mamba: Linear-Time Sequence Modeling",
                methodology="Selective state spaces",
                key_findings=["Matches transformer perplexity"],
            )
        ],
        gap_analysis=GapAnalysis(
            topic="state space models",
            identified_gaps=["Associative recall at scale"],
            untested_assumptions=["Exact copy mechanisms"],
            open_questions=["Can pure SSMs match in-context learning?"],
        ),
    )
    hypo = scientist.generate_hypothesis("state space models", dossier=dossier)
    assert isinstance(hypo, str)
    plan = scientist.generate_plan("state space models", dossier=dossier)
    assert isinstance(plan, str)


# ── ResearchAssistant Agent Tests ────────────────────────────────────────────

def test_research_assistant_creation():
    model = TestModel(call_tools=[])
    assistant = ResearchAssistant(model=model)
    assert assistant.agent.name == "research_assistant"

    res = assistant.run_sync("Compare linear attention vs softmax attention")
    assert isinstance(res, str)


def test_research_assistant_compare_papers():
    model = TestModel(call_tools=[])
    assistant = ResearchAssistant(model=model)
    res = assistant.compare_papers("Mamba vs Transformer")
    assert isinstance(res, str)


def test_research_assistant_analyze_gaps():
    model = TestModel(call_tools=[])
    assistant = ResearchAssistant(model=model)
    gap_result = assistant.analyze_gaps("efficient transformers")
    assert isinstance(gap_result, GapAnalysis)
    assert gap_result.topic


def test_research_assistant_prepare_dossier():
    model = TestModel(call_tools=[])
    assistant = ResearchAssistant(model=model)
    dossier = assistant.prepare_dossier("sparse attention")
    assert isinstance(dossier, ResearchDossier)
    assert dossier.topic


def test_research_assistant_delegate_to_intern():
    model = TestModel(call_tools=[])
    assistant = ResearchAssistant(model=model)
    res = assistant.delegate_to_intern("Check local papers on retrieval")
    assert isinstance(res, str)


# ── CLI Commands ─────────────────────────────────────────────────────────────

def test_cli_intern_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["intern", "--help"])
    assert result.exit_code == 0
    assert "Interact with the PaperTrail AI Research Intern" in result.output
    assert "--investigate" in result.output


def test_cli_assistant_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["assistant", "--help"])
    assert result.exit_code == 0
    assert "Interact with the Senior AI Research Assistant" in result.output
    assert "--compare" in result.output
    assert "--gaps" in result.output
    assert "--dossier" in result.output


def test_cli_scientist_commands_help():
    runner = CliRunner()
    for cmd in ["plan", "hypothesis", "experiment"]:
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0
