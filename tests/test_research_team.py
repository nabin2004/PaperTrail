"""
Unit tests for the 4-tier PydanticAI Research Team:
- Tier 1: Research Intern
- Tier 2: Research Assistant
- Tier 3: Researcher (ScientistAgent)
- Tier 4: Senior Research Lead (LeadResearcher)
"""
import pytest
from click.testing import CliRunner
from pydantic_ai.models.test import TestModel

from papertrail.agents.intern_agent import ResearchIntern
from papertrail.agents.assistant_agent import ResearchAssistant
from papertrail.agents.scientist_agent import ResearcherAgent, ScientistAgent
from papertrail.agents.lead_agent import (
    LeadResearcher,
    create_lead_agent,
    create_scorecard_agent,
    create_roadmap_agent,
)
from papertrail.schemas.schema import (
    HypothesisSpec,
    ExperimentSpec,
    BenchmarkResult,
    PaperReviewScorecard,
    ResearchRoadmap,
    ResearchDossier,
    GapAnalysis,
)
from papertrail.cli import cli


# ── Tier 1: Research Intern Tests ─────────────────────────────────────────────

def test_intern_extended_methods():
    model = TestModel(call_tools=[])
    intern = ResearchIntern(model=model)

    res_collect = intern.collect_data("graph neural networks", limit=3)
    assert isinstance(res_collect, str)

    res_reproduce = intern.reproduce_study("2301.00001")
    assert isinstance(res_reproduce, str)

    res_baseline = intern.run_baseline_experiment("linear attention")
    assert isinstance(res_baseline, str)


# ── Tier 2: Research Assistant Tests ──────────────────────────────────────────

def test_assistant_extended_methods():
    model = TestModel(call_tools=[])
    assistant = ResearchAssistant(model=model)

    bench = assistant.run_benchmarks("transformer vs mamba")
    assert isinstance(bench, BenchmarkResult)
    assert bench.benchmark_name

    plan = assistant.plan_implementation("bidirectional state space model")
    assert isinstance(plan, str)


# ── Tier 3: Researcher Tests ──────────────────────────────────────────────────

def test_researcher_spec_and_interpretation():
    model = TestModel(call_tools=[])
    researcher = ResearcherAgent(model=model)

    hypo_spec = researcher.formulate_hypothesis_spec("associative recall in recurrent models")
    assert isinstance(hypo_spec, HypothesisSpec)
    assert hypo_spec.hypothesis

    exp_spec = researcher.design_experiment_spec("Linear attention scales without degradation")
    assert isinstance(exp_spec, ExperimentSpec)
    assert exp_spec.title

    interp = researcher.interpret_results("Model A achieved 94.2% on Needle-in-a-Haystack vs 91.0% baseline")
    assert isinstance(interp, str)


# ── Tier 4: Senior Research Lead Tests ───────────────────────────────────────

def test_lead_researcher_creation_and_run():
    model = TestModel(call_tools=[])
    lead = LeadResearcher(model=model)
    assert lead.agent.name == "senior_research_lead"

    res = lead.run_sync("What research direction should we pursue in long-context models?")
    assert isinstance(res, str)


def test_lead_researcher_review_quality():
    model = TestModel(call_tools=[])
    lead = LeadResearcher(model=model)

    scorecard = lead.review_paper_quality("Attention Is All You Need")
    assert isinstance(scorecard, PaperReviewScorecard)
    assert scorecard.paper_title
    assert 1 <= scorecard.soundness_score <= 5


def test_lead_researcher_roadmap():
    model = TestModel(call_tools=[])
    lead = LeadResearcher(model=model)

    roadmap = lead.plan_roadmap("sub-quadratic foundation models")
    assert isinstance(roadmap, ResearchRoadmap)
    assert roadmap.initiative


def test_lead_researcher_orchestrate_campaign():
    model = TestModel(call_tools=[])
    lead = LeadResearcher(model=model)

    campaign = lead.orchestrate_campaign("efficient sequence modeling")
    assert isinstance(campaign, str)


# ── CLI Commands Tests ────────────────────────────────────────────────────────

def test_cli_lead_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["lead", "--help"])
    assert result.exit_code == 0
    assert "Interact with the Senior Research Lead & PI" in result.output
    assert "--orchestrate" in result.output
    assert "--review-quality" in result.output
    assert "--roadmap" in result.output


def test_cli_researcher_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["researcher", "--help"])
    assert result.exit_code == 0
    assert "Interact directly with the Tier 3 AI Researcher" in result.output
    assert "--hypothesis" in result.output
    assert "--experiment" in result.output
    assert "--interpret" in result.output
