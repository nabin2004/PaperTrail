"""
Unit tests for PaperTrail Research Laboratories and Cross-Lab Matrix Collaboration.
"""
import pytest
from click.testing import CliRunner
from pydantic_ai.models.test import TestModel

from papertrail.labs.registry import (
    LAB_CATALOG,
    get_all_labs,
    get_lab,
    get_labs_by_category,
    list_lab_codes,
)
from papertrail.labs.lab_agent import ResearchLabAgent
from papertrail.labs.collaboration import MatrixProjectCoordinator
from papertrail.agents.lead_agent import LeadResearcher
from papertrail.schemas.schema import (
    InterLabRequest,
    InterLabResponse,
    MatrixProject,
    CrossLabSynthesis,
)
from papertrail.cli import cli


# ── Lab Catalog & Taxonomy Tests ─────────────────────────────────────────────

def test_lab_catalog_integrity():
    labs = get_all_labs()
    assert len(labs) == 10

    domains = get_labs_by_category("domain")
    methodologies = get_labs_by_category("methodology")
    applications = get_labs_by_category("application")

    assert len(domains) == 7
    assert len(methodologies) == 2
    assert len(applications) == 1

    expected_codes = ["LMI", "AI", "VI", "RCI", "SIKG", "GI", "RLDI", "FMPT", "ISAI", "AISL"]
    assert set(list_lab_codes()) == set(expected_codes)

    for code in expected_codes:
        lab = get_lab(code)
        assert lab is not None
        assert lab.code == code
        assert len(lab.full_name) > 0
        assert len(lab.short_name) > 0
        assert len(lab.focus) > 0
        assert len(lab.key_topics) > 0


# ── Lab Agent Tests ──────────────────────────────────────────────────────────

def test_lab_agent_creation_and_run():
    model = TestModel(call_tools=[])

    # Test Domain Lab
    ai_lab = ResearchLabAgent(lab_code="AI", model=model)
    assert ai_lab.code == "AI"
    res = ai_lab.run_sync("Propose an agent memory architecture")
    assert isinstance(res, str)

    # Test Methodology Lab
    fmpt_lab = ResearchLabAgent(lab_code="FMPT", model=model)
    assert fmpt_lab.code == "FMPT"
    res_fmpt = fmpt_lab.run_sync("Recommend DPO hyperparameters for tool-calling models")
    assert isinstance(res_fmpt, str)


def test_interlab_communication():
    model = TestModel(call_tools=[])

    # AI lab requests assistance from FMPT lab
    fmpt_lab = ResearchLabAgent(lab_code="FMPT", model=model)
    req = InterLabRequest(
        from_lab="AI",
        to_lab="FMPT",
        task_description="We need an SFT and DPO alignment recipe to train agents that do not hallucinate tool arguments.",
        context={"model_family": "Llama-3", "target_tools": ["search", "calculator"]},
    )

    resp = fmpt_lab.handle_interlab_request(req)
    assert isinstance(resp, InterLabResponse)
    assert resp.from_lab == "FMPT"
    assert resp.to_lab == "AI"
    assert len(resp.analysis) > 0


def test_matrix_project_coordinator():
    model = TestModel(call_tools=[])
    coordinator = MatrixProjectCoordinator(model=model)

    proj = MatrixProject(
        project_name="Autonomous Scientific Animator",
        primary_domain="AI",
        collaborating_methodologies=["FMPT"],
        collaborating_applications=["AISL"],
        mission_statement="Post-train an agentic model that generates precise Manim educational physics animations.",
    )

    synthesis = coordinator.execute_matrix_project(proj)
    assert isinstance(synthesis, CrossLabSynthesis)
    assert synthesis.project_name == "Autonomous Scientific Animator"
    assert "AI" in synthesis.participating_labs
    assert "FMPT" in synthesis.participating_labs
    assert "AISL" in synthesis.participating_labs


def test_lead_consult_lab():
    model = TestModel(call_tools=[])
    lead = LeadResearcher(model=model)

    res = lead.consult_lab("ISAI", "How do we optimize KV cache for 1M tokens?")
    assert isinstance(res, str)


# ── CLI Commands Tests ───────────────────────────────────────────────────────

def test_cli_labs():
    runner = CliRunner()
    result = runner.invoke(cli, ["labs"])
    assert result.exit_code == 0
    assert "Language & Multimodal Intelligence Lab" in result.output
    assert "Agentic Intelligence Lab" in result.output
    assert "Foundation Model Post-Training Lab" in result.output
    assert "Intelligent Systems & AI Infrastructure Lab" in result.output
    assert "AI for Science & Learning Lab" in result.output


def test_cli_lab_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["lab", "--help"])
    assert result.exit_code == 0
    assert "Interact directly with a specialized research laboratory" in result.output


def test_cli_matrix_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["matrix", "--help"])
    assert result.exit_code == 0
    assert "Launch a cross-lab matrix project" in result.output
    assert "--domain" in result.output
    assert "--methodology" in result.output
    assert "--application" in result.output
