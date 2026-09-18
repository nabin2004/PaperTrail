"""Tests for PaperTrail Logfire Observability & Tracing Infrastructure."""
import os
import pytest
from click.testing import CliRunner
from logfire.testing import CaptureLogfire

from papertrail.utils.observability import (
    setup_observability,
    extract_model_name,
    cli_command_counter,
    agent_run_counter,
    papers_indexed_counter,
    retrieval_histogram,
)
from papertrail.agents.intern_agent import ResearchIntern
from papertrail.agents.assistant_agent import ResearchAssistant
from papertrail.agents.scientist_agent import ResearcherAgent, ScientistAgent
from papertrail.agents.lead_agent import LeadResearcher
from papertrail.labs.lab_agent import ResearchLabAgent
from papertrail.labs.collaboration import MatrixProjectCoordinator
from papertrail.cli import cli


def test_setup_observability_idempotence():
    """Verify setup_observability initializes Logfire and OpenTelemetry idempotently."""
    assert setup_observability(send_to_logfire=False, console=False) is True
    # Calling again should return True without re-running or raising
    assert setup_observability() is True
    assert os.environ.get("LANGSMITH_OTEL_ENABLED") == "true"
    assert os.environ.get("LANGSMITH_OTEL_ONLY") == "true"


def test_extract_model_name_variants():
    """Verify model name extraction across strings, objects, and environment variables."""
    # 1. Plain string
    assert extract_model_name("gpt-4o") == "gpt-4o"
    assert extract_model_name("claude-3-5-sonnet") == "claude-3-5-sonnet"

    # 2. Object with model_name attribute
    class DummyPydanticAIModel:
        model_name = "custom-llama-3.3"

    assert extract_model_name(DummyPydanticAIModel()) == "custom-llama-3.3"

    # 3. Object with model attribute
    class DummyLangChainModel:
        model = "gpt-4o-mini"

    assert extract_model_name(DummyLangChainModel()) == "gpt-4o-mini"

    # 4. Fallback to env default
    env_default = extract_model_name(None)
    assert isinstance(env_default, str)
    assert len(env_default) > 0


def test_agent_hierarchy_model_names_tracked():
    """Verify that all agent tiers and labs preserve explicit model names."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    custom_model = OpenAIChatModel("custom-gpt-5", provider=OpenAIProvider(api_key="mock"))
    expected_name = "custom-gpt-5"

    intern = ResearchIntern(model=custom_model)
    assert intern.model_name == expected_name

    assistant = ResearchAssistant(model=custom_model)
    assert assistant.model_name == expected_name

    researcher = ResearcherAgent(model=custom_model)
    assert researcher.model_name == expected_name

    scientist = ScientistAgent(model=custom_model)
    assert scientist.model_name == expected_name

    lead = LeadResearcher(model=custom_model)
    assert lead.model_name == expected_name

    lab = ResearchLabAgent(lab_code="AI", model=custom_model)
    assert lab.model_name == expected_name

    coordinator = MatrixProjectCoordinator(model=custom_model)
    assert coordinator.model_name == expected_name



def test_spans_record_model_name_and_metadata(capfire: CaptureLogfire):
    """Verify that agent executions emit Logfire spans with model name and attributes."""
    setup_observability(send_to_logfire=False, console=False)

    lead = LeadResearcher(model="test")
    res = lead.run_sync("Formulate research objective")
    assert isinstance(res, str)

    spans = capfire.exporter.exported_spans_as_dict()
    # Check that a span with agent='senior_research_lead' and model='test' was emitted
    matching_spans = [
        s for s in spans
        if s.get("attributes", {}).get("agent") == "senior_research_lead"
    ]
    assert len(matching_spans) > 0
    span_attrs = matching_spans[0]["attributes"]
    assert span_attrs.get("model") == "test"
    assert "prompt_preview" in span_attrs


def test_lab_agent_spans_record_lab_code_and_model(capfire: CaptureLogfire):
    """Verify specialized lab agent inquiries emit spans with lab_code and model."""
    setup_observability(send_to_logfire=False, console=False)

    lab = ResearchLabAgent(lab_code="FMPT", model="test")
    res = lab.run_sync("What are the best practices for GRPO training?")
    assert isinstance(res, str)

    spans = capfire.exporter.exported_spans_as_dict()
    lab_spans = [
        s for s in spans
        if s.get("attributes", {}).get("lab_code") == "FMPT"
    ]
    assert len(lab_spans) > 0
    assert lab_spans[0]["attributes"].get("model") == "test"


def test_cli_command_spans_and_metrics(capfire: CaptureLogfire):
    """Verify that executing a CLI command generates command spans."""
    setup_observability(send_to_logfire=False, console=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["labs"])
    assert result.exit_code == 0

    spans = capfire.exporter.exported_spans_as_dict()
    cli_spans = [
        s for s in spans
        if "cli.command.labs" in s.get("name", "")
    ]
    assert len(cli_spans) > 0
