# PaperTrail Observability with Pydantic Logfire

PaperTrail incorporates enterprise-grade observability powered by [Pydantic Logfire](https://logfire.pydantic.dev/). Every layer of the system—from CLI commands to multi-tier autonomous agents, specialized research laboratories, vector retrieval, and arXiv ingestion—is instrumented with OpenTelemetry-compliant spans, structured logs, and custom metrics.

---

## 1. Core Architecture

PaperTrail's observability is initialized via `src/papertrail/utils/observability.py`:

```
┌─────────────────────────────────────────────────────────────┐
│                     PaperTrail CLI Root                     │
│                  (setup_observability)                      │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
       ┌────────┴─────────┐          ┌────────┴─────────┐
       │   CLI Commands   │          │  Auto-Instrument │
       │  (Command Spans) │          │  (PydanticAI,    │
       └────────┬─────────┘          │   OpenAI, HTTP)  │
                │                    └──────────────────┘
       ┌────────┴────────────────────────────────────────┐
       │             Multi-Agent Hierarchy               │
       │  • Senior Research Lead (LeadResearcher)       │
       │  • Researcher (ScientistAgent)                  │
       │  • Research Assistant (ResearchAssistant)       │
       │  • Research Intern (ResearchIntern)             │
       │  • 10 Specialized Labs (ResearchLabAgent)       │
       │  • Matrix Coordinator (MatrixProjectCoord)      │
       └────────┬────────────────────────────────────────┘
                │
       ┌────────┴────────────────────────────────────────┐
       │              Data & Ingestion Engine            │
       │  • ChromaDB Hybrid Retrieval (Histograms)       │
       │  • arXiv API Ingestion (Rate-limit & Counters)  │
       └─────────────────────────────────────────────────┘
```

### Key Principles

1. **Zero-Friction Default (`if-token-present`)**:
   PaperTrail automatically detects whether a `LOGFIRE_TOKEN` is configured. If no token is provided, PaperTrail logs locally to the console and continues running without raising errors or blocking offline workflows.
2. **Explicit Model Name Provenance**:
   Every agent and LLM call extracts and stamps the model identifier (e.g., `openai:gpt-4o-mini`, `deepseek/deepseek-chat`, `ollama:llama3`) onto spans and metric labels.
3. **Structured Logging (`{key}` placeholders)**:
   All events use templated message templates and named keyword arguments rather than interpolated f-strings, enabling Logfire to aggregate and index span attributes dynamically.
4. **Library Auto-Instrumentation**:
   PaperTrail instruments `pydantic_ai`, `openai`, `httpx`, `requests`, and Pydantic models automatically.

---

## 2. Quickstart & Configuration

### Enabling Logfire Cloud

To stream traces, logs, and metrics to your Logfire web dashboard:

1. Authenticate with Logfire:
   ```bash
   uv run logfire auth
   ```
2. Or set the environment variable in your `.env` or terminal:
   ```bash
   LOGFIRE_TOKEN=your_write_token_here
   ```

### Running CLI Commands

Every CLI command is automatically recorded under a span named `cli.command.<name>` with execution parameters, model name, and exit codes:

```bash
# Explore research laboratories
uv run papertrail labs

# Query the Lead Researcher with full delegation tracing
uv run papertrail lead ask "What are current breakthroughs in test-time compute scaling?"

# Run multi-lab matrix collaboration
uv run papertrail matrix --domain LMI --methodology FMPT "Scaling laws for reasoning tokens"
```

---

## 3. Model Name Tracking

Every component resolves and exposes its `.model_name` attribute via `extract_model_name()`:

| Component | Default Model | Logfire Attribute Key | Description |
| :--- | :--- | :--- | :--- |
| **Senior Research Lead** (`LeadResearcher`) | `OPENAI_MODEL` (`gpt-4o-mini`) | `model`, `agent` | Strategic coordination, delegation, and review |
| **Researcher** (`ScientistAgent`) | `OPENAI_MODEL` (`gpt-4o-mini`) | `model`, `agent` | Hypotheses, experimental design, and interpretation |
| **Research Assistant** (`ResearchAssistant`) | `OPENAI_MODEL` (`gpt-4o-mini`) | `model`, `agent` | Gap analysis, benchmarking, implementation plans |
| **Research Intern** (`ResearchIntern`) | `OPENAI_MODEL` (`gpt-4o-mini`) | `model`, `agent` | Literature search, data collection, baseline runs |
| **Research Labs** (`ResearchLabAgent`) | `OPENAI_MODEL` (`gpt-4o-mini`) | `model`, `lab_code`, `category` | Deep domain / methodology expertise across 10 labs |
| **Matrix Coordinator** (`MatrixProjectCoordinator`) | `OPENAI_MODEL` (`gpt-4o-mini`) | `model`, `domain_lab`, `method_lab` | Cross-domain cross-methodology collaboration |

When models are instantiated via `src/papertrail/utils/llm.py`, Logfire records an event `llm.provider.initialized` with the target provider, model name, temperature, and streaming status.

---

## 4. Multi-Agent Tracing

When you invoke the Senior Research Lead or trigger inter-lab collaboration, Logfire captures the complete call tree:

```
span: agent.senior_research_lead.run (model="gpt-4o-mini")
 ├── span: tool.task_researcher
 │    └── span: agent.scientist.formulate_hypothesis (model="gpt-4o-mini")
 ├── span: tool.consult_specialized_lab (lab_code="FMPT")
 │    └── span: lab.inquiry.FMPT (model="gpt-4o-mini")
 └── span: tool.task_research_assistant
      └── span: agent.assistant.plan_implementation (model="gpt-4o-mini")
```

Within each span, you can inspect:
- **Input parameters & prompt previews**
- **Token usage & LLM latencies**
- **Tool calls and outputs**
- **Exceptions and self-correcting fallbacks**

---

## 5. Metrics Catalog

PaperTrail registers OpenTelemetry metrics under the `papertrail` namespace:

| Metric Name | Type | Description | Attributes |
| :--- | :--- | :--- | :--- |
| `papertrail_cli_commands_total` | Counter | Total invocations of PaperTrail CLI commands | `command`, `status` |
| `papertrail_agent_runs_total` | Counter | Total executions across all research agents | `agent`, `model`, `action` |
| `papertrail_papers_indexed_total` | Counter | Number of academic papers processed & embedded | `status` |
| `papertrail_retrieval_duration_seconds` | Histogram | Latency distribution of vector & hybrid retrieval | `query_length` |

---

## 6. Testing & Verifying Observability

PaperTrail includes automated tests for all observability features using Logfire's `CaptureLogfire` testing fixture:

```bash
# Run observability test suite
uv run pytest tests/test_observability.py -v
```

This verifies:
- Idempotent configuration of `setup_observability()`
- Model name extraction across string models, custom OpenAI models, and fallback defaults
- Agent hierarchy model provenance
- OpenTelemetry span emission for agents and specialized labs
- CLI command spans and metric increments

---

## 7. Adding New Tools, Agents, or Labs

For guidelines, code patterns, and best practices when extending PaperTrail with new tools, agent tiers, CLI roles, or research labs, please read the [Logfire Developer Guide](logfire_developer_guide.md).

