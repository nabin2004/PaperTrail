# Developer Guide: Observability with Logfire in PaperTrail

This guide explains how developers should instrument new **tools**, **agents**, **roles**, and **departments/labs** using [Pydantic Logfire](https://logfire.pydantic.dev/) in PaperTrail.

---

## 1. Golden Rules of Logfire in PaperTrail

1. **Never use f-strings in `logfire.*` calls**:
   ```python
   # ❌ Bad - suppresses Logfire's template grouping and parameter indexing
   logfire.info(f"Agent {agent_name} called tool {tool_name}")

   # ✅ Good - uses template with named kwargs
   logfire.info("Agent {agent} called tool {tool}", agent=agent_name, tool=tool_name)
   ```
2. **Always stamp the model name**:
   Every agent and LLM call must include the resolved model identifier (e.g. `gpt-4o-mini`, `deepseek/deepseek-chat`, `ollama:llama3`). Use `extract_model_name(model)`.
3. **Use OpenTelemetry Spans for units of work**:
   Wrap user-facing workflows, agent reasoning, delegation steps, and I/O operations in `with logfire.span(...)`.
4. **Make agent delegation tools `async def`**:
   PydanticAI disallows nested synchronous agent runs. All delegation tools must be `async def` and `await subagent.run(...)`.

---

## 2. Adding a New Tool

When creating a new tool (e.g. in `src/papertrail/agents/tools.py` or within an agent definition):

### Step-by-Step Template

```python
import logfire
from pydantic_ai import RunContext

# 1. Plain Tool (does not need agent context)
def analyze_dataset_tool(dataset_path: str, max_samples: int = 100) -> str:
    """Analyze a dataset file and return descriptive statistics."""
    with logfire.span(
        "tool.analyze_dataset path={path} max_samples={max_samples}",
        path=dataset_path,
        max_samples=max_samples,
    ):
        try:
            # Tool execution logic here
            result = f"Summary of {dataset_path} with {max_samples} samples"
            logfire.info("Dataset analysis completed for {path}", path=dataset_path)
            return result
        except Exception as e:
            logfire.exception("Failed to analyze dataset {path}: {error}", path=dataset_path, error=str(e))
            return f"Error analyzing dataset: {e}"


# 2. Context-Aware Tool (needs agent dependencies or run context)
def fetch_paper_citations_tool(ctx: RunContext[None], paper_id: str) -> list[str]:
    """Fetch citation list for a given paper ID."""
    with logfire.span("tool.fetch_paper_citations paper_id={paper_id}", paper_id=paper_id):
        # Implementation...
        citations = ["2305.18290", "1706.03762"]
        logfire.debug("Fetched {count} citations for paper {paper_id}", count=len(citations), paper_id=paper_id)
        return citations
```

### Checklist for Tools
- [ ] Added a descriptive docstring (PydanticAI passes this to the LLM).
- [ ] Wrapped critical I/O, search, or compute in `with logfire.span(...)`.
- [ ] Handled errors gracefully so the tool returns an error string rather than crashing the agent loop.
- [ ] Used structured keyword arguments with `{key}` placeholders.

---

## 3. Adding a New Agent or Role

When adding a new research agent or role (e.g. `DataEngineerAgent`, `EthicsAuditorAgent`):

### Architecture Overview

```
Client / CLI
   │
   ▼
[NewAgent.run / run_sync] ────────► logfire.span("agent.<name>.run", model=..., agent=...)
   │                                  │
   ├── PydanticAI Agent Execution ────┤ (Auto-instrumented LLM calls & token counts)
   │                                  │
   ├── Tools / Delegations ───────────┴─► logfire.span("tool.<name>", ...)
   │
   ▼
agent_runs_counter.add(1, {"agent": ..., "model": ..., "action": ...})
```

### Implementation Template

```python
from typing import Optional
import logfire
from pydantic_ai import Agent
from pydantic_ai.models import Model
from papertrail.utils.llm import get_pydantic_ai_model, extract_model_name
from papertrail.utils.observability import agent_runs_counter

SYSTEM_PROMPT = """You are the Data Engineer Agent in PaperTrail..."""

class DataEngineerAgent:
    """Autonomous agent responsible for dataset extraction and preprocessing."""

    def __init__(self, model: Model | str | None = None) -> None:
        self.model = model
        self.model_name = extract_model_name(model)
        
        # Resolve model instance
        if model is None:
            try:
                resolved_model = get_pydantic_ai_model()
            except Exception:
                resolved_model = "test"
        else:
            resolved_model = model

        # Create the PydanticAI agent
        self.agent = Agent(
            model=resolved_model,
            name="data_engineer",
            instructions=SYSTEM_PROMPT,
            tools=[analyze_dataset_tool],
        )

    def run_sync(self, task: str) -> str:
        """Synchronous execution with Logfire span tracking."""
        with logfire.span(
            "agent.data_engineer.run task={task} model={model}",
            task=task[:80],
            model=self.model_name,
            agent="data_engineer",
        ):
            agent_runs_counter.add(1, {
                "agent": "data_engineer",
                "model": self.model_name,
                "action": "run_sync"
            })
            result = self.agent.run_sync(task)
            return str(result.output)

    async def run(self, task: str) -> str:
        """Asynchronous execution with Logfire span tracking."""
        with logfire.span(
            "agent.data_engineer.run_async task={task} model={model}",
            task=task[:80],
            model=self.model_name,
            agent="data_engineer",
        ):
            agent_runs_counter.add(1, {
                "agent": "data_engineer",
                "model": self.model_name,
                "action": "run_async"
            })
            result = await self.agent.run(task)
            return str(result.output)
```

### Adding Delegation to Existing Agents

If another agent (like `LeadResearcher`) delegates to your new agent:

```python
# In lead_agent.py:
_data_engineer = DataEngineerAgent(model=resolved_model)

@agent.tool_plain
async def task_data_engineer(task: str) -> str:
    """Delegate dataset extraction or preprocessing tasks to the Data Engineer."""
    with logfire.span("lead.delegate.data_engineer task={task}", task=task[:80]):
        return await _data_engineer.run(task)
```

> [!IMPORTANT]
> Always define delegation tools with `async def` and `await subagent.run(...)`. Calling `run_sync` inside a tool triggers PydanticAI's `check_no_nested_sync_run` error.

---

## 4. Adding a New Department or Research Laboratory

PaperTrail organizes domain expertise into specialized research laboratories. When creating a new department or lab (e.g. `QAI` - Quantum AI Lab):

### Step 1: Register in `src/papertrail/labs/registry.py`

Add the laboratory to `LAB_CATALOG`:

```python
"QAI": LabInfo(
    code="QAI",
    full_name="Quantum Artificial Intelligence Lab",
    short_name="Quantum AI (QAI) Lab",
    category="domain",  # or "methodology" or "application"
    focus=(
        "Quantum machine learning, variational quantum circuits, tensor networks, "
        "and hybrid quantum-classical algorithms for optimization and chemistry."
    ),
    key_topics=[
        "quantum machine learning",
        "variational quantum algorithms",
        "quantum neural networks",
        "tensor network contractions",
    ],
),
```

### Step 2: Verification of Automatic Observability

`ResearchLabAgent` in `src/papertrail/labs/lab_agent.py` automatically picks up any lab registered in `LAB_CATALOG`. It instruments every lab consultation with:
- `lab_code`: e.g. `"QAI"`
- `category`: `"domain"`, `"methodology"`, or `"application"`
- `model`: resolved model name
- `interlab_request_type`: domain / methodology request classification

When testing:
```python
lab = ResearchLabAgent(lab_code="QAI")
# Automatically emits span: lab.inquiry.QAI with model=gpt-4o-mini
response = lab.run_sync("Formulate quantum circuit ansatz for combinatorial optimization")
```

---

## 5. Adding a CLI Command for the New Role or Feature

When exposing your agent or feature in `src/papertrail/cli.py`:

```python
from papertrail.utils.observability import cli_command_counter

@cli.command("data-engineer")
@click.argument("task")
@click.option("--model", default=None, help="LLM model override.")
def data_engineer_command(task: str, model: Optional[str]):
    """Run data engineering workflows with full Logfire telemetry."""
    with logfire.span(
        "cli.command.data_engineer task={task} model={model}",
        task=task[:80],
        model=model or "default",
    ) as span:
        cli_command_counter.add(1, {"command": "data_engineer", "status": "invoked"})
        try:
            agent = DataEngineerAgent(model=model)
            span.set_attribute("resolved_model", agent.model_name)
            
            output = agent.run_sync(task)
            cli_command_counter.add(1, {"command": "data_engineer", "status": "success"})
            console.print(Panel(output, title="Data Engineer Report"))
        except Exception as e:
            cli_command_counter.add(1, {"command": "data_engineer", "status": "failed"})
            logfire.exception("CLI data-engineer failed: {error}", error=str(e))
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise click.Abort()
```

---

## 6. Testing Your Observability

Always write an automated test using the `capfire` fixture in `tests/test_observability.py`:

```python
def test_new_agent_logfire_span(capfire: CaptureLogfire):
    """Verify that the new agent emits OpenTelemetry spans with model name."""
    setup_observability(send_to_logfire=False, console=False)

    agent = DataEngineerAgent(model="test")
    res = agent.run_sync("Extract metadata from CSV")
    assert isinstance(res, str)

    spans = capfire.exporter.exported_spans_as_dict()
    matching = [
        s for s in spans
        if s.get("attributes", {}).get("agent") == "data_engineer"
    ]
    assert len(matching) > 0
    assert matching[0]["attributes"].get("model") == "test"
```

Run tests to ensure compliance:
```bash
uv run pytest tests/test_observability.py -v
```
