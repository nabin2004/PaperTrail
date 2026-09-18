# PaperTrail 🔬

> **AI-powered research paper discovery, analysis, and synthesis.**

PaperTrail automatically fetches papers from arXiv, builds a semantic search index, and lets you ask research questions to receive LLM-synthesised, citation-backed answers.

---

[roadmap forward](goals.md)


## Features

| Capability | Description |
|---|---|
| **Ingest** | Fetch arXiv papers by category, download PDFs, extract + clean text, create embeddings |
| **Semantic Search** | Find relevant paper chunks with cosine similarity (FAISS) + reranking |
| **Q&A / Ask** | RAG-based question answering grounded in your paper corpus |
| **Research Report** | Full pipeline: plan → retrieve → synthesise → critique → evaluate |
| **Trends** | Surface trending keywords and categories across indexed papers |
| **Offline-first** | Embeddings run locally (no API key needed for ingest + search) |

---

## Quick Start

### 1. Install

```bash
# Clone / open the project
cd PaperTrail

# Install with pip (editable)
pip install -e .

# Or with uv
uv pip install -e .
```

### 2. Configure (optional – needed for synthesis)

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

> **No OpenAI key?**  Ingest and search work without any API key.  
> Synthesis commands fall back to raw excerpt compilation.

### 3. Discover & Ingest Papers from arXiv

You can search arXiv by topic, review titles and abstracts, and select which papers to download and index:

```bash
# Search arXiv directly and interactively choose papers to download:
papertrail arxiv "transformer attention mechanism"
# (or use the alias)
papertrail discover "diffusion models"

# Search arXiv via the search command with --arxiv:
papertrail search "transformer attention mechanism" --arxiv

# Non-interactive / scripted downloads:
papertrail arxiv "reinforcement learning from human feedback" --limit 5 --select 1,3
papertrail arxiv "state space models mamba" --limit 5 --download-all

# Batch ingest by category or topic query:
papertrail ingest --categories cs.AI,cs.LG --max-results 10
papertrail ingest --query "retrieval augmented generation" -n 5
```

> **Interactive Prompt Tips:**
> - Enter `1, 3-5` to select specific papers.
> - Enter `all` to download and index all retrieved papers.
> - Enter `a 1` to preview the abstract of paper #1 before deciding.
> - Enter `q` to cancel.

### 4. Search Local Papers

Once papers are indexed, search through them semantically:

```bash
papertrail search "transformer attention mechanism"
papertrail search "diffusion models image generation" --top-k 8
```

### 5. Autonomous AI Research Team (PydanticAI)

PaperTrail features a **4-tier academic research team** powered by PydanticAI. You primarily talk to the **Senior Researcher / Lead**, who delegates tasks down the hierarchy. You can also interact directly with any tier. See [docs/research_team.md](docs/research_team.md) for full details.

#### 🌟 Primary Contact: Senior Research Lead (Tier 4)
Steers research direction, reviews paper quality with peer-review rubrics, and orchestrates campaigns across the team:
```bash
# General inquiry to the Lead (orchestrates team behind the scenes):
papertrail lead "Should we pivot from pure Transformers to hybrid Mamba-Transformer models?"

# End-to-end multi-tier research campaign:
papertrail lead "State space models for genomic sequence modeling" --orchestrate

# Peer-review quality assessment on a paper:
papertrail lead "Attention Is All You Need" --review-quality
papertrail lead "Attention Is All You Need" --review-quality --json

# Strategic research roadmap:
papertrail lead "Sub-quadratic foundation models" --roadmap
```

#### 🔬 Tier 3: Researcher
Formulates testable scientific hypotheses, designs controlled empirical experiments, and interprets results against theory:
```bash
# Formulate testable hypothesis:
papertrail researcher "associative recall in recurrent linear models" --hypothesis

# Design controlled experiment protocol:
papertrail researcher "Mamba outperforms Transformers on synthetic induction heads" --experiment

# Interpret empirical results against theory:
papertrail researcher "Model achieved 94.2% at 32k, but dropped to 61.5% at 64k" --interpret
```

#### 🛠️ Tier 2: Research Assistant
Executes benchmarks, compiles comparative matrices, plans engineering implementations, and identifies literature gaps:
```bash
# Benchmark evaluation across models:
papertrail assistant "Transformer vs Mamba vs RWKV" --benchmark

# Side-by-side comparative literature matrix:
papertrail assistant "Sparse Attention vs Linear Attention" --compare

# Engineering implementation blueprint:
papertrail assistant "Bidirectional selective state space block" --implement

# Literature gap analysis & structured dossier:
papertrail assistant "Sub-quadratic attention mechanisms" --gaps
papertrail assistant "Long-context language models" --dossier
```

#### 📚 Tier 1: Research Intern
Searches literature, downloads & indexes papers from arXiv, assesses reproduction feasibility, and runs baseline checks:
```bash
# Broad literature inquiry:
papertrail intern "What papers do we have on attention, and what is missing?"

# Autonomous multi-step literature review:
papertrail intern "State space models vs transformers" --investigate

# Bulk download & index papers from arXiv:
papertrail intern "multi-modal state space models" --collect

# Reproduction feasibility check:
papertrail intern "2312.00752" --reproduce
```

### 6. Research Laboratories & Matrix Projects

PaperTrail organizes research into **10 specialized laboratories** spanning Core Domains, Cross-Cutting Methodologies, and Applications. Matrix projects combine domains and methodologies (e.g. `Agentic Intelligence × Foundation Model Post-Training`). See [docs/research_organization.md](docs/research_organization.md).

```bash
# List all 10 laboratories by category:
papertrail labs

# Direct inquiry to a specific laboratory:
papertrail lab lmi "How do vision-language models handle fine-grained spatial grounding?"
papertrail lab ai "What planning algorithms prevent looping in agent workflows?"
papertrail lab fmpt "Compare DPO vs GRPO for mathematical reasoning alignment."
papertrail lab isai "How does KV cache compression impact long-context inference throughput?"
papertrail lab aisl "How can agents generate verifiable Manim animations for education?"

# Launch a cross-lab matrix project (Domain x Methodologies x Applications):
papertrail matrix "Autonomous Physics Tutor" \
  --domain AI \
  --methodology FMPT,ISAI \
  --application AISL \
  --mission "Post-train an agent that generates interactive Manim physics explanations."
```

### 7. Ask a question & Generate reports

```bash
# Direct Q&A
papertrail ask "What are the key differences between BERT and GPT architectures?"

# Full research report with critique and grounding evaluations
papertrail report "How do large language models handle long context?" --output report.md
```

### 8. Explore trends & papers

```bash
papertrail trends
papertrail list
```

---

## Architecture

```
arXiv API
    │
    ▼
ArxivClient ──► Paper metadata (JSON)
    │
    ▼
PDF Download ──► data/raw_papers/
    │
    ▼
Text Extraction (PyMuPDF)
    │
    ▼
Text Cleaning ──► data/processed/
    │
    ▼
Chunking (LangChain RecursiveCharacterTextSplitter) ──► data/chunks/
    │
    ▼
Embeddings (sentence-transformers / all-MiniLM-L6-v2)
    │
    ▼
FAISS Index ──► data/indices/
    │
    ▼
PaperRetriever + Reranker
    │
    ▼
ResearchAgent (plan → retrieve → synthesise → critique → evaluate)
    │
    ▼
ResearchReport
```

See [docs/architecture.md](docs/architecture.md) for full details.

---

## Data Layout

```
data/
  raw_papers/     PDF files (arxiv_id.pdf)
  processed/      Cleaned text files (.txt)
  chunks/         JSONL chunk files (.jsonl)
  indices/        FAISS index + metadata
  metadata/       Paper metadata JSON files
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | – | OpenAI API key (for synthesis / ask / report) |
| `OPENAI_MODEL` | `gpt-4o-mini` | ChatOpenAI model name |
| `OLLAMA_BASE_URL` | – | Ollama server URL (local LLM alternative) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `DATA_DIR` | `data` | Root directory for all stored data |

---

## CLI Reference

```
papertrail arxiv    [QUERY] [--limit 10] [--sort relevance] [--select 1,2] [--download-all]
papertrail discover [QUERY] [--limit 10] [--sort relevance]
papertrail ingest   [--categories cs.AI,cs.LG] [--query QUERY] [--max-results 10]
papertrail search   QUERY [--top-k 5] [--rerank/--no-rerank] [--arxiv]
papertrail ask      QUESTION [--top-k 6]
papertrail report   QUESTION [--top-k 6] [--output report.md]
papertrail list
papertrail trends   [--top-n 20]
papertrail reset    [--yes]
```

---

## Development

```bash
# Run demo pipeline
python main.py

# Run tests
pytest tests/

# Install dev deps
pip install -e ".[dev]"
```

---

## Tech Stack

- **LangChain** – chains, prompt templates, output parsers
- **sentence-transformers** – local embeddings (`all-MiniLM-L6-v2`)
- **FAISS** – fast approximate nearest-neighbour search
- **PyMuPDF** – PDF text extraction
- **Click + Rich** – CLI and terminal UI
- **Pydantic v2** – data validation and schemas
- **PydanticAI** – hierarchical agent execution, tool calling, and structured outputs
- **Pydantic Logfire** – end-to-end tracing, model tracking, spans, and metrics
- **OpenAI / Ollama** – LLM synthesis (optional)

---

## Observability & Tracing

PaperTrail integrates [Pydantic Logfire](https://logfire.pydantic.dev/) for production-grade observability across all commands, agents, specialized labs, retrieval pipelines, and LLM calls.

- **Zero configuration**: Runs out of the box with `send_to_logfire="if-token-present"`.
- **Model Name Tracking**: Automatically records exact model names (`gpt-4o-mini`, `deepseek`, Ollama models) on all spans and metric events.
- **Hierarchical Agent Spans**: Tracks delegation between Senior Lead, Researchers, Assistants, Interns, and 10 Research Labs.
- **Custom Metrics**: Collects CLI command usage, agent executions, indexing throughput, and retrieval latency histograms.

For full setup instructions, cloud dashboard viewing, and metric schemas, see [`docs/observability.md`](docs/observability.md).  
For developer instructions on instrumenting new tools, agents, roles, or labs, see [`docs/logfire_developer_guide.md`](docs/logfire_developer_guide.md).

---

## Roadmap

Skeleton implementations and documentation are available for these planned features:

| Feature | Status | Docs | Code |
|---|---|---|---|
| Streaming synthesis output | 🔲 Skeleton | [streaming.md](docs/features/streaming.md) | [chains/streaming.py](src/papertrail/chains/streaming.py) |
| Multi-modal support (figures, tables) | 🔲 Skeleton | [multimodal.md](docs/features/multimodal.md) | [processing/multimodal.py](src/papertrail/processing/multimodal.py) |
| Export to Obsidian / Notion | 🔲 Skeleton | [export.md](docs/features/export.md) | [export/](src/papertrail/export/) |
| Web UI (FastAPI + React) | 🔲 Skeleton | [web-ui.md](docs/features/web-ui.md) | [api/main.py](src/papertrail/api/main.py) |
| Scheduled paper ingestion | 🔲 Skeleton | [scheduler.md](docs/features/scheduler.md) | [scheduler/jobs.py](src/papertrail/scheduler/jobs.py) |
| Cross-encoder reranker | 🔲 Skeleton | [cross-encoder.md](docs/features/cross-encoder.md) | [retrieval/reranker.py](src/papertrail/retrieval/reranker.py) |
