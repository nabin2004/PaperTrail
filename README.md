# PaperTrail 🔬

> **AI-powered research paper discovery, analysis, and synthesis.**

PaperTrail automatically fetches papers from arXiv, builds a semantic search index, and lets you ask research questions to receive LLM-synthesised, citation-backed answers.

---

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

### 3. Ingest papers

```bash
# Fetch 10 cs.AI + cs.LG papers, download + index them
papertrail ingest --categories cs.AI,cs.LG --max-results 10

# Ingest more papers
papertrail ingest -c cs.CL,cs.CV -n 20
```

### 4. Search

```bash
papertrail search "transformer attention mechanism"
papertrail search "diffusion models image generation" --top-k 8
```

### 5. Ask a question

```bash
papertrail ask "What are the key differences between BERT and GPT architectures?"
```

### 6. Generate a full research report

```bash
papertrail report "How do large language models handle long context?" --output report.md
```

### 7. Explore trends

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
papertrail ingest   [--categories cs.AI,cs.LG] [--max-results 10]
papertrail search   QUERY [--top-k 5] [--no-rerank]
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
- **OpenAI / Ollama** – LLM synthesis (optional)

---

## Roadmap

- [ ] Streaming synthesis output
- [ ] Multi-modal support (figures, tables)
- [ ] Export to Obsidian / Notion
- [ ] Web UI (FastAPI + React)
- [ ] Scheduled paper ingestion (cron)
- [ ] Cross-encoder reranker upgrade
