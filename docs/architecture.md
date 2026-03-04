# Architecture Overview

## High-Level Design

PaperTrail is organised into six vertical layers that form a pipeline from raw data ingestion to synthesised research output.

```
┌─────────────────────────────────────────────────────────┐
│                        CLI / API                        │
│              (papertrail.cli  ·  main.py)               │
├──────────────┬──────────────────┬───────────────────────┤
│   Ingestion  │    Processing    │      Retrieval        │
│  arxiv_client│   cleaners       │   vectorstore (FAISS) │
│  pdf_loader  │   embeddings     │   retrievers          │
│  metadata    │   splitters      │   reranker            │
├──────────────┴──────────────────┴───────────────────────┤
│                     Agent Layer                         │
│         ResearchAgent  ·  tools (LangChain)             │
├─────────────────────────────────────────────────────────┤
│                     Chain Layer                         │
│       PlanningChain · SynthesisChain · CritiqueChain    │
├──────────────┬──────────────────┬───────────────────────┤
│    Memory    │   Evaluation     │      Schemas          │
│  PaperMemory │  faithfulness    │  Paper · ChunkRecord  │
│  TrendMemory │  coverage        │  SearchResult · Report│
└──────────────┴──────────────────┴───────────────────────┘
```

---

## Layer Details

### 1. Ingestion (`src/papertrail/ingestion/`)

| Module | Responsibility |
|---|---|
| `arxiv_client.py` | Queries the arXiv Atom API; parses XML into `Paper` objects |
| `pdf_loader.py` | Downloads PDFs via HTTPS, extracts text with PyMuPDF |
| `metadata.py` | Persists `Paper` objects as JSON for fast offline lookup |

**Data flow:**
```
ArxivClient.fetch_papers()
    → List[Paper]
    → download_pdf(paper)        → data/raw_papers/<id>.pdf
    → save_processed_text(pdf)   → data/processed/<id>.txt
    → save_metadata(paper)       → data/metadata/<id>.json
```

### 2. Processing (`src/papertrail/processing/`)

| Module | Responsibility |
|---|---|
| `cleaners.py` | Unicode normalisation, whitespace collapse, de-hyphenation |
| `splitters.py` | `RecursiveCharacterTextSplitter` (500 chars / 100 overlap) → JSONL |
| `embeddings.py` | HuggingFace `all-MiniLM-L6-v2` via `langchain-huggingface`; returns `np.ndarray` |

**Embedding model:** `all-MiniLM-L6-v2`
- Dimension: 384
- Normalised: L2 (cosine similarity = inner product)
- Approx size: ~90 MB (downloaded once to `~/.cache/huggingface`)

### 3. Retrieval (`src/papertrail/retrieval/`)

| Module | Responsibility |
|---|---|
| `vectorstore.py` | `VectorStore` wraps FAISS `IndexFlatIP`; persists to `data/indices/` |
| `retrievers.py` | `PaperRetriever` embeds query, calls `VectorStore.search`, enriches with metadata |
| `reranker.py` | Cosine `rerank()` – re-scores results with per-chunk embedding vs query |

**FAISS index type:** `IndexFlatIP` (exact inner-product, no approximation)  
Suitable for up to ~100k chunks; swap to `IndexIVFFlat` for larger corpora.

### 4. Memory (`src/papertrail/memory/`)

| Module | Responsibility |
|---|---|
| `paper_memory.py` | In-session `dict[arxiv_id → Paper]` cache |
| `trend_memory.py` | Keyword + category counter over all ingested papers |

Memory is in-RAM only (no persistence) – it lives for the duration of a CLI command or agent session.

### 5. Chains (`src/papertrail/chains/`)

All chains use **LCEL** (LangChain Expression Language):  
`ChatPromptTemplate | LLM | StrOutputParser`

| Chain | Temperature | Purpose |
|---|---|---|
| `planning.py` | 0.2 | Generates structured JSON search plan from a research question |
| `synthesis.py` | 0.3 | Synthesises retrieved chunks into a cited answer |
| `critique.py` | 0.2 | Identifies hallucinations, gaps, and inconsistencies |

All chains **gracefully degrade** when no LLM is configured:  
- Planning → keyword extraction  
- Synthesis → formatted excerpt compilation  
- Critique → heuristic word-count feedback  

### 6. Agent (`src/papertrail/agents/`)

`ResearchAgent` orchestrates the full pipeline:

```
question
  ├─ PlanningChain.plan_research(question)      → ResearchPlan
  ├─ PaperRetriever.retrieve(q, k=20)           → List[SearchResult]  (per query)
  ├─ rerank(question, all_results, top_k=6)     → List[SearchResult]
  ├─ SynthesisChain.synthesize(question, top)   → str
  ├─ CritiqueChain.critique(synth, ...)         → str
  ├─ score_faithfulness(synth, top)             → float
  ├─ score_coverage(question, top)              → float
  └─ ResearchReport(...)
```

`tools.py` wraps agent capabilities as LangChain `BaseTool` objects for future ReAct/function-calling integration.

### 7. Evaluation (`src/papertrail/evaluation/`)

| Module | Score Range | Primary Method | Fallback |
|---|---|---|---|
| `faithfulness.py` | 0–1 | LLM scoring prompt | Bigram overlap |
| `coverage.py` | 0–1 | LLM scoring prompt | Keyword coverage |

---

## Dependency Graph

```
schemas ◄──────────────────────────────────┐
   ▲                                        │
   │                                        │
ingestion ──► processing ──► retrieval ──► agents
                                │               │
                             memory          chains
                                │               │
                             evaluation ◄────────┘
```

---

## Configuration

All path constants respect the `DATA_DIR` environment variable (defaults to `./data`).  
The LLM factory in `utils/llm.py` checks `OPENAI_API_KEY` then `OLLAMA_BASE_URL` in order.

---

## Scalability Notes

| Bottleneck | Current | Upgrade path |
|---|---|---|
| Index type | `IndexFlatIP` (exact) | `IndexIVFFlat` for >100k chunks |
| Embeddings | Local CPU | GPU batch mode via `encode_kwargs` |
| LLM | Single call | Streaming with `| stream` in LCEL chain |
| Storage | Local JSON | SQLite / PostgreSQL + `pgvector` |
