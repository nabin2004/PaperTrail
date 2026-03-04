# API Reference

## `papertrail.schemas.schema`

### `Paper`
Pydantic model representing an arXiv paper.

| Field | Type | Description |
|---|---|---|
| `arxiv_id` | `str` | arXiv identifier (e.g. `2303.08774`) |
| `title` | `str` | Paper title |
| `abstract` | `str` | Full abstract |
| `authors` | `List[str]` | List of author names |
| `primary_category` | `str` | Primary arXiv category (e.g. `cs.LG`) |
| `categories` | `List[str]` | All categories |
| `published` | `datetime` | Publication date |
| `updated` | `datetime` | Last update date |
| `pdf_url` | `HttpUrl \| None` | Direct PDF link |
| `indexed` | `bool` | Whether this paper has been embedded+indexed |

### `SearchResult`
A single chunk retrieved from the vector store.

| Field | Type | Description |
|---|---|---|
| `paper_id` | `str` | arXiv ID of the source paper |
| `chunk_id` | `int` | Chunk index within the paper |
| `text` | `str` | Chunk text |
| `score` | `float` | Similarity score (higher = more relevant) |
| `title` | `str \| None` | Paper title (enriched from metadata) |
| `authors` | `List[str] \| None` | Author names |
| `published` | `str \| None` | Publication date string |

### `ResearchPlan`

| Field | Type | Description |
|---|---|---|
| `question` | `str` | Original research question |
| `concepts` | `List[str]` | Key concepts extracted |
| `queries` | `List[str]` | Search queries to run |
| `categories` | `List[str]` | Suggested arXiv categories |
| `paper_types` | `List[str]` | Expected paper types |
| `raw` | `str` | Raw LLM output |

### `ResearchReport`

| Field | Type | Description |
|---|---|---|
| `question` | `str` | Research question |
| `plan` | `ResearchPlan \| None` | Generated plan |
| `synthesis` | `str` | LLM-generated synthesis |
| `critique` | `str` | LLM-generated critique |
| `sources` | `List[SearchResult]` | Retrieved + reranked sources |
| `faithfulness_score` | `float` | 0–1 faithfulness estimate |
| `coverage_score` | `float` | 0–1 coverage estimate |
| `created_at` | `datetime` | Report timestamp |

---

## `papertrail.ingestion`

### `ArxivClient`
```python
client = ArxivClient(categories=["cs.AI", "cs.LG"], max_results=10)
papers: List[Paper] = client.fetch_papers()
```

### `pdf_loader`
```python
pdf_path = download_pdf(paper)                   # → Path | None
text_path = save_processed_text(pdf_path, clean=True)  # → Path
text = extract_text(pdf_path)                    # → str
```

### `metadata`
```python
save_metadata(paper)                # write JSON
paper = load_metadata("2303.08774") # read JSON → Paper | None
papers = load_all_metadata()        # List[Paper]
ids = list_paper_ids()              # List[str]
exists = metadata_exists("...")     # bool
```

---

## `papertrail.processing`

### `cleaners`
```python
cleaned = clean_text(raw_str)                 # → str
cleaned = clean_file(path)                    # → str
texts = clean_files([path1, path2])           # → List[str]
out = clean_and_save(src_path, dst_path)      # → Path
```

### `splitters`
```python
chunks_path = split_and_save("data/processed/2303.08774.txt")  # → Path (JSONL)
chunks = load_chunks("2303.08774")       # → List[Dict]
```

### `embeddings`
```python
vecs = embed_texts(["text1", "text2"])   # → np.ndarray (N, 384)
q_vec = embed_query("attention is all you need")  # → np.ndarray (384,)
dim = embedding_dim()                    # → 384
```

---

## `papertrail.retrieval`

### `VectorStore`
```python
store = VectorStore(index_name="papers", dim=384)

# Add embeddings + metadata dicts
store.add_chunks(embeddings, chunk_dicts)

# Search
results = store.search(query_vec, k=10)  # → List[(score, chunk_dict)]

# Introspection
store.total_chunks          # int
store.indexed_paper_ids()   # List[str]

# Maintenance
store.reset()               # wipe index + files
```

### `PaperRetriever`
```python
retriever = PaperRetriever(index_name="papers")
results: List[SearchResult] = retriever.retrieve("query text", k=10)
retriever.is_empty()   # bool
retriever.stats()      # {"total_chunks": ..., "indexed_papers": ...}
```

### `reranker`
```python
from papertrail.retrieval.reranker import rerank

top = rerank(query="...", results=search_results, top_k=5)
# → List[SearchResult] re-scored by cosine similarity
```

---

## `papertrail.memory`

### `PaperMemory`
```python
mem = PaperMemory()
mem.add(paper)
mem.add_many(papers)
p = mem.get("2303.08774")            # Paper | None
all_papers = mem.all()               # List[Paper]
by_cat = mem.search_by_category("cs.LG")
by_auth = mem.search_by_author("Vaswani")
```

### `TrendMemory`
```python
from papertrail.memory.trend_memory import TrendMemory

trends = TrendMemory()
trends.update(papers)                # ingest
kw = trends.top_keywords(20)         # List[(word, count)]
cats = trends.top_categories(10)     # List[(cat, count)]
print(trends.summary())              # human-readable summary
```

---

## `papertrail.chains`

All chain functions return `str` and require an LLM to be configured
(see `papertrail.utils.llm.get_llm`).

```python
from papertrail.chains.planning  import plan_research   # → ResearchPlan
from papertrail.chains.synthesis import synthesize      # → str
from papertrail.chains.critique  import critique        # → str

plan = plan_research("What is LoRA fine-tuning?")
synth = synthesize("What is LoRA?", results)
crit  = critique(synth, "What is LoRA?", results)
```

---

## `papertrail.agents`

### `ResearchAgent`
```python
from papertrail.agents.research_agent import ResearchAgent

agent = ResearchAgent(
    index_name="papers",
    retrieve_k=20,   # candidates before reranking
    rerank_k=6,      # final sources used for synthesis
)

# Full pipeline → ResearchReport
report = agent.research("How do diffusion models generate images?")

# Lightweight Q&A
answer = agent.ask("What is RLHF?")

# Status
agent.is_ready()   # bool
agent.stats()      # dict
```

---

## `papertrail.evaluation`

```python
from papertrail.evaluation.faithfulness import score_faithfulness
from papertrail.evaluation.coverage     import score_coverage

faith = score_faithfulness(synthesis_str, sources)  # float 0–1
cov   = score_coverage(question_str, sources)        # float 0–1
```

Both functions prefer LLM-based scoring and fall back to heuristics automatically.

---

## `papertrail.utils.llm`

```python
from papertrail.utils.llm import get_llm

llm = get_llm(temperature=0.3)         # cached; auto-selects OpenAI or Ollama
llm = get_llm(temperature=0.0, model="gpt-4o")
```

Priority: `OPENAI_API_KEY` → `OLLAMA_BASE_URL` → `ValueError`

---

## CLI

```
papertrail [OPTIONS] COMMAND [ARGS]...

Commands:
  ingest   Fetch arXiv papers and index them
  search   Semantic search
  ask      RAG question answering
  report   Full research report
  list     List indexed papers
  trends   Show trending keywords
  reset    Wipe all stored data
```

See `papertrail COMMAND --help` for per-command options.
