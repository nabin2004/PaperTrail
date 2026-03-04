# Cross-Encoder Reranker Upgrade

High-accuracy reranking using cross-encoder models for improved search relevance.

## Status

🔲 **Partially Implemented** – API available, requires `sentence-transformers`

## Overview

Cross-encoder models provide significantly better relevance scoring than bi-encoder cosine similarity by jointly encoding query-document pairs. This upgrade path enables higher-quality search results at the cost of additional computation.

## Files

- [retrieval/reranker.py](../src/papertrail/retrieval/reranker.py) – Reranker implementations

## Current Implementation

### Cosine Reranker (MVP)

The default reranker uses cosine similarity between query and document embeddings:

```python
from papertrail.retrieval.reranker import rerank

results = rerank(query, candidates, top_k=5, method="cosine")
```

**Pros:**
- Fast (embeddings are pre-computed)
- No additional model download
- Low memory usage

**Cons:**
- Lower accuracy than cross-encoders
- No cross-attention between query and document

### Cross-Encoder Reranker

```python
from papertrail.retrieval.reranker import rerank

# Use cross-encoder for higher accuracy
results = rerank(query, candidates, top_k=5, method="cross-encoder")
```

**Pros:**
- Significantly higher accuracy
- Joint query-document encoding
- State-of-the-art relevance scoring

**Cons:**
- Slower (O(n) model inference per query)
- Requires model download (~100MB+)
- Higher memory usage

## Configuration

### Environment Variables

```bash
# Set default reranker type
export RERANKER_TYPE=cross-encoder  # or "cosine"

# Choose cross-encoder model
export CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Available Models

| Model | Size | Speed | Quality |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90MB | Fast | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 120MB | Medium | Better |
| `BAAI/bge-reranker-base` | 278MB | Medium | Excellent |
| `BAAI/bge-reranker-large` | 560MB | Slow | Best |
| `mixedbread-ai/mxbai-rerank-base-v1` | 278MB | Medium | Excellent |

## Implementation Tasks

### Core

- [x] Cosine reranker (implemented)
- [x] Cross-encoder reranker skeleton
- [ ] Model caching and lazy loading
- [ ] GPU support detection
- [ ] Batch scoring optimization

### Advanced

- [ ] Hybrid reranking (cosine + cross-encoder)
- [ ] Score fusion strategies
- [ ] Reciprocal Rank Fusion (RRF)
- [ ] BM25 signal integration

### Cloud Alternatives

- [ ] Cohere Rerank API integration
- [ ] Jina Reranker API
- [ ] Voyage AI Reranker

## Usage Examples

### Basic Cross-Encoder

```python
from papertrail.retrieval.reranker import rerank

# Retrieve candidates with bi-encoder
candidates = retriever.retrieve(query, k=50)

# Rerank with cross-encoder
top_results = rerank(
    query=query,
    results=candidates,
    top_k=10,
    method="cross-encoder"
)
```

### Custom Model

```python
from papertrail.retrieval.reranker import _rerank_cross_encoder

results = _rerank_cross_encoder(
    query=query,
    results=candidates,
    top_k=10,
    model_name="BAAI/bge-reranker-large"
)
```

### Hybrid Reranking (TODO)

```python
from papertrail.retrieval.reranker import rerank_hybrid

results = rerank_hybrid(
    query=query,
    results=candidates,
    top_k=10,
    cosine_weight=0.3,
    cross_encoder_weight=0.7
)
```

## Benchmarks

Typical improvements on MS MARCO:

| Method | MRR@10 | Latency (100 docs) |
|---|---|---|
| Cosine (bi-encoder) | 0.32 | 5ms |
| MiniLM-L-6 (cross-encoder) | 0.39 | 150ms |
| BGE-reranker-base | 0.42 | 300ms |
| BGE-reranker-large | 0.44 | 600ms |

## Score Normalization

Cross-encoder scores have different ranges than cosine similarity:

```python
# Cosine: [-1, 1] (typically [0, 1] for positive embeddings)
# Cross-encoder: unbounded, typically [-10, 10]

# Normalize to [0, 1] for comparison
def normalize_scores(scores):
    min_s, max_s = min(scores), max(scores)
    return [(s - min_s) / (max_s - min_s + 1e-8) for s in scores]
```

## Dependencies

```toml
# Already in requirements
sentence-transformers = ">=3.0.0"

# Optional for cloud rerankers
cohere = ">=5.0.0"
```

## Architecture

```
Query + 50 Candidates
         │
         ▼
┌─────────────────────────┐
│   Cross-Encoder Model   │
│  (MiniLM / BGE / etc)   │
│                         │
│  For each (q, doc):     │
│    score = model(q, d)  │
└────────────┬────────────┘
             │
             ▼  Sort by score
┌─────────────────────────┐
│   Top-K Reranked        │
│   Results               │
└─────────────────────────┘
```

## References

- [Sentence-BERT Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MS MARCO Leaderboard](https://microsoft.github.io/MSMARCO-Passage-Ranking-Submissions/leaderboard/)
- [BGE Reranker (BAAI)](https://huggingface.co/BAAI/bge-reranker-base)
- [Cohere Rerank](https://docs.cohere.com/docs/rerank)
