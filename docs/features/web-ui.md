# Web UI (FastAPI + React)

Modern web interface for PaperTrail with API backend and React frontend.

## Status

🔲 **Not Implemented** – Skeleton available

## Overview

A web-based interface that provides a modern, responsive UI for paper discovery, search, and research synthesis with real-time streaming.

## Files

- [api/main.py](../src/papertrail/api/main.py) – FastAPI backend skeleton

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                      │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐  │
│  │ Search │  │ Papers │  │  Ask   │  │ Reports  │  │
│  └────────┘  └────────┘  └────────┘  └──────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────▼──────────────────────────────┐
│                 FastAPI Backend                      │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐  │
│  │ Routes │  │  Auth  │  │  Jobs  │  │ WebSock  │  │
│  └────────┘  └────────┘  └────────┘  └──────────┘  │
├─────────────────────────────────────────────────────┤
│              PaperTrail Core Library                 │
└─────────────────────────────────────────────────────┘
```

## Backend Implementation Tasks

### Core API

- [ ] FastAPI application setup
- [ ] CORS configuration for React
- [ ] Request/response Pydantic models
- [ ] Error handling middleware
- [ ] API versioning (/api/v1/)

### Endpoints

- [ ] `GET /papers` – List papers with pagination
- [ ] `GET /papers/{id}` – Paper details
- [ ] `POST /search` – Semantic search
- [ ] `POST /ask` – RAG question answering
- [ ] `POST /ask/stream` – Streaming SSE response
- [ ] `POST /report` – Generate research report
- [ ] `POST /ingest` – Trigger ingestion
- [ ] `GET /trends` – Trending keywords
- [ ] `GET /stats` – System statistics

### WebSocket

- [ ] `/ws/ingest` – Ingestion progress
- [ ] `/ws/ask` – Streaming Q&A
- [ ] Connection management
- [ ] Heartbeat/ping-pong

### Background Tasks

- [ ] Ingestion job queue
- [ ] Progress tracking
- [ ] Job status endpoint

### Authentication (Optional)

- [ ] API key authentication
- [ ] JWT tokens
- [ ] Rate limiting

## Frontend Implementation Tasks

### Setup

- [ ] Vite + React + TypeScript
- [ ] TailwindCSS styling
- [ ] React Query for data fetching
- [ ] React Router for navigation

### Pages

- [ ] **Dashboard** – Stats, recent papers, quick search
- [ ] **Search** – Search interface with filters
- [ ] **Papers** – Paper list with details modal
- [ ] **Ask** – Q&A interface with streaming
- [ ] **Reports** – Report generation and history
- [ ] **Settings** – Configuration, API keys

### Components

- [ ] SearchBar with autocomplete
- [ ] PaperCard with metadata
- [ ] MarkdownRenderer for synthesis
- [ ] StreamingText for real-time output
- [ ] SourcesList with citations
- [ ] TrendChart for keywords

## API Examples

### Search Papers

```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "attention mechanism transformers",
  "top_k": 10,
  "rerank": true
}

Response:
{
  "results": [
    {
      "paper_id": "1706.03762",
      "title": "Attention Is All You Need",
      "text": "...",
      "score": 0.89,
      "authors": ["Vaswani et al."]
    }
  ],
  "total": 1
}
```

### Streaming Q&A

```http
POST /api/v1/ask/stream
Content-Type: application/json

{
  "question": "What is RLHF?",
  "top_k": 6
}

Response (text/event-stream):
data: {"type": "token", "content": "RLHF"}
data: {"type": "token", "content": " stands"}
data: {"type": "token", "content": " for"}
...
data: {"type": "sources", "sources": [...]}
data: {"type": "done"}
```

## Project Structure

```
papertrail/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── routes/
│   │   ├── papers.py
│   │   ├── search.py
│   │   ├── ask.py
│   │   └── ingest.py
│   ├── models/
│   │   ├── requests.py
│   │   └── responses.py
│   └── websocket/
│       └── handlers.py
│
frontend/                  # Separate repo or monorepo
├── src/
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   ├── api/
│   └── App.tsx
├── package.json
└── vite.config.ts
```

## Running the Stack

### Backend

```bash
# Install API dependencies
pip install fastapi uvicorn python-multipart

# Run development server
uvicorn papertrail.api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - api
```

## Dependencies

### Backend

```toml
fastapi = ">=0.111.0"
uvicorn = ">=0.30.0"
python-multipart = ">=0.0.9"
websockets = ">=12.0"
```

### Frontend

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.23.0",
    "@tanstack/react-query": "^5.40.0",
    "axios": "^1.7.0"
  },
  "devDependencies": {
    "vite": "^5.2.0",
    "typescript": "^5.4.0",
    "tailwindcss": "^3.4.0"
  }
}
```

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [TanStack Query](https://tanstack.com/query/latest)
- [Vite](https://vitejs.dev/)
