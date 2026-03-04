# Streaming Synthesis

Real-time token-by-token synthesis output for responsive UI experiences.

## Status

🔲 **Not Implemented** – Skeleton available

## Overview

Streaming synthesis enables real-time display of generated answers as they're produced by the LLM, providing immediate feedback to users instead of waiting for the complete response.

## Files

- [chains/streaming.py](../src/papertrail/chains/streaming.py) – Streaming synthesis implementation

## Features to Implement

### Synchronous Streaming

```python
from papertrail.chains.streaming import stream_synthesis

for chunk in stream_synthesis("What is RLHF?", results):
    print(chunk, end="", flush=True)
```

### Async Streaming

```python
from papertrail.chains.streaming import astream_synthesis

async for chunk in astream_synthesis("What is RLHF?", results):
    await websocket.send(chunk)
```

### LangChain Integration

- Use LCEL's `.stream()` method on the synthesis chain
- Implement `StreamingCallbackHandler` for token-level callbacks
- Support cancellation via `StopIteration`

## Implementation Tasks

- [ ] Integrate with LangChain's streaming API
- [ ] Add async streaming support via `astream()`
- [ ] Implement `StreamingCallbackHandler` class
- [ ] Add WebSocket endpoint in API
- [ ] CLI streaming output with Rich Live display
- [ ] Handle backpressure for slow consumers
- [ ] Support streaming for critique chain
- [ ] Add token count tracking during stream

## API Design

### HTTP Server-Sent Events (SSE)

```
POST /ask/stream
Content-Type: application/json

{"question": "What is RLHF?", "top_k": 6}

Response (event-stream):
data: {"type": "token", "content": "Reinforcement"}
data: {"type": "token", "content": " Learning"}
data: {"type": "token", "content": " from"}
...
data: {"type": "sources", "sources": [...]}
data: {"type": "done"}
```

### WebSocket

```javascript
ws.send(JSON.stringify({question: "What is RLHF?"}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "token") {
    appendToOutput(data.content);
  }
};
```

## Dependencies

```toml
# No additional dependencies for basic streaming
# For WebSocket: fastapi, uvicorn
```

## References

- [LangChain Streaming](https://python.langchain.com/docs/expression_language/streaming)
- [FastAPI SSE](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
