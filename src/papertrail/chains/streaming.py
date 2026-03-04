"""
Streaming synthesis – yields tokens incrementally for real-time UI display.

TODO: Implement streaming synthesis chain using LCEL streaming.
"""
from __future__ import annotations

from typing import AsyncIterator, Iterator, List

from papertrail.schemas.schema import SearchResult


# ──────────────────────────────────────────────────────────────
# Synchronous streaming
# ──────────────────────────────────────────────────────────────

def stream_synthesis(
    question: str,
    results: List[SearchResult],
    chunk_size: int = 20,
) -> Iterator[str]:
    """
    Stream a synthesis answer token-by-token.

    Parameters
    ----------
    question : str
        The research question to answer.
    results : List[SearchResult]
        Retrieved paper chunks to ground the answer.
    chunk_size : int
        Approximate number of characters per yielded chunk.

    Yields
    ------
    str
        Incremental text chunks as they are generated.

    Example
    -------
    >>> for chunk in stream_synthesis("What is RLHF?", results):
    ...     print(chunk, end="", flush=True)

    TODO
    ----
    - Integrate with LangChain's `.stream()` on the synthesis chain
    - Add token-level callbacks for precise streaming
    - Support cancellation via StopIteration
    """
    # Stub: fall back to non-streaming synthesis
    from papertrail.chains.synthesis import synthesize

    full_response = synthesize(question, results)

    # Simulate streaming by chunking the response
    for i in range(0, len(full_response), chunk_size):
        yield full_response[i : i + chunk_size]


# ──────────────────────────────────────────────────────────────
# Async streaming
# ──────────────────────────────────────────────────────────────

async def astream_synthesis(
    question: str,
    results: List[SearchResult],
) -> AsyncIterator[str]:
    """
    Async streaming synthesis for FastAPI / WebSocket integration.

    TODO
    ----
    - Use `chain.astream()` for true async token streaming
    - Integrate with SSE (Server-Sent Events) endpoint
    - Add backpressure handling for slow clients
    """
    # Stub: wrap sync version
    for chunk in stream_synthesis(question, results):
        yield chunk


# ──────────────────────────────────────────────────────────────
# Streaming callback handler (for LangChain integration)
# ──────────────────────────────────────────────────────────────

class StreamingCallbackHandler:
    """
    LangChain callback handler for capturing streaming tokens.

    TODO
    ----
    - Inherit from BaseCallbackHandler
    - Implement on_llm_new_token() for real-time token capture
    - Add queue-based buffering for async consumers
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.tokens.append(token)

    def get_full_response(self) -> str:
        return "".join(self.tokens)

    def clear(self) -> None:
        self.tokens.clear()
