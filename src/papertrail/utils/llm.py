"""
LLM factory – supports OpenAI (default) or Ollama (local).

Priority:
1. OPENAI_API_KEY  →  ChatOpenAI
2. OLLAMA_BASE_URL →  ChatOllama (langchain-community)
3. Raises ValueError with helpful message
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.3, model: str | None = None) -> BaseChatModel:
    """Return a configured LLM instance (cached per temperature+model)."""
    _load_dotenv()

    # ── Try OpenAI ────────────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-..."):
        from langchain_openai import ChatOpenAI
        chosen = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=chosen, temperature=temperature, api_key=api_key)

    # ── Try Ollama ────────────────────────────────────────────────────────────
    ollama_url = os.getenv("OLLAMA_BASE_URL", "")
    if ollama_url:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
            chosen = model or os.getenv("OLLAMA_MODEL", "llama3.2")
            return ChatOllama(base_url=ollama_url, model=chosen, temperature=temperature)
        except ImportError:
            pass

    raise ValueError(
        "No LLM configured.\n"
        "  Option 1 – OpenAI:  set OPENAI_API_KEY in your .env file\n"
        "  Option 2 – Ollama:  run Ollama locally and set OLLAMA_BASE_URL=http://localhost:11434\n"
        "See .env.example for details."
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
