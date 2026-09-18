"""
LLM factory – supports OpenAI, Groq, OpenRouter, and Ollama.

Priority:
1. OPENAI_API_KEY (+ optional OPENAI_BASE_URL)  →  OpenAI / Groq / OpenRouter
2. OLLAMA_BASE_URL →  Ollama (local)
3. Raises ValueError with helpful message
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.3, model: str | None = None) -> BaseChatModel:
    """Return a configured LangChain LLM instance (cached per temperature+model)."""
    _load_dotenv()

    # ── Try OpenAI / Compatible (Groq, OpenRouter) ────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-..."):
        from langchain_openai import ChatOpenAI
        chosen = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        kwargs: dict[str, Any] = {
            "model": chosen,
            "temperature": temperature,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

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


def get_pydantic_ai_model(model: str | None = None):
    """
    Return a PydanticAI-compatible Model based on environment configuration.
    Supports OpenAI, Groq, OpenRouter, and Ollama.
    """
    _load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    chosen = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key and not api_key.startswith("sk-..."):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        return OpenAIChatModel(chosen, provider=provider)

    ollama_url = os.getenv("OLLAMA_BASE_URL", "")
    if ollama_url:
        from pydantic_ai.models.ollama import OllamaModel
        from pydantic_ai.providers.ollama import OllamaProvider
        ollama_model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        provider = OllamaProvider(base_url=ollama_url)
        return OllamaModel(ollama_model, provider=provider)

    raise ValueError(
        "No LLM configured for PydanticAI.\n"
        "  Option 1 – OpenAI/Groq/OpenRouter: set OPENAI_API_KEY in your .env file\n"
        "  Option 2 – Ollama: run Ollama locally and set OLLAMA_BASE_URL=http://localhost:11434\n"
        "See .env.example for details."
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
