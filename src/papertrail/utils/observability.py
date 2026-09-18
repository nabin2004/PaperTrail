"""
PaperTrail Logfire Observability & Tracing Infrastructure.

Configures Pydantic Logfire, OpenTelemetry instrumentation for PydanticAI,
OpenAI, LangChain, HTTPX, Requests, and Pydantic models with explicit model tracking.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional
import logfire

_IS_CONFIGURED: bool = False

# Custom Metrics
cli_command_counter = logfire.metric_counter(
    "papertrail_cli_commands_total",
    unit="1",
    description="Total count of PaperTrail CLI command invocations",
)

agent_run_counter = logfire.metric_counter(
    "papertrail_agent_runs_total",
    unit="1",
    description="Total count of PaperTrail agent executions",
)

papers_indexed_counter = logfire.metric_counter(
    "papertrail_papers_indexed_total",
    unit="1",
    description="Total count of research papers indexed into vector store",
)

retrieval_histogram = logfire.metric_histogram(
    "papertrail_retrieval_duration_seconds",
    unit="s",
    description="Duration of paper search and retrieval operations in seconds",
)


def setup_observability(
    service_name: str = "papertrail",
    service_version: str = "0.1.0",
    environment: Optional[str] = None,
    send_to_logfire: Optional[str | bool] = None,
    console: Optional[bool] = None,
) -> bool:
    """
    Initialize and configure Logfire for PaperTrail.

    - Idempotent: safe to invoke multiple times.
    - Enables LangChain OpenTelemetry integration.
    - Instruments PydanticAI, OpenAI, HTTPX, Requests, and Pydantic.
    - Bridges standard library logging.
    - Gracefully falls back if no LOGFIRE_TOKEN is present without crashing.
    """
    global _IS_CONFIGURED
    if _IS_CONFIGURED:
        return True

    env = environment or os.getenv("ENVIRONMENT") or os.getenv("DEPLOYMENT_ENVIRONMENT") or "development"
    
    # Configure console output (default off to keep Rich CLI output clean unless requested)
    if console is None:
        console_env = os.getenv("LOGFIRE_CONSOLE", "").strip().lower()
        console = console_env in ("1", "true", "yes", "on")

    # Native LangChain/LangSmith OpenTelemetry settings (langsmith >= 0.4.25)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
    os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")

    # If send_to_logfire not explicitly provided, default to 'if-token-present'
    if send_to_logfire is None:
        send_to_logfire = "if-token-present"

    try:
        logfire.configure(
            service_name=service_name,
            service_version=service_version,
            environment=env,
            send_to_logfire=send_to_logfire,
            console=console,
        )
    except Exception as exc:
        # Fallback in case of configuration errors in constrained environments
        logfire.configure(send_to_logfire=False, console=False)
        logfire.warn("Logfire fallback configuration activated due to: {error}", error=str(exc))

    # Instrument libraries
    _instrument_libraries()

    # Bridge Python standard library logging
    try:
        root_logger = logging.getLogger()
        handler = logfire.LogfireLoggingHandler()
        # Avoid duplicate handlers if called multiple times
        if not any(isinstance(h, logfire.LogfireLoggingHandler) for h in root_logger.handlers):
            root_logger.addHandler(handler)
    except Exception:
        pass

    # Suppress verbose third-party loggers
    for noisy in ("httpcore", "httpx", "urllib3", "filelock", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _IS_CONFIGURED = True
    logfire.info(
        "PaperTrail observability initialized for {service} v{version} [{env}]",
        service=service_name,
        version=service_version,
        env=env,
    )
    return True


def _instrument_libraries() -> None:
    """Instrument installed AI, HTTP, and data libraries."""
    # 1. PydanticAI
    try:
        logfire.instrument_pydantic_ai()
    except Exception as exc:
        logfire.debug("PydanticAI instrumentation skipped: {error}", error=str(exc))

    # 2. OpenAI
    try:
        logfire.instrument_openai()
    except Exception as exc:
        logfire.debug("OpenAI instrumentation skipped: {error}", error=str(exc))

    # 3. HTTPX
    try:
        logfire.instrument_httpx()
    except Exception as exc:
        logfire.debug("HTTPX instrumentation skipped: {error}", error=str(exc))

    # 4. Requests (used by arXiv client & PDF download)
    try:
        logfire.instrument_requests()
    except Exception as exc:
        logfire.debug("Requests instrumentation skipped: {error}", error=str(exc))

    # 5. Pydantic data validation
    try:
        logfire.instrument_pydantic()
    except Exception as exc:
        logfire.debug("Pydantic instrumentation skipped: {error}", error=str(exc))


def extract_model_name(model_obj_or_name: Any = None) -> str:
    """
    Extract a normalized string model name from a string, PydanticAI Model,
    LangChain BaseChatModel, or system configuration.
    """
    if model_obj_or_name is not None:
        if isinstance(model_obj_or_name, str):
            return model_obj_or_name
        if hasattr(model_obj_or_name, "model_name") and getattr(model_obj_or_name, "model_name"):
            return str(getattr(model_obj_or_name, "model_name"))
        if hasattr(model_obj_or_name, "model") and getattr(model_obj_or_name, "model"):
            return str(getattr(model_obj_or_name, "model"))
        if hasattr(model_obj_or_name, "name") and getattr(model_obj_or_name, "name"):
            return str(getattr(model_obj_or_name, "name"))
        return str(model_obj_or_name)

    # Fallback to environment variables
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and not openai_key.startswith("sk-..."):
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if os.getenv("OLLAMA_BASE_URL", ""):
        return os.getenv("OLLAMA_MODEL", "llama3.2")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
