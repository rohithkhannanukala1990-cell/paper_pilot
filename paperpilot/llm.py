"""Anthropic Messages API helpers (only module that imports the anthropic SDK)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from paperpilot.config import settings

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


class LLMError(Exception):
    """Readable failure from the language-model provider."""


def _get_client() -> anthropic.Anthropic:
    """Lazy-load ``anthropic.Anthropic()`` (uses ``ANTHROPIC_API_KEY`` from the env)."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _join_text(content: list[Any]) -> str:
    """Concatenate text blocks from a Messages API response."""
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _friendly_llm_message(exc: BaseException) -> str:
    """Map provider exceptions to a short user-facing message (no stack details)."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "Invalid or missing ANTHROPIC_API_KEY. Check your .env and try again."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "The language model denied this request. Check API key permissions."
    if isinstance(exc, anthropic.RateLimitError):
        return "The language model rate limit was exceeded after several retries. Try again later."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the language model after several retries. Check your network."
    if isinstance(exc, anthropic.InternalServerError):
        return "The language model service had an internal error after several retries. Try again later."
    if isinstance(exc, anthropic.BadRequestError):
        return "The language model rejected the request. Try a shorter question or fewer papers."
    if isinstance(exc, anthropic.APIStatusError):
        return "The language model returned an unexpected error. Please try again."
    if isinstance(exc, anthropic.APIError):
        return "Language model request failed. Please try again."
    return "Language model request failed. Please try again."


def _reraise_as_llm_error(exc: BaseException) -> None:
    """Raise ``LLMError`` with a readable message, chaining the original exception."""
    raise LLMError(_friendly_llm_message(exc)) from exc


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(
        (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        )
    ),
)
def _complete_once(system: str, user: str, max_tokens: int) -> str:
    """Single ``messages.create`` call (retried on transient errors)."""
    client = _get_client()
    t0 = time.perf_counter()
    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency = time.perf_counter() - t0
    logger.info(
        "generate model=%s latency=%.2fs input_tokens=%s output_tokens=%s",
        settings.llm_model,
        latency,
        message.usage.input_tokens,
        message.usage.output_tokens,
    )
    return _join_text(message.content)


def complete(system: str, user: str, max_tokens: int = 1200) -> str:
    """Call ``messages.create`` and return the joined text blocks."""
    try:
        return _complete_once(system, user, max_tokens)
    except Exception as exc:
        logger.warning("LLM complete failed: %s", type(exc).__name__)
        _reraise_as_llm_error(exc)
        raise  # pragma: no cover — _reraise_as_llm_error always raises


def stream(system: str, user: str, max_tokens: int = 1200) -> Iterator[str]:
    """Yield text deltas from ``messages.stream``."""
    try:
        client = _get_client()
        t0 = time.perf_counter()
        with client.messages.stream(
            model=settings.llm_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as event_stream:
            for text in event_stream.text_stream:
                yield text
            final = event_stream.get_final_message()
        latency = time.perf_counter() - t0
        logger.info(
            "generate(stream) model=%s latency=%.2fs input_tokens=%s output_tokens=%s",
            settings.llm_model,
            latency,
            final.usage.input_tokens,
            final.usage.output_tokens,
        )
    except Exception as exc:
        logger.warning("LLM stream failed: %s", type(exc).__name__)
        _reraise_as_llm_error(exc)


# To swap in OpenAI or Ollama, rewrite only `complete` and `stream` above so they keep
# the same signatures and return/yield plain text. Point settings.llm_model (and any
# extra env vars) at the new provider; leave prompts.py and service.py unchanged.
