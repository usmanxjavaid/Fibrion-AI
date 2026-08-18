"""
backend/core/llm_client.py

Two providers: OpenRouter (primary, Claude models) and Groq (fallback,
open-weight models). Not chained via LangChain's .with_fallbacks() -
see module docstring history for why - fallback is handled explicitly
here so it correctly triggers on both a raised API error and a
structured-output parsing failure.
"""

import time
from typing import Literal, Optional, Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.config import settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("llm_client")

T = TypeVar("T", bound=BaseModel)
ModelTier = Literal["fast", "reasoning"]
Provider = Literal["openrouter", "gemini", "groq"]



_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": settings.openrouter_api_key,
        "models": {"fast": settings.default_model_fast, "reasoning": settings.default_model_reasoning},
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": settings.groq_api_key,
        "models": {"fast": "openai/gpt-oss-20b", "reasoning": "openai/gpt-oss-120b"},
    },
}



def get_llm(tier: ModelTier, provider: Provider = "openrouter",
            temperature: float = 0.0, max_tokens: int = 1024) -> ChatOpenAI:
    cfg = _PROVIDERS[provider]
    return ChatOpenAI(
        model=cfg["models"][tier], api_key=cfg["api_key"], base_url=cfg["base_url"],
        temperature=temperature, max_tokens=max_tokens,
    )

_NON_RETRYABLE_STATUS_CODES = {401, 402, 403}

def _try_provider(provider, tier, prompt, output_schema, max_retries, max_tokens):
    llm = get_llm(tier, provider=provider, max_tokens=max_tokens).with_structured_output(
        output_schema, include_raw=True
    )
    attempt_prompt = prompt
    last_error = None

    for attempt in range(max_retries + 1):
        start = time.monotonic()
        try:
            response = llm.invoke(attempt_prompt)
        except Exception as e:
            last_error = str(e)
            if getattr(e, "status_code", None) in _NON_RETRYABLE_STATUS_CODES:
                logger.warning(f"[{provider}] non-retryable ({e.status_code}), skipping remaining retries: {last_error}")
                break
            logger.warning(f"[{provider}] raised on attempt {attempt + 1}: {last_error}")
            attempt_prompt = f"{prompt}\n\nPrevious error: {last_error}\nRespond again, matching the required format."
            continue

        elapsed = time.monotonic() - start
        usage = response["raw"].response_metadata.get("token_usage", {})
        metadata = {
            "provider": provider, "model": _PROVIDERS[provider]["models"][tier],
            "attempt": attempt + 1, "elapsed_sec": round(elapsed, 2),
            "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
        }

        if response["parsed"] is not None:
            logger.info(f"LLM call succeeded: {metadata}")
            return response["parsed"], metadata

        last_error = response.get("parsing_error")
        logger.warning(f"[{provider}] bad parse on attempt {attempt + 1}: {last_error}")
        attempt_prompt = f"{prompt}\n\nPrevious response didn't match the schema. Error: {last_error}\nRespond again."

    return None, {"provider": provider, "model": _PROVIDERS[provider]["models"][tier],
                   "failed": True, "error": str(last_error)}


_FALLBACK_ORDER = ["gemini", "groq"]

def call_structured(
    tier: ModelTier, prompt: str, output_schema: Type[T],
    max_retries: int = 1, max_tokens: int = 1024, use_fallback: bool = True,
) -> tuple[Optional[T], dict]:
    parsed, metadata = _try_provider("openrouter", tier, prompt, output_schema, max_retries, max_tokens)
    if parsed is not None:
        return parsed, metadata

    if not use_fallback:
        return None, metadata

    errors = {"openrouter": metadata}
    for provider in _FALLBACK_ORDER:
        if not _PROVIDERS[provider]["api_key"]:
            continue
        logger.warning(f"Trying {provider} after prior failures: {list(errors.keys())}")
        parsed, fb_metadata = _try_provider(provider, tier, prompt, output_schema, max_retries, max_tokens)
        if parsed is not None:
            return parsed, fb_metadata
        errors[provider] = fb_metadata

    logger.error(f"All providers failed: {errors}")
    return None, {"errors": errors}