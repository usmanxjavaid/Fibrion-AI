"""
backend/core/llm_client.py

Single wrapper around every LLM call in the pipeline, routed through
OpenRouter via langchain_openai.ChatOpenAI - not the raw openai SDK -
so structured output and (once enabled) LangSmith tracing work with
no extra code in any agent.
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

_TIER_TO_MODEL = {
    "fast": settings.default_model_fast,
    "reasoning": settings.default_model_reasoning,
}


def get_llm(tier: ModelTier, temperature: float = 0.0, max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        model=_TIER_TO_MODEL[tier],
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def call_structured(
    tier: ModelTier,
    prompt: str,
    output_schema: Type[T],
    max_retries: int = 1,
    max_tokens: int = 1024,
) -> tuple[Optional[T], dict]:
    llm = get_llm(tier, max_tokens=max_tokens).with_structured_output(output_schema, include_raw=True)

    attempt_prompt = prompt
    last_error = None

    for attempt in range(max_retries + 1):
        start = time.monotonic()
        try:
            response = llm.invoke(attempt_prompt)
        except Exception as e:
            last_error = str(e)
            logger.warning(f"LLM call raised directly on attempt {attempt + 1}: {last_error}")
            attempt_prompt = (
                f"{prompt}\n\nYour previous response caused an error: {last_error}\n"
                f"Please respond again, correctly matching the required format."
            )
            continue

        elapsed = time.monotonic() - start
        usage = response["raw"].response_metadata.get("token_usage", {})
        metadata = {
            "model": _TIER_TO_MODEL[tier],
            "attempt": attempt + 1,
            "elapsed_sec": round(elapsed, 2),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        }
        logger.info(f"LLM call: {metadata}")

        if response["parsed"] is not None:
            return response["parsed"], metadata

        last_error = response.get("parsing_error")
        logger.warning(f"Structured output failed validation (attempt {attempt + 1}): {last_error}")
        attempt_prompt = (
            f"{prompt}\n\nYour previous response didn't match the required "
            f"format. Error: {last_error}\nPlease respond again, correctly "
            f"matching the schema."
        )

    logger.error(f"Structured output failed after {max_retries + 1} attempts: {last_error}")
    return None, {"model": _TIER_TO_MODEL[tier], "failed": True, "error": str(last_error)}