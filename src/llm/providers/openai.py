"""OpenAI provider.

Uses the ``openai`` SDK with JSON schema structured outputs
(``response_format={"type": "json_schema", ...}``) to guarantee a valid,
typed response without extra parsing logic.

Install dependency:  pip install openai>=1.50.0
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from src.llm.cost import compute_cost
from src.llm.provider import LLMProvider, LLMResponse, ProviderConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = "gpt-4o-mini"

_CONTEXT_TOKENS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
}


def _context_for_model(model: str) -> int:
    for prefix, tokens in _CONTEXT_TOKENS.items():
        if model.startswith(prefix):
            return tokens
    return 128_000


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: pip install openai>=1.50.0"
            ) from exc

        api_key = config.resolved_api_key()
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._model = config.model or _DEFAULT_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        return _context_for_model(self._model)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return compute_cost(self._model, input_tokens, output_tokens)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self.config.temperature
        t0 = time.monotonic()

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temp,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        latency_ms = (time.monotonic() - t0) * 1000
        usage = response.usage
        input_tok = usage.prompt_tokens if usage else 0
        output_tok = usage.completion_tokens if usage else 0
        content = response.choices[0].message.content or ""

        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=compute_cost(self._model, input_tok, output_tok),
            latency_ms=latency_ms,
        )

    async def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: Optional[float] = None,
    ) -> Tuple[T, LLMResponse]:
        """Extract a Pydantic model using OpenAI structured outputs.

        The Pydantic model's JSON schema is passed as ``json_schema`` inside
        ``response_format``.  The API guarantees the response matches the schema.
        """
        temp = temperature if temperature is not None else self.config.temperature
        t0 = time.monotonic()

        schema = response_model.model_json_schema()
        # Strip unsupported keywords for OpenAI structured outputs
        schema.pop("title", None)

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=4096,
            temperature=temp,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        latency_ms = (time.monotonic() - t0) * 1000
        usage = response.usage
        input_tok = usage.prompt_tokens if usage else 0
        output_tok = usage.completion_tokens if usage else 0
        raw_content = response.choices[0].message.content or "{}"

        parsed = response_model.model_validate_json(raw_content)

        llm_response = LLMResponse(
            content=raw_content,
            model=self._model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=compute_cost(self._model, input_tok, output_tok),
            latency_ms=latency_ms,
        )
        return parsed, llm_response
