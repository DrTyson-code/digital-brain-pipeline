"""Anthropic Claude provider.

Uses the ``anthropic`` SDK with ``tool_use`` to achieve structured extraction —
Claude is asked to call a tool whose input schema exactly matches the target
Pydantic model, which guarantees valid JSON and clean deserialization.

Install dependency:  pip install anthropic>=0.40.0
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from src.llm.cost import compute_cost
from src.llm.provider import LLMProvider, LLMResponse, ProviderConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Default model — good price/quality balance
_DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Context windows per model family
_CONTEXT_TOKENS: dict[str, int] = {
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
}


def _context_for_model(model: str) -> int:
    for prefix, tokens in _CONTEXT_TOKENS.items():
        if model.startswith(prefix):
            return tokens
    return 200_000  # safe default for Claude models


class ClaudeProvider(LLMProvider):
    """LLM provider backed by the Anthropic API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeProvider. "
                "Install it with: pip install anthropic>=0.40.0"
            ) from exc

        api_key = config.resolved_api_key()
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._model = config.model or _DEFAULT_MODEL

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

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
        """Return a free-form text completion."""
        temp = temperature if temperature is not None else self.config.temperature
        t0 = time.monotonic()

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temp,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        latency_ms = (time.monotonic() - t0) * 1000
        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens
        content = response.content[0].text if response.content else ""

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
        """Extract a Pydantic model via Claude's tool_use feature.

        A synthetic tool named ``extract_data`` is defined whose input schema
        matches ``response_model``'s JSON schema.  Claude is forced to call it
        via ``tool_choice``, which guarantees structured output.
        """
        temp = temperature if temperature is not None else self.config.temperature
        t0 = time.monotonic()

        schema = response_model.model_json_schema()
        tool_def = {
            "name": "extract_data",
            "description": (
                f"Extract structured data matching the {response_model.__name__} schema"
            ),
            "input_schema": schema,
        }

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temp,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "extract_data"},
        )

        latency_ms = (time.monotonic() - t0) * 1000
        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens

        # Find the tool_use block
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )
        if tool_block is None:
            raise ValueError("Claude did not return a tool_use block in the response")

        raw_input = tool_block.input
        # tool_block.input is already a dict when using the SDK; serialize for Pydantic
        if isinstance(raw_input, dict):
            parsed = response_model.model_validate(raw_input)
        else:
            parsed = response_model.model_validate_json(str(raw_input))

        llm_response = LLMResponse(
            content=json.dumps(raw_input),
            model=self._model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=compute_cost(self._model, input_tok, output_tok),
            latency_ms=latency_ms,
        )
        return parsed, llm_response
