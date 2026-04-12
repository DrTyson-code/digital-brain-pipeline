"""Ollama provider for local model inference.

Makes HTTP calls to a running Ollama server (default: http://localhost:11434).
Uses ``format: "json"`` in the chat API to enable JSON mode.  Response is then
manually validated against the target Pydantic model.

All models are free — no API key required, no cost tracking needed.

Install dependency:  pip install httpx>=0.27.0
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.llm.cost import compute_cost
from src.llm.provider import LLMProvider, LLMResponse, ProviderConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = "llama3.1:8b"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'httpx' package is required for OllamaProvider. "
                "Install it with: pip install httpx>=0.27.0"
            ) from exc

        base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._base_url = base_url
        self._chat_url = f"{base_url}/api/chat"
        self._model = config.model or _DEFAULT_MODEL
        self._httpx = httpx  # keep reference so subclasses can override

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        # Ollama models vary; 8192 is a conservative safe default
        return 8_192

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # Local models are always free

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    async def _post(self, payload: dict) -> dict:
        """POST to the Ollama chat endpoint and return the parsed JSON body."""
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(self._chat_url, json=payload)
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self.config.temperature
        t0 = time.monotonic()

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temp, "num_predict": max_tokens},
        }

        data = await self._post(payload)
        latency_ms = (time.monotonic() - t0) * 1000
        content = data.get("message", {}).get("content", "")

        # Ollama reports eval_count (output) and prompt_eval_count (input)
        input_tok = data.get("prompt_eval_count", 0)
        output_tok = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=0.0,
            latency_ms=latency_ms,
        )

    async def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: Optional[float] = None,
    ) -> Tuple[T, LLMResponse]:
        """Extract a Pydantic model using Ollama's JSON mode.

        Ollama doesn't enforce a specific schema, so we add the schema to the
        system prompt and validate manually.  On parse failure we retry once
        with the validation error fed back to the model.
        """
        temp = temperature if temperature is not None else self.config.temperature

        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\n"
            f"You MUST return valid JSON matching this exact schema:\n{schema_str}"
        )

        total_input = total_output = 0
        t0 = time.monotonic()

        for attempt in range(self.config.max_retries):
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": temp},
            }

            data = await self._post(payload)
            raw_content = data.get("message", {}).get("content", "{}")
            total_input += data.get("prompt_eval_count", 0)
            total_output += data.get("eval_count", 0)

            try:
                parsed = response_model.model_validate_json(raw_content)
                latency_ms = (time.monotonic() - t0) * 1000
                llm_response = LLMResponse(
                    content=raw_content,
                    model=self._model,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    cost_usd=0.0,
                    latency_ms=latency_ms,
                )
                return parsed, llm_response

            except (ValidationError, json.JSONDecodeError) as exc:
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        "Ollama JSON parse failed (attempt %d/%d): %s — retrying",
                        attempt + 1,
                        self.config.max_retries,
                        exc,
                    )
                    # Feed the error back via the user prompt for the retry
                    user_prompt = (
                        f"{user_prompt}\n\n"
                        f"Your previous response failed validation:\n{exc}\n"
                        f"Previous response was:\n{raw_content}\n"
                        f"Please fix the JSON and try again."
                    )
                else:
                    latency_ms = (time.monotonic() - t0) * 1000
                    raise ValueError(
                        f"Ollama failed to return valid {response_model.__name__} "
                        f"after {self.config.max_retries} attempts"
                    ) from exc

        # Unreachable, but satisfies type checker
        raise RuntimeError("extract_structured: exhausted retries without returning")
