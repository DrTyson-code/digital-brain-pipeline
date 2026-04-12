"""Abstract LLM provider interface and factory.

Defines the contract all provider implementations must fulfill, plus the
ProviderConfig Pydantic model and create_provider factory function.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    """Wrapper for any LLM API response, with usage metadata."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool = False


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider instance."""

    provider: str  # "claude" | "openai" | "ollama"
    model: Optional[str] = None  # None = use provider default
    api_key: Optional[str] = None  # None = read from env
    api_key_env: Optional[str] = None  # env var name, e.g. "ANTHROPIC_API_KEY"
    base_url: Optional[str] = None  # Override for ollama or proxies
    temperature: float = 0.0
    max_retries: int = 2
    timeout_seconds: int = 30

    def resolved_api_key(self) -> Optional[str]:
        """Return api_key, falling back to the env var named in api_key_env."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


class LLMProvider(ABC, Generic[T]):
    """Abstract base for all LLM providers.

    Every implementation must:
    - Support basic text completion via ``complete()``
    - Support structured extraction (Pydantic model) via ``extract_structured()``
    - Report cost estimates via ``estimate_cost()``
    - Expose ``model_name`` and ``max_context_tokens`` properties
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Return a free-form text completion."""
        ...

    @abstractmethod
    async def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: Optional[float] = None,
    ) -> Tuple[T, LLMResponse]:
        """Return a validated Pydantic model plus raw response metadata.

        Implementations should use tool_use / function_calling for Claude/OpenAI
        and JSON mode + manual parse for Ollama.
        """
        ...

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost before making a call, without hitting the API."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the resolved model name (including provider default)."""
        ...

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Return the context window size for the current model."""
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(config: ProviderConfig) -> LLMProvider:
    """Instantiate the appropriate LLMProvider from a ProviderConfig.

    Imports are deferred so that projects using only one provider don't need
    all optional dependencies installed.
    """
    if config.provider == "claude":
        from src.llm.providers.claude import ClaudeProvider  # noqa: PLC0415
        return ClaudeProvider(config)
    elif config.provider == "openai":
        from src.llm.providers.openai import OpenAIProvider  # noqa: PLC0415
        return OpenAIProvider(config)
    elif config.provider == "ollama":
        from src.llm.providers.ollama import OllamaProvider  # noqa: PLC0415
        return OllamaProvider(config)
    else:
        raise ValueError(
            f"Unknown provider: {config.provider!r}. "
            "Expected 'claude', 'openai', or 'ollama'."
        )
