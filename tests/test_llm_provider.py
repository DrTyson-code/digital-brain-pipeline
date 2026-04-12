"""Tests for the LLM provider abstraction and mock provider.

Real API calls are never made here — tests use a MockProvider with canned
responses so they run offline, fast, and deterministically.
"""

from __future__ import annotations

import asyncio
from typing import Type, TypeVar

import pytest
from pydantic import BaseModel

from src.llm.provider import (
    LLMProvider,
    LLMResponse,
    ProviderConfig,
    create_provider,
)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Mock provider — returns canned responses without any network I/O
# ---------------------------------------------------------------------------


class MockProvider(LLMProvider):
    """Deterministic provider for testing.

    Pass ``canned_json`` to have ``extract_structured`` return a specific
    JSON string as if it came from the LLM.
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        canned_text: str = "mock response",
        canned_json: str = "{}",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        cfg = config or ProviderConfig(provider="mock")
        super().__init__(cfg)
        self._canned_text = canned_text
        self._canned_json = canned_json
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.complete_calls: list[dict] = []
        self.extract_calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def max_context_tokens(self) -> int:
        return 8_192

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.complete_calls.append(
            {"system": system_prompt, "user": user_prompt, "temperature": temperature}
        )
        return LLMResponse(
            content=self._canned_text,
            model=self.model_name,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=0.0,
            latency_ms=1.0,
        )

    async def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float | None = None,
    ) -> tuple[T, LLMResponse]:
        self.extract_calls.append(
            {"system": system_prompt, "user": user_prompt, "model": response_model.__name__}
        )
        parsed = response_model.model_validate_json(self._canned_json)
        response = LLMResponse(
            content=self._canned_json,
            model=self.model_name,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=0.0,
            latency_ms=1.0,
        )
        return parsed, response


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# ProviderConfig tests
# ---------------------------------------------------------------------------


def test_provider_config_defaults():
    cfg = ProviderConfig(provider="claude")
    assert cfg.temperature == 0.0
    assert cfg.max_retries == 2
    assert cfg.timeout_seconds == 30
    assert cfg.model is None


def test_provider_config_resolved_api_key_explicit():
    cfg = ProviderConfig(provider="claude", api_key="sk-test-123")
    assert cfg.resolved_api_key() == "sk-test-123"


def test_provider_config_resolved_api_key_env(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sk-from-env")
    cfg = ProviderConfig(provider="claude", api_key_env="TEST_LLM_KEY")
    assert cfg.resolved_api_key() == "sk-from-env"


def test_provider_config_resolved_api_key_missing_env(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = ProviderConfig(provider="claude", api_key_env="MISSING_KEY")
    assert cfg.resolved_api_key() is None


# ---------------------------------------------------------------------------
# create_provider factory tests
# ---------------------------------------------------------------------------


def test_create_provider_unknown_raises():
    cfg = ProviderConfig(provider="nonexistent")
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider(cfg)


# ---------------------------------------------------------------------------
# MockProvider / LLMProvider interface tests
# ---------------------------------------------------------------------------


def test_mock_provider_complete():
    provider = MockProvider(canned_text="hello world")
    resp = run(provider.complete("sys", "user"))
    assert resp.content == "hello world"
    assert resp.model == "mock-model"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50
    assert resp.cost_usd == 0.0
    assert len(provider.complete_calls) == 1


def test_mock_provider_complete_records_prompts():
    provider = MockProvider()
    run(provider.complete("system text", "user text", temperature=0.5))
    call = provider.complete_calls[0]
    assert call["system"] == "system text"
    assert call["user"] == "user text"
    assert call["temperature"] == 0.5


def test_mock_provider_extract_structured():
    class SimpleModel(BaseModel):
        value: int
        label: str

    provider = MockProvider(canned_json='{"value": 42, "label": "test"}')
    result, response = run(
        provider.extract_structured("sys", "user", SimpleModel)
    )
    assert isinstance(result, SimpleModel)
    assert result.value == 42
    assert result.label == "test"
    assert response.model == "mock-model"
    assert len(provider.extract_calls) == 1


def test_mock_provider_extract_records_model_name():
    class Dummy(BaseModel):
        x: int = 0

    provider = MockProvider(canned_json='{"x": 1}')
    run(provider.extract_structured("sys", "user", Dummy))
    assert provider.extract_calls[0]["model"] == "Dummy"


def test_mock_provider_estimate_cost_is_zero():
    provider = MockProvider()
    assert provider.estimate_cost(100_000, 50_000) == 0.0


def test_mock_provider_model_properties():
    provider = MockProvider()
    assert provider.model_name == "mock-model"
    assert provider.max_context_tokens == 8_192


def test_mock_provider_multiple_calls():
    provider = MockProvider()
    for _ in range(5):
        run(provider.complete("s", "u"))
    assert len(provider.complete_calls) == 5


# ---------------------------------------------------------------------------
# LLMResponse model tests
# ---------------------------------------------------------------------------


def test_llm_response_defaults():
    resp = LLMResponse(
        content="hi",
        model="test",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=100.0,
    )
    assert resp.cached is False


def test_llm_response_copy_with_cached():
    resp = LLMResponse(
        content="hi",
        model="test",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=100.0,
    )
    cached = resp.model_copy(update={"cached": True})
    assert cached.cached is True
    assert resp.cached is False  # original unchanged
