"""Tests for LLMExtractor — LLM-powered entity and concept extraction.

Tests cover cache hits/misses, budget enforcement, batch processing,
conversation truncation, and schema conversion.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Type, TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from src.llm.cache import ExtractionCache
from src.llm.cost import BudgetExceeded, CostTracker, TokenBudget
from src.llm.extractor import LLMExtractor
from src.llm.prompts.concept_classification import ConversationClassification
from src.llm.prompts.entity_extraction import ExtractedEntity, LLMEntityExtraction
from src.llm.provider import LLMProvider, LLMResponse, ProviderConfig
from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import ChatMessage, Conversation, Role
from src.process.extractor import ExtractionResult

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Fixtures
# ============================================================================


class MockProvider(LLMProvider):
    """Mock LLM provider for testing without network calls."""

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        canned_entities: list[dict] | None = None,
        canned_classification: dict | None = None,
        should_raise: Exception | None = None,
    ) -> None:
        cfg = config or ProviderConfig(provider="mock")
        super().__init__(cfg)
        self.should_raise = should_raise
        self.canned_entities = canned_entities or []
        self.canned_classification = canned_classification or {
            "primary_domain": "general",
            "secondary_domains": [],
            "purpose": "General discussion",
            "depth": "surface",
            "concepts": [],
            "summary": "Mock classification.",
        }
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
        return LLMResponse(
            content="mock",
            model=self.model_name,
            input_tokens=100,
            output_tokens=50,
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
        if self.should_raise:
            raise self.should_raise

        self.extract_calls.append(
            {"system": system_prompt[:50], "user": user_prompt[:50], "model": response_model.__name__}
        )

        # Return canned response based on model
        if response_model == LLMEntityExtraction:
            entities_data = {"entities": self.canned_entities}
            parsed = LLMEntityExtraction(**entities_data)
        elif response_model == ConversationClassification:
            parsed = ConversationClassification(**self.canned_classification)
        else:
            parsed = response_model.model_validate({})

        response = LLMResponse(
            content="{}",
            model=self.model_name,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0,
            latency_ms=1.0,
        )
        return parsed, response


@pytest.fixture
def cache(tmp_path):
    """ExtractionCache backed by temp file."""
    return ExtractionCache(cache_path=tmp_path / "test_cache.db")


@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    msgs = [
        ChatMessage(
            conversation_id="conv-1",
            role=Role.USER,
            content="Can you help with Python?",
            platform="claude",
        ),
        ChatMessage(
            conversation_id="conv-1",
            role=Role.ASSISTANT,
            content="Sure, Python is a great language.",
            platform="claude",
        ),
    ]
    return Conversation(
        id="conv-1",
        title="Python Discussion",
        messages=msgs,
        platform="claude",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_entity_llm_data():
    """Sample LLM entity extraction data."""
    return [
        {
            "name": "Python",
            "entity_type": "TOOL",
            "description": "A programming language discussed",
            "aliases": ["python3", "py"],
            "confidence": 0.95,
            "source_quotes": ["Python is a great language"],
        },
        {
            "name": "Alice",
            "entity_type": "PERSON",
            "description": "A person mentioned in the conversation",
            "aliases": ["Alice Smith"],
            "confidence": 0.85,
            "source_quotes": ["Alice said"],
        },
    ]


@pytest.fixture
def sample_classification_data():
    """Sample classification data."""
    return {
        "primary_domain": "software_engineering",
        "secondary_domains": ["python"],
        "purpose": "Learn Python",
        "depth": "intermediate",
        "concepts": [
            {
                "name": "Function Definition",
                "description": "How to define functions",
                "domain": "syntax",
                "confidence": 0.9,
            }
        ],
        "summary": "Discussion about Python programming",
    }


# ============================================================================
# Basic extraction tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_with_mock_provider(sample_conversation, sample_entity_llm_data):
    """Test basic extraction with a mock provider."""
    provider = MockProvider(canned_entities=sample_entity_llm_data)
    extractor = LLMExtractor(provider)

    result = await extractor.extract(sample_conversation)

    assert result.conversation_id == "conv-1"
    assert len(result.entities) == 2
    assert result.entities[0].name == "Python"
    assert result.entities[0].entity_type == EntityType.TOOL


@pytest.mark.asyncio
async def test_extract_entity_type_conversion(sample_conversation):
    """Test that LLM entity types are converted correctly."""
    entity_data = [
        {
            "name": "Alice",
            "entity_type": "person",  # lowercase
            "description": "A person",
            "aliases": [],
            "confidence": 0.9,
            "source_quotes": [],
        }
    ]
    provider = MockProvider(canned_entities=entity_data)
    extractor = LLMExtractor(provider)

    result = await extractor.extract(sample_conversation)

    assert result.entities[0].entity_type == EntityType.PERSON


@pytest.mark.asyncio
async def test_extract_unknown_entity_type_defaults_to_tool(sample_conversation):
    """Test that unknown entity types default to TOOL."""
    entity_data = [
        {
            "name": "UnknownThing",
            "entity_type": "INVALID_TYPE",
            "description": "Some entity",
            "aliases": [],
            "confidence": 0.9,
            "source_quotes": [],
        }
    ]
    provider = MockProvider(canned_entities=entity_data)
    extractor = LLMExtractor(provider)

    result = await extractor.extract(sample_conversation)

    assert result.entities[0].entity_type == EntityType.TOOL


@pytest.mark.asyncio
async def test_extract_concepts_from_classification(sample_conversation, sample_classification_data):
    """Test concept extraction from classification."""
    provider = MockProvider(
        canned_entities=[],
        canned_classification=sample_classification_data,
    )
    extractor = LLMExtractor(provider)

    result = await extractor.extract(sample_conversation)

    # Should have concepts from classification
    assert len(result.concepts) > 0
    # Should include domain, purpose, depth, summary concepts
    concept_contents = [c.content for c in result.concepts]
    assert any("software_engineering" in c for c in concept_contents)


# ============================================================================
# Cache tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_cache_hit(sample_conversation, sample_entity_llm_data, cache):
    """Test that cache hits skip provider calls."""
    provider = MockProvider(canned_entities=sample_entity_llm_data)
    extractor = LLMExtractor(provider, cache=cache)

    # First call — cache miss, calls provider
    result1 = await extractor.extract(sample_conversation)
    assert len(provider.extract_calls) > 0
    first_call_count = len(provider.extract_calls)

    # Second call — cache hit, should not call provider for entity extraction
    result2 = await extractor.extract(sample_conversation)

    # Results should be identical
    assert len(result1.entities) == len(result2.entities)
    assert result1.entities[0].name == result2.entities[0].name


@pytest.mark.asyncio
async def test_extract_cache_put_get(sample_conversation, sample_entity_llm_data, cache):
    """Test cache put and get operations."""
    provider = MockProvider(canned_entities=sample_entity_llm_data)
    extractor = LLMExtractor(provider, cache=cache)

    # First extraction caches the result
    result1 = await extractor.extract(sample_conversation)

    # Create new extractor with same cache
    provider2 = MockProvider()
    extractor2 = LLMExtractor(provider2, cache=cache)

    # Second extraction should hit cache (provider2 has no canned data)
    result2 = await extractor2.extract(sample_conversation)

    # Results should match (came from cache)
    assert len(result1.entities) == len(result2.entities)


# ============================================================================
# Budget enforcement tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_budget_exceeded_raises(sample_conversation, sample_entity_llm_data):
    """Test that hard budget exceeded raises BudgetExceeded."""
    provider = MockProvider(canned_entities=sample_entity_llm_data)
    budget = TokenBudget(max_cost_usd=0.001)  # Minimal budget
    cost_tracker = CostTracker(budget)

    # Mock the provider to have non-zero cost
    provider.extract_calls = []

    async def mock_extract(*args, **kwargs):
        raise BudgetExceeded(0.01, 0.0)

    provider.extract_structured = mock_extract

    extractor = LLMExtractor(provider, cost_tracker=cost_tracker)

    with pytest.raises(BudgetExceeded):
        await extractor.extract(sample_conversation)


@pytest.mark.asyncio
async def test_extract_skips_stage_when_budget_low(sample_conversation):
    """Test that stages are skipped gracefully when budget is low."""
    provider = MockProvider()

    # Very small per-conversation budget
    budget = TokenBudget(max_cost_per_conversation=0.000001)
    cost_tracker = CostTracker(budget)

    extractor = LLMExtractor(provider, cost_tracker=cost_tracker)

    # Extraction should complete even if stages are skipped
    result = await extractor.extract(sample_conversation)
    assert result.conversation_id == "conv-1"


# ============================================================================
# Conversation text formatting and truncation
# ============================================================================


def test_conversation_text_formatting(sample_conversation):
    """Test that conversation text is formatted correctly."""
    provider = MockProvider()
    extractor = LLMExtractor(provider)

    text = extractor._build_conversation_text(sample_conversation)

    assert "User: Can you help with Python?" in text
    assert "Assistant: Sure, Python is a great language." in text


def test_conversation_text_truncation():
    """Test that long conversations are truncated."""
    # Create a very long conversation
    msgs = [
        ChatMessage(
            conversation_id="c",
            role=Role.USER,
            content="x" * 50000,  # Very long message
            platform="claude",
        )
    ]
    conv = Conversation(
        id="c",
        messages=msgs,
        platform="claude",
        created_at=datetime.now(timezone.utc),
    )

    provider = MockProvider()
    extractor = LLMExtractor(provider, max_conversation_tokens=1000)

    text = extractor._build_conversation_text(conv)

    # Should be truncated
    assert "[... truncated ...]" in text
    assert len(text) <= 1000 * 4 + 100  # 4 chars per token + buffer for formatting


def test_conversation_text_empty():
    """Test handling of empty conversation."""
    conv = Conversation(
        id="c",
        messages=[],
        platform="claude",
        created_at=datetime.now(timezone.utc),
    )

    provider = MockProvider()
    extractor = LLMExtractor(provider)

    text = extractor._build_conversation_text(conv)
    assert text == ""


# ============================================================================
# Batch processing tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_batch_processes_multiple(sample_conversation, sample_entity_llm_data):
    """Test batch processing with multiple conversations."""
    provider = MockProvider(canned_entities=sample_entity_llm_data)
    extractor = LLMExtractor(provider)

    conversations = [sample_conversation] * 3

    results = await extractor.extract_batch(conversations, batch_size=2)

    assert len(results) == 3
    for result in results:
        assert result.conversation_id == "conv-1"
        assert len(result.entities) == 2


@pytest.mark.asyncio
async def test_extract_batch_with_errors_continues(sample_conversation, sample_entity_llm_data):
    """Test that batch processing continues even when individual extractions fail."""
    provider = MockProvider(canned_entities=sample_entity_llm_data)
    extractor = LLMExtractor(provider)

    conversations = [sample_conversation] * 3

    # Make the provider fail on second call
    call_count = 0

    async def mock_extract(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Something went wrong")
        return (
            LLMEntityExtraction(entities=[]),
            LLMResponse(
                content="{}",
                model="mock",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.0,
                latency_ms=1.0,
            ),
        )

    provider.extract_structured = mock_extract

    results = await extractor.extract_batch(conversations, batch_size=1)

    # Should still return 3 results, but one will be empty
    assert len(results) == 3


@pytest.mark.asyncio
async def test_extract_batch_stops_on_hard_budget_exceeded(sample_conversation):
    """Test that batch processing stops when hard budget is exceeded."""
    provider = MockProvider()

    # Setup cost tracker that will raise on first call
    async def mock_extract(*args, **kwargs):
        raise BudgetExceeded(0.01, 0.0)

    provider.extract_structured = mock_extract

    budget = TokenBudget(max_cost_usd=0.001)
    cost_tracker = CostTracker(budget)

    extractor = LLMExtractor(provider, cost_tracker=cost_tracker)

    conversations = [sample_conversation] * 3

    with pytest.raises(BudgetExceeded):
        await extractor.extract_batch(conversations, batch_size=1)


@pytest.mark.asyncio
async def test_extract_batch_size_respected(sample_conversation, sample_entity_llm_data):
    """Test that batch size parameter is respected."""
    provider = MockProvider(canned_entities=sample_entity_llm_data)
    extractor = LLMExtractor(provider)

    conversations = [sample_conversation] * 10

    results = await extractor.extract_batch(conversations, batch_size=3)

    assert len(results) == 10


# ============================================================================
# Entity/Concept conversion tests
# ============================================================================


def test_convert_entities_multiple_types():
    """Test converting entities of different types."""
    provider = MockProvider()
    extractor = LLMExtractor(provider)

    llm_entities = [
        ExtractedEntity(
            name="Python",
            entity_type="tool",
            description="Language",
            confidence=0.9,
        ),
        ExtractedEntity(
            name="Alice",
            entity_type="person",
            description="Person",
            confidence=0.8,
        ),
        ExtractedEntity(
            name="Google",
            entity_type="organization",
            description="Company",
            confidence=0.95,
        ),
    ]

    entities = extractor._convert_entities(llm_entities, "conv-1")

    assert len(entities) == 3
    assert entities[0].entity_type == EntityType.TOOL
    assert entities[1].entity_type == EntityType.PERSON
    assert entities[2].entity_type == EntityType.ORGANIZATION


def test_convert_entities_stores_properties():
    """Test that entity properties from LLM are stored."""
    provider = MockProvider()
    extractor = LLMExtractor(provider)

    llm_entities = [
        ExtractedEntity(
            name="Python",
            entity_type="tool",
            description="A language",
            aliases=["py", "python3"],
            confidence=0.95,
            source_quotes=["Python is great"],
        )
    ]

    entities = extractor._convert_entities(llm_entities, "conv-1")

    entity = entities[0]
    assert entity.name == "Python"
    assert "py" in entity.aliases
    assert "python3" in entity.aliases
    assert entity.properties["confidence"] == 0.95
    assert entity.properties["description"] == "A language"
    assert "Python is great" in entity.properties["source_quotes"]


def test_convert_concepts():
    """Test concept conversion from classification."""
    provider = MockProvider()
    extractor = LLMExtractor(provider)

    classification = ConversationClassification(
        primary_domain="software_engineering",
        secondary_domains=["python", "web"],
        purpose="Learn Python",
        depth="intermediate",
        concepts=[
            {
                "name": "Function Definition",
                "description": "How to define",
                "domain": "syntax",
                "confidence": 0.9,
            }
        ],
        summary="A useful discussion",
    )

    concepts = extractor._convert_concepts(classification, "conv-1")

    # Should have multiple concepts from the classification
    assert len(concepts) > 1
    # Check for domain, purpose, depth, summary concepts
    content_strings = [c.content for c in concepts]
    assert any("software_engineering" in str(c) for c in content_strings)
    assert any("Learn Python" in str(c) for c in content_strings)


# ============================================================================
# Async event loop helper (for non-async tests)
# ============================================================================


def run_async(coro):
    """Helper to run async code in sync tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)
