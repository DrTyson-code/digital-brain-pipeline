"""Tests for the per-conversation concept cap.

Codex multi-commit cold review (2026-05-02) flagged that long Cowork-style
sessions can produce 1000+ regex matches per conversation, dominating Stage 3c
merger runtime (O(n*m) over rule and LLM concept lists).

The cap is applied in two places:
1. Primary: EntityConceptExtractor.extract() truncates per-conversation concepts
2. Defensive: ExtractionMerger.merge() also truncates inputs

These tests pin both behaviors plus the case where the cap is disabled.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from src.models.concept import Concept, ConceptType
from src.models.message import ChatMessage, Conversation
from src.process.extractor import (
    DEFAULT_MAX_CONCEPTS_PER_CONVERSATION,
    EntityConceptExtractor,
    ExtractionResult,
)


def _conv_with_pattern_hits(num_hits: int, conv_id: str = "longconv") -> Conversation:
    """Build a Conversation that triggers num_hits decision-pattern matches.

    The DECISION_PATTERNS regex looks for "decided to X." — we synthesize a
    text body of N such sentences, each unique, so the per-conversation
    dedup doesn't collapse them.
    """
    body_parts = []
    for i in range(num_hits):
        # Each unique enough to survive the seen-set dedup, and >10 chars
        body_parts.append(
            f"We decided to ship feature variant number {i:04d} after review."
        )
    body = " ".join(body_parts)
    msg = ChatMessage(
        conversation_id=conv_id,
        role="user",
        content=body,
        timestamp=datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
        platform="claude",
    )
    return Conversation(
        id=conv_id,
        title=f"long {conv_id}",
        platform="claude",
        created_at=datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
        messages=[msg],
        topics=["programming"],
    )


def test_extractor_caps_at_default(caplog: pytest.LogCaptureFixture) -> None:
    """Extractor truncates concepts to the default cap and logs a warning."""
    caplog.set_level(logging.WARNING, logger="src.process.extractor")
    conv = _conv_with_pattern_hits(num_hits=300)
    extractor = EntityConceptExtractor()  # default cap = 200
    result = extractor.extract(conv)
    assert len(result.concepts) <= DEFAULT_MAX_CONCEPTS_PER_CONVERSATION
    assert len(result.concepts) == DEFAULT_MAX_CONCEPTS_PER_CONVERSATION
    text = caplog.text
    assert "Concept cap applied" in text
    assert conv.id in text
    assert "cap=200" in text


def test_extractor_respects_explicit_cap(caplog: pytest.LogCaptureFixture) -> None:
    """A custom cap value is respected and logged."""
    caplog.set_level(logging.WARNING, logger="src.process.extractor")
    conv = _conv_with_pattern_hits(num_hits=80)
    extractor = EntityConceptExtractor(max_concepts_per_conversation=50)
    result = extractor.extract(conv)
    assert len(result.concepts) == 50
    assert "Concept cap applied" in caplog.text
    assert "cap=50" in caplog.text


def test_extractor_no_cap_when_zero(caplog: pytest.LogCaptureFixture) -> None:
    """Setting cap=0 disables truncation entirely."""
    caplog.set_level(logging.WARNING, logger="src.process.extractor")
    conv = _conv_with_pattern_hits(num_hits=300)
    extractor = EntityConceptExtractor(max_concepts_per_conversation=0)
    result = extractor.extract(conv)
    assert len(result.concepts) > 200
    assert "Concept cap applied" not in caplog.text


def test_extractor_under_cap_no_truncation(caplog: pytest.LogCaptureFixture) -> None:
    """Conversations with fewer concepts than the cap aren't truncated or warned."""
    caplog.set_level(logging.WARNING, logger="src.process.extractor")
    conv = _conv_with_pattern_hits(num_hits=10)
    extractor = EntityConceptExtractor(max_concepts_per_conversation=200)
    result = extractor.extract(conv)
    # 10 hits + topic concept ≤ 200
    assert len(result.concepts) <= 200
    assert "Concept cap applied" not in caplog.text


def test_merger_defensive_cap(caplog: pytest.LogCaptureFixture) -> None:
    """Merger truncates oversized inputs with a warning."""
    from src.llm.merger import ExtractionMerger

    caplog.set_level(logging.WARNING, logger="src.llm.merger")

    # Build a rule_result with 50 concepts and merger cap=20
    rule_concepts = [
        Concept(
            content=f"rule concept {i:03d} about a topic",
            concept_type=ConceptType.DECISION,
            source_conversation_id="c1",
        )
        for i in range(50)
    ]
    rule_result = ExtractionResult(
        conversation_id="c1", entities=[], concepts=rule_concepts
    )
    llm_result = ExtractionResult(conversation_id="c1", entities=[], concepts=[])

    merger = ExtractionMerger(
        similarity_threshold=0.85, max_concepts_per_conversation=20
    )
    merged = merger.merge(rule_result, llm_result)

    text = caplog.text
    assert "Merger defensive cap" in text
    assert "rule_result for conversation c1" in text
    # Merged should respect the cap on the rule side
    assert len(merged.concepts) <= 20


def test_merger_cap_zero_disables(caplog: pytest.LogCaptureFixture) -> None:
    """Merger with cap=0 doesn't truncate."""
    from src.llm.merger import ExtractionMerger

    caplog.set_level(logging.WARNING, logger="src.llm.merger")

    rule_concepts = [
        Concept(
            content=f"rule concept {i:03d} content",
            concept_type=ConceptType.DECISION,
            source_conversation_id="c1",
        )
        for i in range(30)
    ]
    rule_result = ExtractionResult(
        conversation_id="c1", entities=[], concepts=rule_concepts
    )
    llm_result = ExtractionResult(conversation_id="c1", entities=[], concepts=[])

    merger = ExtractionMerger(
        similarity_threshold=0.85, max_concepts_per_conversation=0
    )
    merger.merge(rule_result, llm_result)

    assert "Merger defensive cap" not in caplog.text
