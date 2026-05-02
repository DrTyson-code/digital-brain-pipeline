"""Smoke tests for audit-5 observability log additions.

Verifies that each curation stage emits the new per-stage summary logs
(distribution / breakdown / examples) that calibration depends on.
These tests don't pin exact log strings — just that the diagnostic
log lines fire when the stage runs on representative data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

import pytest

from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import ChatMessage, Conversation
from src.models.relationship import Relationship, RelationshipType
from src.process.contradiction_detector import ContradictionDetector
from src.process.entity_resolver import EntityResolver
from src.process.extractor import ExtractionResult
from src.process.review_queue import ReviewQueueGenerator
from src.process.source_scorer import SourceScorer
from src.process.temporal_tracker import ConceptStatus, TemporalTracker


def _conv(msg_count: int = 5, words_each: int = 30, conv_id: str = "c1") -> Conversation:
    """Build a minimal Conversation with the required signal."""
    msgs = [
        ChatMessage(
            conversation_id=conv_id,
            role="user" if i % 2 == 0 else "assistant",
            content=" ".join(["word"] * words_each),
            timestamp=datetime(2026, 5, 1, 12, i, tzinfo=timezone.utc),
            platform="claude",
        )
        for i in range(msg_count)
    ]
    return Conversation(
        id=conv_id,
        title=f"conv {conv_id}",
        platform="claude",
        created_at=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        messages=msgs,
        topics=["programming"],
    )


def test_source_scorer_logs_histogram(caplog: pytest.LogCaptureFixture) -> None:
    """Source scorer should log avg/min/P50/P95/max + a 5-bin histogram."""
    caplog.set_level(logging.INFO, logger="src.process.source_scorer")
    scorer = SourceScorer()
    convs = [_conv(msg_count=i + 1, words_each=20, conv_id=f"c{i}") for i in range(10)]
    weights = scorer.score_batch(convs)
    assert len(weights) == 10
    text = caplog.text
    assert "Source scoring:" in text
    assert "P50=" in text and "P95=" in text
    assert "Source weight distribution" in text
    assert "[0.0-0.2]=" in text


def test_entity_resolver_logs_breakdown(caplog: pytest.LogCaptureFixture) -> None:
    """Entity resolver should log merge rate + per-type breakdown when merges happen."""
    caplog.set_level(logging.INFO, logger="src.process.entity_resolver")
    e1 = Entity(name="PostgreSQL", entity_type=EntityType.TOOL)
    e2 = Entity(name="Postgres", entity_type=EntityType.TOOL)  # should merge with e1
    e3 = Entity(name="Redis", entity_type=EntityType.TOOL)  # distinct
    extraction = ExtractionResult(
        conversation_id="c1", entities=[e1, e2, e3], concepts=[]
    )
    resolver = EntityResolver(similarity_threshold=0.85)
    _, merge_map = resolver.resolve([extraction])
    assert len(merge_map) >= 1  # PostgreSQL/Postgres should merge
    text = caplog.text
    assert "Entity resolution:" in text
    assert "merge rate" in text
    # Either type breakdown or example log fires when merges happen
    assert "Entity merges by type" in text or "Entity merge example" in text


def test_entity_resolver_logs_zero_clean(caplog: pytest.LogCaptureFixture) -> None:
    """No-merge case should still log cleanly without breakdown noise."""
    caplog.set_level(logging.INFO, logger="src.process.entity_resolver")
    e1 = Entity(name="Apple", entity_type=EntityType.ORGANIZATION)
    e2 = Entity(name="Microsoft", entity_type=EntityType.ORGANIZATION)
    extraction = ExtractionResult(
        conversation_id="c1", entities=[e1, e2], concepts=[]
    )
    resolver = EntityResolver(similarity_threshold=0.85)
    _, merge_map = resolver.resolve([extraction])
    assert len(merge_map) == 0
    assert "Entity resolution:" in caplog.text
    assert "0 merged" in caplog.text


def test_contradiction_detector_logs_zero_explicit(caplog: pytest.LogCaptureFixture) -> None:
    """Detector with no contradictions should log zero explicitly with threshold context."""
    caplog.set_level(logging.INFO, logger="src.process.contradiction_detector")
    c1 = Concept(
        content="We should use Postgres for the DB",
        concept_type=ConceptType.DECISION,
        source_conversation_id="c1",
    )
    c2 = Concept(
        content="Decided to deploy on Vercel",
        concept_type=ConceptType.DECISION,
        source_conversation_id="c2",
    )
    detector = ContradictionDetector()
    result = detector.detect([c1, c2])
    assert result == []
    text = caplog.text
    assert "0 contradictions found" in text
    assert "threshold=" in text  # explicit threshold context
    assert "persistent zero" in text  # calibration-aware framing


def test_temporal_tracker_logs_status_distribution(caplog: pytest.LogCaptureFixture) -> None:
    """Temporal tracker should log status distribution across all concepts."""
    caplog.set_level(logging.INFO, logger="src.process.temporal_tracker")
    concepts = [
        Concept(
            content=f"Decision {i}",
            concept_type=ConceptType.DECISION,
            source_conversation_id=f"c{i}",
        )
        for i in range(3)
    ]
    tracker = TemporalTracker()
    statuses = tracker.track(concepts)
    assert len(statuses) == 3
    text = caplog.text
    assert "Temporal tracking:" in text
    assert "status distribution" in text
    assert "active=" in text


def test_review_queue_logs_breakdowns(caplog: pytest.LogCaptureFixture) -> None:
    """Review queue should log priority/object_type/reason breakdowns when items present."""
    caplog.set_level(logging.INFO, logger="src.process.review_queue")
    concepts = [
        Concept(content="Low conf", concept_type=ConceptType.INSIGHT, confidence=0.3),
        Concept(content="Mid conf", concept_type=ConceptType.INSIGHT, confidence=0.5),
    ]
    gen = ReviewQueueGenerator(min_review_confidence=0.6)
    items, review_ids = gen.generate(
        entities=[],
        concepts=concepts,
        contradictions=[],
    )
    assert len(items) == 2
    text = caplog.text
    assert "Review queue:" in text
    assert "by priority:" in text
    assert "by object_type:" in text
    assert "by reason category:" in text
    assert "low_confidence=" in text
