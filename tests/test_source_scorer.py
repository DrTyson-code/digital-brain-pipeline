"""Tests for SourceScorer — conversation quality scoring."""

from datetime import datetime, timezone, timedelta

import pytest

from src.models.base import Platform
from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import ChatMessage, Conversation, Role
from src.process.extractor import ExtractionResult
from src.process.source_scorer import SourceScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conversation(
    messages_text: list,
    conv_id: str = "conv",
    topics: list = None,
    timestamps: list = None,
    model: str = None,
) -> Conversation:
    msgs = []
    for i, text in enumerate(messages_text):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        ts = timestamps[i] if timestamps else None
        msgs.append(
            ChatMessage(
                conversation_id=conv_id,
                role=role,
                content=text,
                platform=Platform.CLAUDE,
                timestamp=ts,
                model=model if role == Role.ASSISTANT else None,
            )
        )
    return Conversation(
        id=conv_id,
        messages=msgs,
        platform=Platform.CLAUDE,
        created_at=datetime.now(timezone.utc),
        topics=topics or [],
    )


def _make_extraction(conv_id: str, with_decision: bool = False) -> ExtractionResult:
    concepts = []
    if with_decision:
        concepts.append(
            Concept(
                concept_type=ConceptType.DECISION,
                content="Use PostgreSQL as primary database",
                source_conversation_id=conv_id,
            )
        )
    return ExtractionResult(conversation_id=conv_id, concepts=concepts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_score_empty_conversation_is_low():
    conv = _make_conversation([], conv_id="empty")
    scorer = SourceScorer()
    score = scorer.score(conv)
    assert 0.0 <= score <= 1.0
    assert score < 0.3  # Nothing in the conversation


def test_score_in_range():
    conv = _make_conversation(["Hello", "Hi there"], conv_id="c1")
    scorer = SourceScorer()
    score = scorer.score(conv)
    assert 0.0 <= score <= 1.0


def test_longer_conversation_scores_higher():
    short = _make_conversation(["Hi", "Hello"], conv_id="short")
    long_msgs = ["This is a very detailed message about an important technical topic. " * 5] * 20
    long = _make_conversation(long_msgs, conv_id="long")

    scorer = SourceScorer()
    assert scorer.score(long) > scorer.score(short)


def test_topics_raise_score():
    no_topics = _make_conversation(["Hello world"], conv_id="c1", topics=[])
    many_topics = _make_conversation(
        ["Hello world"], conv_id="c2",
        topics=["programming", "python", "databases", "docker", "testing"],
    )
    scorer = SourceScorer()
    assert scorer.score(many_topics) > scorer.score(no_topics)


def test_extraction_with_decisions_raises_score():
    conv = _make_conversation(["We decided something important"], conv_id="c1")
    scorer = SourceScorer()

    score_no_ext = scorer.score(conv, None)
    score_with_ext = scorer.score(conv, _make_extraction("c1", with_decision=True))
    assert score_with_ext > score_no_ext


def test_tool_call_raises_score():
    without_model = _make_conversation(["Help me", "Sure"], conv_id="c1", model=None)
    with_model = _make_conversation(["Help me", "Sure"], conv_id="c2", model="claude-3")
    scorer = SourceScorer()
    assert scorer.score(with_model) > scorer.score(without_model)


def test_session_duration_raises_score():
    t0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    short_ts = [t0, t0 + timedelta(seconds=30)]
    long_ts = [t0, t0 + timedelta(hours=1)]
    short = _make_conversation(["Hi", "Hello"], conv_id="c1", timestamps=short_ts)
    long = _make_conversation(["Hi", "Hello"], conv_id="c2", timestamps=long_ts)
    scorer = SourceScorer()
    assert scorer.score(long) > scorer.score(short)


def test_score_batch_returns_all_ids():
    convs = [_make_conversation(["Hello"], f"c{i}") for i in range(5)]
    scorer = SourceScorer()
    weights = scorer.score_batch(convs)
    assert set(weights.keys()) == {f"c{i}" for i in range(5)}
    for w in weights.values():
        assert 0.0 <= w <= 1.0


def test_score_batch_uses_extraction_map():
    conv = _make_conversation(["We decided to use Redis"], conv_id="cx")
    ext = _make_extraction("cx", with_decision=True)
    scorer = SourceScorer()
    weights_no_ext = scorer.score_batch([conv])
    weights_with_ext = scorer.score_batch([conv], [ext])
    assert weights_with_ext["cx"] > weights_no_ext["cx"]


def test_score_never_exceeds_one():
    # Saturate every signal
    t0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    ts = [t0 + timedelta(minutes=i) for i in range(40)]
    big_text = "word " * 300  # 300 words per message
    msgs = [big_text] * 40
    conv = _make_conversation(
        msgs,
        conv_id="max",
        topics=["a", "b", "c", "d", "e"],
        timestamps=ts,
        model="claude-3",
    )
    ext = _make_extraction("max", with_decision=True)
    scorer = SourceScorer()
    assert scorer.score(conv, ext) <= 1.0
