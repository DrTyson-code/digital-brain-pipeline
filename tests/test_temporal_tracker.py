"""Tests for TemporalTracker — lifecycle status of decisions/action items."""

from datetime import datetime, timezone, timedelta

from src.models.base import Platform
from src.models.concept import Concept, ConceptType
from src.models.message import ChatMessage, Conversation, Role
from src.process.temporal_tracker import TemporalTracker, ConceptStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conv(conv_id: str, created: datetime) -> Conversation:
    msg = ChatMessage(
        conversation_id=conv_id,
        role=Role.USER,
        content="hello",
        platform=Platform.CLAUDE,
    )
    return Conversation(
        id=conv_id,
        messages=[msg],
        platform=Platform.CLAUDE,
        created_at=created,
    )


def _decision(content: str, conv_id: str = None) -> Concept:
    return Concept(
        concept_type=ConceptType.DECISION,
        content=content,
        source_conversation_id=conv_id,
    )


def _action(content: str, conv_id: str = None) -> Concept:
    return Concept(
        concept_type=ConceptType.ACTION_ITEM,
        content=content,
        source_conversation_id=conv_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_concepts_get_a_status():
    concepts = [
        _decision("Use PostgreSQL"),
        _action("Set up CI pipeline"),
        Concept(concept_type=ConceptType.INSIGHT, content="Indexes matter a lot"),
    ]
    tracker = TemporalTracker()
    statuses = tracker.track(concepts)
    assert len(statuses) == len(concepts)
    for concept in concepts:
        assert concept.id in statuses


def test_new_concept_is_active():
    d = _decision("Use PostgreSQL as primary database")
    tracker = TemporalTracker()
    statuses = tracker.track([d])
    assert statuses[d.id].status == "active"


def test_supersession_detection_by_keyword():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=7)

    conv_old = _conv("c_old", t0)
    conv_new = _conv("c_new", t1)

    older = _decision("Use PostgreSQL for the main database", conv_id="c_old")
    newer = _decision(
        "Actually switched to MySQL instead of PostgreSQL for the main database",
        conv_id="c_new",
    )

    tracker = TemporalTracker(similarity_threshold=0.35)
    statuses = tracker.track([older, newer], [conv_old, conv_new])

    assert statuses[older.id].status == "superseded"
    assert statuses[older.id].superseded_by_id == newer.id
    assert statuses[newer.id].status == "active"


def test_no_supersession_for_dissimilar_content():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=7)

    conv_old = _conv("c_old", t0)
    conv_new = _conv("c_new", t1)

    older = _decision("Use PostgreSQL for the main database", conv_id="c_old")
    newer = _decision("Hold daily stand-up meetings at 9am", conv_id="c_new")

    tracker = TemporalTracker()
    statuses = tracker.track([older, newer], [conv_old, conv_new])

    # Not similar → no supersession
    assert statuses[older.id].status == "active"


def test_no_cross_type_supersession():
    """Decisions should not supersede action items."""
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=1)

    conv_old = _conv("c_old", t0)
    conv_new = _conv("c_new", t1)

    older = _decision("Use Redis for caching", conv_id="c_old")
    newer = _action(
        "Actually switched to Memcached instead of Redis for caching",
        conv_id="c_new",
    )

    tracker = TemporalTracker(similarity_threshold=0.35)
    statuses = tracker.track([older, newer], [conv_old, conv_new])

    # Different types → no supersession
    assert statuses[older.id].status == "active"


def test_valid_from_uses_conversation_date():
    t0 = datetime(2024, 6, 15, tzinfo=timezone.utc)
    conv = _conv("conv1", t0)
    d = _decision("Use async processing", conv_id="conv1")

    tracker = TemporalTracker()
    statuses = tracker.track([d], [conv])

    assert statuses[d.id].valid_from == t0


def test_valid_from_falls_back_to_concept_created_at():
    d = _decision("Use async processing")  # no source_conversation_id
    tracker = TemporalTracker()
    statuses = tracker.track([d])

    assert statuses[d.id].valid_from is not None


def test_superseded_by_title_is_set():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=1)
    conv_old = _conv("c_old", t0)
    conv_new = _conv("c_new", t1)

    older = _decision("Use PostgreSQL for the main store", conv_id="c_old")
    newer = _decision(
        "Actually switched to MySQL instead of PostgreSQL for the main store",
        conv_id="c_new",
    )

    tracker = TemporalTracker(similarity_threshold=0.35)
    statuses = tracker.track([older, newer], [conv_old, conv_new])

    if statuses[older.id].status == "superseded":
        assert statuses[older.id].superseded_by_title is not None
        assert len(statuses[older.id].superseded_by_title) > 0


def test_already_superseded_is_not_superseded_again():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=1)
    t2 = t1 + timedelta(days=1)
    conv0 = _conv("c0", t0)
    conv1 = _conv("c1", t1)
    conv2 = _conv("c2", t2)

    d0 = _decision("Use PostgreSQL for storage", conv_id="c0")
    d1 = _decision(
        "Actually switched to MySQL instead of PostgreSQL for storage",
        conv_id="c1",
    )
    d2 = _decision(
        "Actually switched to SQLite instead of MySQL for storage",
        conv_id="c2",
    )

    tracker = TemporalTracker(similarity_threshold=0.30)
    statuses = tracker.track([d0, d1, d2], [conv0, conv1, conv2])

    # d0 should be superseded by d1 (first match in chronological order)
    if statuses[d0.id].status == "superseded":
        assert statuses[d0.id].superseded_by_id == d1.id
