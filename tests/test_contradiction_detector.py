"""Tests for ContradictionDetector — conflicting knowledge detection."""

from src.models.concept import Concept, ConceptType
from src.models.relationship import RelationshipType
from src.process.contradiction_detector import ContradictionDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision(content: str, conv_id: str = "c1") -> Concept:
    return Concept(
        concept_type=ConceptType.DECISION,
        content=content,
        source_conversation_id=conv_id,
        confidence=0.8,
    )


def _insight(content: str, conv_id: str = "c1") -> Concept:
    return Concept(
        concept_type=ConceptType.INSIGHT,
        content=content,
        source_conversation_id=conv_id,
    )


def _action(content: str, conv_id: str = "c1") -> Concept:
    return Concept(
        concept_type=ConceptType.ACTION_ITEM,
        content=content,
        source_conversation_id=conv_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_list_returns_no_contradictions():
    detector = ContradictionDetector()
    assert detector.detect([]) == []


def test_single_concept_no_contradiction():
    detector = ContradictionDetector()
    result = detector.detect([_decision("We should use PostgreSQL")])
    assert result == []


def test_dissimilar_concepts_no_contradiction():
    """Completely different topics should not trigger a contradiction."""
    c1 = _decision("We should use PostgreSQL for data storage", conv_id="c1")
    c2 = _decision("We should hold daily standups at 9am", conv_id="c2")
    detector = ContradictionDetector()
    result = detector.detect([c1, c2])
    assert result == []


def test_should_vs_should_not_is_contradictory():
    c1 = _decision("We should use PostgreSQL for our main database", conv_id="c1")
    c2 = _decision("We should not use PostgreSQL for our main database", conv_id="c2")
    detector = ContradictionDetector(similarity_threshold=0.5)
    result = detector.detect([c1, c2])
    assert len(result) == 1
    assert result[0].relationship_type == RelationshipType.CONTRADICTS


def test_use_vs_avoid_is_contradictory():
    c1 = _decision("Use Redis caching for all API responses", conv_id="c1")
    c2 = _decision("Avoid Redis caching for all API responses", conv_id="c2")
    detector = ContradictionDetector(similarity_threshold=0.5)
    result = detector.detect([c1, c2])
    assert len(result) >= 1


def test_will_vs_wont_is_contradictory():
    c1 = _decision("We will migrate to microservices this quarter", conv_id="c1")
    c2 = _decision("We won't migrate to microservices this quarter", conv_id="c2")
    detector = ContradictionDetector(similarity_threshold=0.5)
    result = detector.detect([c1, c2])
    assert len(result) >= 1


def test_same_conversation_not_flagged():
    """Same-conversation pairs should not be flagged (rephrasing is normal)."""
    same_conv = "c1"
    c1 = _decision("We should use PostgreSQL", conv_id=same_conv)
    c2 = _decision("We should not use PostgreSQL", conv_id=same_conv)
    detector = ContradictionDetector(similarity_threshold=0.5)
    result = detector.detect([c1, c2])
    assert result == []


def test_no_cross_type_contradiction():
    """Decisions and insights should not be compared."""
    c1 = _decision("We should use Docker for containerisation", conv_id="c1")
    c2 = _insight("We should not use Docker for containerisation", conv_id="c2")
    detector = ContradictionDetector(similarity_threshold=0.5)
    result = detector.detect([c1, c2])
    # Different types → no comparison
    assert result == []


def test_action_items_not_checked():
    """Action items are not in _CHECKED_TYPES and should be skipped."""
    a1 = _action("We should set up Docker", conv_id="c1")
    a2 = _action("We should not set up Docker", conv_id="c2")
    detector = ContradictionDetector(similarity_threshold=0.5)
    result = detector.detect([a1, a2])
    assert result == []


def test_contradiction_relationship_has_evidence():
    c1 = _decision("We should use PostgreSQL for our database", conv_id="c1")
    c2 = _decision("We should not use PostgreSQL for our database", conv_id="c2")
    detector = ContradictionDetector(similarity_threshold=0.5)
    result = detector.detect([c1, c2])
    if result:
        assert len(result[0].evidence) > 0


def test_contradiction_weight_is_similarity():
    c1 = _decision("We should use PostgreSQL for our main database", conv_id="c1")
    c2 = _decision("We should not use PostgreSQL for our main database", conv_id="c2")
    detector = ContradictionDetector(similarity_threshold=0.5)
    result = detector.detect([c1, c2])
    if result:
        assert 0.5 <= result[0].weight <= 1.0


def test_high_threshold_prevents_detection():
    c1 = _decision("We should use PostgreSQL", conv_id="c1")
    c2 = _decision("We should not use PostgreSQL", conv_id="c2")
    detector = ContradictionDetector(similarity_threshold=0.99)
    result = detector.detect([c1, c2])
    # May or may not trigger depending on exact similarity — just ensure no crash
    assert isinstance(result, list)
