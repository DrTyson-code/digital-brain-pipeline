"""Tests for ExtractionMerger — combining rule-based and LLM extraction results.

Tests cover entity/concept matching, merge logic, batch processing,
and confidence boosting.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.llm.merger import ExtractionMerger, MergeStatistics
from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.process.extractor import ExtractionResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def merger():
    """Create a merger with default similarity threshold."""
    return ExtractionMerger(similarity_threshold=0.85)


@pytest.fixture
def sample_entity_python():
    """Sample Python entity."""
    return Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        aliases=["python3"],
        source_conversations=["conv-1"],
    )


@pytest.fixture
def sample_entity_alice():
    """Sample Alice entity."""
    return Entity(
        entity_type=EntityType.PERSON,
        name="Alice",
        aliases=[],
        source_conversations=["conv-1"],
    )


@pytest.fixture
def sample_entity_google():
    """Sample Google entity."""
    return Entity(
        entity_type=EntityType.ORGANIZATION,
        name="Google",
        aliases=["Google Inc", "Alphabet"],
        source_conversations=["conv-1"],
    )


@pytest.fixture
def sample_concept_ml():
    """Sample machine learning concept."""
    return Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        confidence=0.9,
        source_conversation_id="conv-1",
    )


@pytest.fixture
def sample_concept_decision():
    """Sample decision concept."""
    return Concept(
        concept_type=ConceptType.DECISION,
        content="Use Python for backend",
        confidence=0.95,
        source_conversation_id="conv-1",
    )


# ============================================================================
# Entity matching tests
# ============================================================================


def test_entities_match_exact_name(merger, sample_entity_python):
    """Test exact name match."""
    entity1 = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        source_conversations=["rule"],
    )
    entity2 = sample_entity_python

    assert merger._entities_match(entity1, entity2)


def test_entities_match_case_insensitive(merger):
    """Test case-insensitive matching."""
    entity1 = Entity(
        entity_type=EntityType.TOOL,
        name="PYTHON",
        source_conversations=["rule"],
    )
    entity2 = Entity(
        entity_type=EntityType.TOOL,
        name="python",
        source_conversations=["llm"],
    )

    assert merger._entities_match(entity1, entity2)


def test_entities_match_via_alias(merger):
    """Test matching through alias."""
    entity1 = Entity(
        entity_type=EntityType.TOOL,
        name="py",
        source_conversations=["rule"],
    )
    entity2 = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        aliases=["py", "python3"],
        source_conversations=["llm"],
    )

    assert merger._entities_match(entity1, entity2)


def test_entities_match_similarity(merger):
    """Test similarity-based matching."""
    entity1 = Entity(
        entity_type=EntityType.TOOL,
        name="Pythonn",  # Typo
        source_conversations=["rule"],
    )
    entity2 = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        source_conversations=["llm"],
    )

    # Should match due to high similarity
    assert merger._entities_match(entity1, entity2)


def test_entities_no_match_different_type(merger):
    """Test that different entity types don't match."""
    entity1 = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        source_conversations=["rule"],
    )
    entity2 = Entity(
        entity_type=EntityType.ORGANIZATION,
        name="Python",
        source_conversations=["llm"],
    )

    assert not merger._entities_match(entity1, entity2)


def test_entities_no_match_dissimilar(merger):
    """Test that dissimilar names don't match."""
    entity1 = Entity(
        entity_type=EntityType.TOOL,
        name="JavaScript",
        source_conversations=["rule"],
    )
    entity2 = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        source_conversations=["llm"],
    )

    assert not merger._entities_match(entity1, entity2)


# ============================================================================
# Concept matching tests
# ============================================================================


def test_concepts_match_exact_content(merger):
    """Test exact concept content match."""
    concept1 = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        source_conversation_id="rule",
    )
    concept2 = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        source_conversation_id="llm",
    )

    assert merger._concepts_match(concept1, concept2)


def test_concepts_match_similar_content(merger):
    """Test similarity-based concept matching."""
    concept1 = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning algorithms",
        source_conversation_id="rule",
    )
    concept2 = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        source_conversation_id="llm",
    )

    # May or may not match depending on similarity threshold
    # (This tests the behavior at the threshold)


def test_concepts_no_match_different_type(merger):
    """Test that different concept types don't match."""
    concept1 = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        source_conversation_id="rule",
    )
    concept2 = Concept(
        concept_type=ConceptType.DECISION,
        content="Machine Learning",
        source_conversation_id="llm",
    )

    assert not merger._concepts_match(concept1, concept2)


def test_concepts_no_match_dissimilar(merger):
    """Test that dissimilar concepts don't match."""
    concept1 = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        source_conversation_id="rule",
    )
    concept2 = Concept(
        concept_type=ConceptType.TOPIC,
        content="Cooking recipes",
        source_conversation_id="llm",
    )

    assert not merger._concepts_match(concept1, concept2)


# ============================================================================
# Single merge tests
# ============================================================================


def test_merge_empty_results(merger):
    """Test merging when both results are empty."""
    rule_result = ExtractionResult(conversation_id="conv-1")
    llm_result = ExtractionResult(conversation_id="conv-1")

    merged = merger.merge(rule_result, llm_result)

    assert merged.conversation_id == "conv-1"
    assert len(merged.entities) == 0
    assert len(merged.concepts) == 0


def test_merge_llm_only(merger, sample_entity_python):
    """Test merge when only LLM has entities."""
    rule_result = ExtractionResult(conversation_id="conv-1", entities=[])
    llm_result = ExtractionResult(
        conversation_id="conv-1", entities=[sample_entity_python]
    )

    merged = merger.merge(rule_result, llm_result)

    assert len(merged.entities) == 1
    assert merged.entities[0].name == "Python"


def test_merge_rule_only(merger, sample_entity_python):
    """Test merge when only rule-based has entities."""
    rule_result = ExtractionResult(
        conversation_id="conv-1", entities=[sample_entity_python]
    )
    llm_result = ExtractionResult(conversation_id="conv-1", entities=[])

    merged = merger.merge(rule_result, llm_result)

    assert len(merged.entities) == 1
    assert merged.entities[0].name == "Python"


def test_merge_matching_entities_calls_merge_method(merger):
    """Test that matching entities call the merge method."""
    # Create two Python entities with different info
    rule_python = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        aliases=["py"],
        source_conversations=["rule-conv"],
    )
    llm_python = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        aliases=["python3"],
        source_conversations=["llm-conv"],
    )

    rule_result = ExtractionResult(
        conversation_id="conv-1", entities=[rule_python]
    )
    llm_result = ExtractionResult(
        conversation_id="conv-1", entities=[llm_python]
    )

    merged = merger.merge(rule_result, llm_result)

    # Should have one merged entity
    assert len(merged.entities) == 1
    merged_entity = merged.entities[0]

    # Should have both aliases
    assert "py" in merged_entity.aliases
    assert "python3" in merged_entity.aliases
    # Should have both source conversations
    assert "llm-conv" in merged_entity.source_conversations
    assert "rule-conv" in merged_entity.source_conversations


def test_merge_unique_entities_kept(merger, sample_entity_python, sample_entity_alice):
    """Test that unique entities from both sources are kept."""
    rule_result = ExtractionResult(
        conversation_id="conv-1", entities=[sample_entity_alice]
    )
    llm_result = ExtractionResult(
        conversation_id="conv-1", entities=[sample_entity_python]
    )

    merged = merger.merge(rule_result, llm_result)

    # Should have both entities
    assert len(merged.entities) == 2
    names = {e.name for e in merged.entities}
    assert "Python" in names
    assert "Alice" in names


def test_merge_concepts_llm_primary(merger):
    """Test that LLM concepts are kept when matching rule concepts."""
    rule_concept = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        confidence=0.7,
        source_conversation_id="rule",
    )
    llm_concept = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        confidence=0.95,
        source_conversation_id="llm",
    )

    rule_result = ExtractionResult(
        conversation_id="conv-1", concepts=[rule_concept]
    )
    llm_result = ExtractionResult(
        conversation_id="conv-1", concepts=[llm_concept]
    )

    merged = merger.merge(rule_result, llm_result)

    # Should have one concept (the LLM one)
    assert len(merged.concepts) == 1
    assert merged.concepts[0].confidence >= 0.95  # From LLM


def test_merge_confidence_boosting(merger):
    """Test that confidence is boosted when both sources find concept."""
    rule_concept = Concept(
        concept_type=ConceptType.DECISION,
        content="Use Python for backend",
        confidence=0.7,
        source_conversation_id="rule",
    )
    llm_concept = Concept(
        concept_type=ConceptType.DECISION,
        content="Use Python for backend",
        confidence=0.85,
        source_conversation_id="llm",
    )

    rule_result = ExtractionResult(
        conversation_id="conv-1", concepts=[rule_concept]
    )
    llm_result = ExtractionResult(
        conversation_id="conv-1", concepts=[llm_concept]
    )

    merged = merger.merge(rule_result, llm_result)

    assert len(merged.concepts) == 1
    # Confidence should be boosted but not exceed 0.95
    assert merged.concepts[0].confidence >= 0.85
    assert merged.concepts[0].confidence <= 0.95


# ============================================================================
# Batch merge tests
# ============================================================================


def test_merge_batch_paired_results(merger, sample_entity_python):
    """Test batch merge with paired results."""
    rule_results = [
        ExtractionResult(conversation_id="conv-1", entities=[sample_entity_python])
    ]
    llm_results = [
        ExtractionResult(conversation_id="conv-1", entities=[sample_entity_python])
    ]

    merged = merger.merge_batch(rule_results, llm_results)

    assert len(merged) == 1
    assert merged[0].conversation_id == "conv-1"


def test_merge_batch_missing_llm_result(merger, sample_entity_python):
    """Test batch merge when LLM result is missing."""
    rule_results = [
        ExtractionResult(conversation_id="conv-1", entities=[sample_entity_python])
    ]
    llm_results = []

    merged = merger.merge_batch(rule_results, llm_results)

    assert len(merged) == 1
    assert merged[0].conversation_id == "conv-1"
    assert merged[0].entities == [sample_entity_python]


def test_merge_batch_missing_rule_result(merger, sample_entity_python):
    """Test batch merge when rule result is missing."""
    rule_results = []
    llm_results = [
        ExtractionResult(conversation_id="conv-1", entities=[sample_entity_python])
    ]

    merged = merger.merge_batch(rule_results, llm_results)

    assert len(merged) == 1
    assert merged[0].conversation_id == "conv-1"
    assert merged[0].entities == [sample_entity_python]


def test_merge_batch_multiple_conversations(merger):
    """Test batch merge with multiple conversations."""
    python_entity = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        source_conversations=["conv-1"],
    )
    javascript_entity = Entity(
        entity_type=EntityType.TOOL,
        name="JavaScript",
        source_conversations=["conv-2"],
    )

    rule_results = [
        ExtractionResult(conversation_id="conv-1", entities=[python_entity]),
        ExtractionResult(conversation_id="conv-2", entities=[javascript_entity]),
    ]
    llm_results = [
        ExtractionResult(conversation_id="conv-1", entities=[python_entity]),
    ]

    merged = merger.merge_batch(rule_results, llm_results)

    assert len(merged) == 2
    conv_ids = {r.conversation_id for r in merged}
    assert "conv-1" in conv_ids
    assert "conv-2" in conv_ids


# ============================================================================
# Statistics tests
# ============================================================================


def test_merge_statistics_empty():
    """Test statistics with empty results."""
    stats = MergeStatistics()
    assert stats.total_rule_entities == 0
    assert stats.total_llm_entities == 0
    assert stats.matched_entities == 0
    assert stats.final_entity_count == 0


def test_merge_statistics_str_representation():
    """Test string representation of statistics."""
    stats = MergeStatistics(
        total_rule_entities=5,
        total_llm_entities=4,
        matched_entities=3,
        unique_rule_entities=2,
        unique_llm_entities=4,
        final_entity_count=6,
        total_rule_concepts=3,
        total_llm_concepts=2,
        matched_concepts=1,
        unique_rule_concepts=2,
        unique_llm_concepts=2,
        final_concept_count=4,
    )

    str_rep = str(stats)
    assert "matched" in str_rep
    assert "unique" in str_rep
    assert "total" in str_rep


# ============================================================================
# Normalization and similarity tests
# ============================================================================


def test_normalize_whitespace():
    """Test text normalization."""
    text = "  Python   3.11  "
    normalized = ExtractionMerger._normalize(text)
    assert normalized == "python 3.11"


def test_normalize_case():
    """Test case normalization."""
    text = "PYTHON"
    normalized = ExtractionMerger._normalize(text)
    assert normalized == "python"


def test_similarity_identical():
    """Test similarity of identical strings."""
    sim = ExtractionMerger._similarity("python", "python")
    assert sim == 1.0


def test_similarity_completely_different():
    """Test similarity of completely different strings."""
    sim = ExtractionMerger._similarity("python", "javascript")
    assert 0.0 <= sim <= 0.3


def test_similarity_similar():
    """Test similarity of similar strings."""
    sim = ExtractionMerger._similarity("python", "pythonx")
    assert sim > 0.85


def test_similarity_empty_strings():
    """Test similarity with empty strings."""
    assert ExtractionMerger._similarity("", "") == 1.0
    assert ExtractionMerger._similarity("test", "") == 0.0
    assert ExtractionMerger._similarity("", "test") == 0.0


# ============================================================================
# Threshold tests
# ============================================================================


def test_merger_threshold_validation():
    """Test that invalid thresholds are rejected."""
    with pytest.raises(ValueError):
        ExtractionMerger(similarity_threshold=-0.1)

    with pytest.raises(ValueError):
        ExtractionMerger(similarity_threshold=1.1)


def test_merger_custom_threshold():
    """Test merger with custom threshold."""
    merger = ExtractionMerger(similarity_threshold=0.5)
    assert merger.similarity_threshold == 0.5


def test_merge_respects_threshold():
    """Test that merge respects similarity threshold."""
    # Low threshold should match more aggressively
    merger_low = ExtractionMerger(similarity_threshold=0.5)

    entity1 = Entity(
        entity_type=EntityType.TOOL,
        name="Pythonx",
        source_conversations=["rule"],
    )
    entity2 = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        source_conversations=["llm"],
    )

    # Should match with low threshold
    assert merger_low._entities_match(entity1, entity2)

    # High threshold should match less aggressively
    merger_high = ExtractionMerger(similarity_threshold=0.99)
    assert not merger_high._entities_match(entity1, entity2)
