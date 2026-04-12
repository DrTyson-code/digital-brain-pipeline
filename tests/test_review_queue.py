"""Tests for ReviewQueueGenerator and CurationResult."""

import tempfile
from pathlib import Path

import pytest

from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.relationship import Relationship, RelationshipType
from src.process.review_queue import CurationResult, ReviewItem, ReviewQueueGenerator
from src.process.temporal_tracker import ConceptStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _concept(
    content: str,
    confidence: float = 0.8,
    concept_type: ConceptType = ConceptType.DECISION,
    conv_id: str = "c1",
) -> Concept:
    return Concept(
        concept_type=concept_type,
        content=content,
        confidence=confidence,
        source_conversation_id=conv_id,
    )


def _entity(name: str, etype: EntityType = EntityType.TOOL) -> Entity:
    return Entity(entity_type=etype, name=name, source_conversations=["c1"])


def _contradiction(src_id: str, tgt_id: str) -> Relationship:
    return Relationship(
        source_id=src_id,
        target_id=tgt_id,
        relationship_type=RelationshipType.CONTRADICTS,
    )


# ---------------------------------------------------------------------------
# Tests: ReviewQueueGenerator.generate
# ---------------------------------------------------------------------------


def test_empty_inputs_returns_empty_queue():
    gen = ReviewQueueGenerator()
    items, review_ids = gen.generate([], [], [], None, None)
    assert items == []
    assert review_ids == set()


def test_low_confidence_concept_flagged():
    low = _concept("Low confidence thing", confidence=0.4)
    high = _concept("High confidence thing", confidence=0.9)
    gen = ReviewQueueGenerator(min_review_confidence=0.6)
    items, review_ids = gen.generate([], [low, high], [], None, None)

    assert low.id in review_ids
    assert high.id not in review_ids


def test_low_confidence_priority():
    very_low = _concept("Barely extracted", confidence=0.3)
    medium_low = _concept("Possibly extracted", confidence=0.55)
    gen = ReviewQueueGenerator(min_review_confidence=0.6)
    items, _ = gen.generate([], [very_low, medium_low], [], None, None)

    priorities = {i.object_id: i.priority for i in items}
    assert priorities[very_low.id] == "low"
    assert priorities[medium_low.id] == "normal"


def test_merged_entity_flagged():
    canonical = _entity("Python")
    merged = _entity("python")  # will be in merge_map as merged
    merge_map = {merged.id: canonical.id}

    gen = ReviewQueueGenerator()
    items, review_ids = gen.generate([canonical], [], [], merge_map, None)

    assert canonical.id in review_ids


def test_contradiction_both_sides_flagged():
    c1 = _concept("Use PostgreSQL", conv_id="c1")
    c2 = _concept("Do not use PostgreSQL", conv_id="c2")
    rel = _contradiction(c1.id, c2.id)

    gen = ReviewQueueGenerator()
    items, review_ids = gen.generate([], [c1, c2], [rel], None, None)

    assert c1.id in review_ids
    assert c2.id in review_ids


def test_contradiction_items_have_high_priority():
    c1 = _concept("Use PostgreSQL", conv_id="c1")
    c2 = _concept("Do not use PostgreSQL", conv_id="c2")
    rel = _contradiction(c1.id, c2.id)

    gen = ReviewQueueGenerator()
    items, _ = gen.generate([], [c1, c2], [rel], None, None)

    priorities = {i.object_id: i.priority for i in items}
    assert priorities.get(c1.id) == "high" or priorities.get(c2.id) == "high"


def test_superseded_concept_flagged():
    c = _concept("Use PostgreSQL", confidence=0.9)
    status = ConceptStatus(
        concept_id=c.id,
        status="superseded",
    )

    gen = ReviewQueueGenerator(min_review_confidence=0.6)
    items, review_ids = gen.generate([], [c], [], None, {c.id: status})

    assert c.id in review_ids


def test_active_concept_not_flagged_by_status():
    c = _concept("Use PostgreSQL", confidence=0.9)
    status = ConceptStatus(concept_id=c.id, status="active")

    gen = ReviewQueueGenerator(min_review_confidence=0.6)
    items, review_ids = gen.generate([], [c], [], None, {c.id: status})

    # High confidence + active → should not be in review queue
    assert c.id not in review_ids


def test_no_duplicate_review_items_for_same_id():
    """Each object ID should appear at most once in the review queue."""
    c = _concept("A thing", confidence=0.5)
    status = ConceptStatus(concept_id=c.id, status="superseded")

    gen = ReviewQueueGenerator(min_review_confidence=0.6)
    items, review_ids = gen.generate([], [c], [], None, {c.id: status})

    # Even though c is both low-confidence AND superseded, it should appear
    # only once in review_ids (the set deduplicates automatically).
    assert len(review_ids) == 1


# ---------------------------------------------------------------------------
# Tests: CurationResult defaults
# ---------------------------------------------------------------------------


def test_curation_result_defaults():
    result = CurationResult()
    assert result.source_weights == {}
    assert result.merge_map == {}
    assert result.concept_statuses == {}
    assert result.contradictions == []
    assert result.review_items == []
    assert result.review_ids == set()
    assert result.corrections == []


# ---------------------------------------------------------------------------
# Tests: ReviewQueueGenerator.load_corrections
# ---------------------------------------------------------------------------


def test_load_corrections_missing_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir) / "vault"
        vault.mkdir()
        gen = ReviewQueueGenerator()
        corrections = gen.load_corrections(vault)
        assert corrections == []


def test_load_corrections_single_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir) / "vault"
        corrections_dir = vault / "_corrections"
        corrections_dir.mkdir(parents=True)

        corr_file = corrections_dir / "fix1.yaml"
        corr_file.write_text(
            "original_id: abc123\ncorrection_type: rename\ncorrected_value: NewName\n",
            encoding="utf-8",
        )

        gen = ReviewQueueGenerator()
        corrections = gen.load_corrections(vault)
        assert len(corrections) == 1
        assert corrections[0]["original_id"] == "abc123"
        assert corrections[0]["correction_type"] == "rename"


def test_load_corrections_list_yaml():
    """A YAML file containing a list of corrections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir) / "vault"
        corrections_dir = vault / "_corrections"
        corrections_dir.mkdir(parents=True)

        corr_file = corrections_dir / "batch.yaml"
        corr_file.write_text(
            "- original_id: id1\n  correction_type: delete\n  corrected_value: ~\n"
            "- original_id: id2\n  correction_type: rename\n  corrected_value: Better Name\n",
            encoding="utf-8",
        )

        gen = ReviewQueueGenerator()
        corrections = gen.load_corrections(vault)
        assert len(corrections) == 2


def test_load_corrections_bad_file_skipped():
    """Malformed YAML files should be skipped with a warning, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir) / "vault"
        corrections_dir = vault / "_corrections"
        corrections_dir.mkdir(parents=True)

        bad_file = corrections_dir / "bad.yaml"
        bad_file.write_text("{{invalid: yaml: content: [\n", encoding="utf-8")

        gen = ReviewQueueGenerator()
        corrections = gen.load_corrections(vault)
        # Malformed file is skipped; no crash
        assert corrections == []
