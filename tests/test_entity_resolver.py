"""Tests for EntityResolver — corpus-level fuzzy entity deduplication."""

from src.models.base import Platform
from src.models.entity import Entity, EntityType
from src.process.entity_resolver import EntityResolver
from src.process.extractor import ExtractionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, conv_id: str = "c1") -> Entity:
    return Entity(
        entity_type=EntityType.TOOL, name=name, source_conversations=[conv_id]
    )


def _person(name: str, conv_id: str = "c1") -> Entity:
    return Entity(
        entity_type=EntityType.PERSON, name=name, source_conversations=[conv_id]
    )


def _ext(conv_id: str, entities: list) -> ExtractionResult:
    return ExtractionResult(conversation_id=conv_id, entities=entities)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_duplicates_unchanged():
    """Unique entities should pass through untouched."""
    e1 = _tool("Python", "c1")
    e2 = _tool("Docker", "c2")
    extractions = [_ext("c1", [e1]), _ext("c2", [e2])]

    resolver = EntityResolver(similarity_threshold=0.85)
    updated, merge_map = resolver.resolve(extractions)

    assert merge_map == {}
    all_entities = [e for ex in updated for e in ex.entities]
    assert len(all_entities) == 2


def test_exact_name_duplicate_merged():
    """Two entities with the same name should be merged into one."""
    e1 = _tool("Python", "c1")
    e2 = _tool("Python", "c2")
    extractions = [_ext("c1", [e1]), _ext("c2", [e2])]

    resolver = EntityResolver(similarity_threshold=0.85)
    updated, merge_map = resolver.resolve(extractions)

    assert len(merge_map) == 1
    all_entities = [e for ex in updated for e in ex.entities]
    assert len(all_entities) == 1
    # Canonical entity should have both source conversations
    assert "c1" in all_entities[0].source_conversations
    assert "c2" in all_entities[0].source_conversations


def test_fuzzy_name_duplicate_merged():
    """Entities with similar but not identical names should be merged."""
    e1 = _person("John Smith", "c1")
    e2 = _person("John Smyth", "c2")  # typo variant
    extractions = [_ext("c1", [e1]), _ext("c2", [e2])]

    resolver = EntityResolver(similarity_threshold=0.80)
    updated, merge_map = resolver.resolve(extractions)

    assert len(merge_map) == 1


def test_no_cross_type_merging():
    """Entities of different types must never be merged even if names match."""
    e1 = _tool("Python", "c1")
    e2 = _person("Python", "c2")  # same name, different type
    extractions = [_ext("c1", [e1]), _ext("c2", [e2])]

    resolver = EntityResolver(similarity_threshold=0.85)
    updated, merge_map = resolver.resolve(extractions)

    assert merge_map == {}
    all_entities = [e for ex in updated for e in ex.entities]
    assert len(all_entities) == 2


def test_high_threshold_prevents_fuzzy_merge():
    """With a very high threshold, fuzzy variants should not merge."""
    e1 = _person("John Smith", "c1")
    e2 = _person("John Smyth", "c2")
    extractions = [_ext("c1", [e1]), _ext("c2", [e2])]

    resolver = EntityResolver(similarity_threshold=0.99)
    updated, merge_map = resolver.resolve(extractions)

    assert merge_map == {}


def test_merge_map_keys_are_merged_ids():
    """merge_map keys should be the IDs of entities that were merged away."""
    e1 = _tool("Postgres", "c1")
    e2 = _tool("PostgreSQL", "c2")
    extractions = [_ext("c1", [e1]), _ext("c2", [e2])]

    resolver = EntityResolver(similarity_threshold=0.70)
    updated, merge_map = resolver.resolve(extractions)

    if merge_map:
        for merged_id, canonical_id in merge_map.items():
            assert merged_id != canonical_id
            # Canonical should still exist in the updated results
            canonical_ids = {e.id for ex in updated for e in ex.entities}
            assert canonical_id in canonical_ids


def test_canonical_appears_only_once():
    """After resolution each canonical entity should appear exactly once."""
    e1 = _tool("Python", "c1")
    e2 = _tool("Python", "c2")
    e3 = _tool("python", "c3")  # lowercase variant
    extractions = [_ext("c1", [e1]), _ext("c2", [e2]), _ext("c3", [e3])]

    resolver = EntityResolver(similarity_threshold=0.85)
    updated, merge_map = resolver.resolve(extractions)

    all_entities = [e for ex in updated for e in ex.entities]
    python_ids = [e.id for e in all_entities if "python" in e.name.lower()]
    assert len(set(python_ids)) == 1


def test_existing_entities_become_canonicals():
    """Existing vault entities should take precedence as canonicals."""
    vault_entity = _tool("Python", "vault")
    new_entity = _tool("Python", "new_conv")
    extractions = [_ext("new_conv", [new_entity])]

    resolver = EntityResolver(similarity_threshold=0.85)
    updated, merge_map = resolver.resolve(extractions, existing_entities=[vault_entity])

    # new_entity should have been merged into vault_entity
    assert new_entity.id in merge_map
    assert merge_map[new_entity.id] == vault_entity.id


def test_remap_relationships_updates_ids():
    """remap_relationships should update stale IDs and drop self-loops."""
    from src.models.relationship import Relationship, RelationshipType

    merged_id = "aaa"
    canonical_id = "bbb"

    rel = Relationship(
        source_id=merged_id,
        target_id="ccc",
        relationship_type=RelationshipType.RELATES_TO,
    )
    self_loop = Relationship(
        source_id=merged_id,
        target_id=merged_id,
        relationship_type=RelationshipType.RELATES_TO,
    )

    resolver = EntityResolver()
    result = resolver.remap_relationships(
        [rel, self_loop], {merged_id: canonical_id}
    )

    assert len(result) == 1
    assert result[0].source_id == canonical_id
    # Self-loop (merged_id → merged_id → canonical → canonical) should be dropped
    assert result[0].source_id != result[0].target_id
