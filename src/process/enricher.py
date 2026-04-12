"""Enrich and deduplicate extracted objects.

Post-processing stage that:
- Merges duplicate entities (same name / aliases)
- Resolves entity references across conversations
- Adds computed metadata (frequency, first/last seen)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.models.entity import Entity
from src.models.relationship import Relationship
from src.process.extractor import ExtractionResult

logger = logging.getLogger(__name__)


class Enricher:
    """Deduplicate entities and enrich metadata across the full corpus."""

    def __init__(self, deduplicate: bool = True) -> None:
        self.deduplicate = deduplicate

    def enrich(
        self,
        extractions: list[ExtractionResult],
        relationships: list[Relationship],
    ) -> tuple[list[ExtractionResult], list[Relationship]]:
        """Run enrichment and deduplication on the full extraction set.

        Returns updated extractions and relationships with merged entity IDs.
        """
        if self.deduplicate:
            id_map = self._deduplicate_entities(extractions)
            relationships = self._remap_relationships(relationships, id_map)

        return extractions, relationships

    def _deduplicate_entities(
        self, extractions: list[ExtractionResult]
    ) -> dict[str, str]:
        """Merge entities with matching names across all extractions.

        Returns a mapping of old_id → canonical_id for merged entities.
        """
        # Index all entities by lowercase name
        name_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for ext_idx, extraction in enumerate(extractions):
            for ent_idx, entity in enumerate(extraction.entities):
                for name in entity.all_names:
                    name_index[name.lower()].append((ext_idx, ent_idx))

        id_map: dict[str, str] = {}
        merged: set[str] = set()

        for name, locations in name_index.items():
            if len(locations) < 2:
                continue

            # Use the first occurrence as canonical
            canon_ext, canon_idx = locations[0]
            canonical = extractions[canon_ext].entities[canon_idx]

            if canonical.id in merged:
                continue

            for ext_idx, ent_idx in locations[1:]:
                other = extractions[ext_idx].entities[ent_idx]
                if other.id == canonical.id or other.id in merged:
                    continue
                if other.entity_type != canonical.entity_type:
                    continue

                # Merge other into canonical
                canonical.merge(other)
                id_map[other.id] = canonical.id
                merged.add(other.id)

        # Remove merged entities from extractions
        for extraction in extractions:
            extraction.entities = [
                e for e in extraction.entities if e.id not in merged
            ]

        logger.info(
            "Deduplicated %d entities (merged into %d canonical)",
            len(merged),
            len(merged) - len(id_map) + len(set(id_map.values())),
        )
        return id_map

    def _remap_relationships(
        self,
        relationships: list[Relationship],
        id_map: dict[str, str],
    ) -> list[Relationship]:
        """Update relationship source/target IDs after deduplication."""
        for rel in relationships:
            if rel.source_id in id_map:
                rel.source_id = id_map[rel.source_id]
            if rel.target_id in id_map:
                rel.target_id = id_map[rel.target_id]

        # Remove self-referential relationships created by merging
        relationships = [r for r in relationships if r.source_id != r.target_id]
        return relationships
