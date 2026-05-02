"""Corpus-level entity deduplication using fuzzy name matching.

Complements the rule-based exact-match deduplication in Enricher by catching
variants like "John Smith" / "J. Smith", "PostgreSQL" / "Postgres", etc.

Approach:
1. Block by entity type — only compare PERSON to PERSON, TOOL to TOOL, etc.
2. Use difflib.SequenceMatcher for similarity across all name + alias pairs.
3. When similarity >= threshold, merge the duplicate into the canonical entity
   (canonical = the first one encountered).
4. Return updated extractions plus a merge_map for wikilink redirection.
5. Optionally incorporate entities already in the vault (existing_entities)
   so new pipeline runs don't create duplicates of known entities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from src.models.entity import Entity
from src.models.relationship import Relationship
from src.process.extractor import ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class EntityResolver:
    """Fuzzy-match entity deduplication across the full extraction corpus."""

    similarity_threshold: float = 0.85

    def resolve(
        self,
        extractions: List[ExtractionResult],
        existing_entities: Optional[List[Entity]] = None,
    ) -> tuple[List[ExtractionResult], Dict[str, str]]:
        """Deduplicate entities across the corpus via fuzzy name matching.

        Args:
            extractions: List of ExtractionResult objects from all conversations.
            existing_entities: Optional entities already in the vault to check
                               new entities against (for incremental runs).

        Returns:
            (updated_extractions, merge_map) where merge_map maps
            merged_entity_id → canonical_entity_id.

            Each canonical entity appears at most once across all results.
        """
        # Collect all entities: existing (vault) entities first so they become
        # the canonicals when a fuzzy match is found.
        all_entities: List[Entity] = list(existing_entities or [])
        for ex in extractions:
            all_entities.extend(ex.entities)

        # Resolve per-type (blocking step)
        merge_map: Dict[str, str] = {}
        canonical_list: List[Entity] = []
        by_type: Dict[str, List[Entity]] = {}
        for entity in all_entities:
            by_type.setdefault(entity.entity_type.value, []).append(entity)

        # Track per-type merge counts and example merges for observability
        merges_by_type: Dict[str, int] = {t: 0 for t in by_type}
        example_merges: List[tuple[str, str, str]] = []  # (type, merged_name, canonical_name)

        for type_name, type_entities in by_type.items():
            type_canonicals: List[Entity] = []
            for entity in type_entities:
                if entity.id in merge_map:
                    continue  # already merged into a canonical
                best = self._find_canonical(entity, type_canonicals)
                if best is not None:
                    merge_map[entity.id] = best.id
                    merges_by_type[type_name] = merges_by_type.get(type_name, 0) + 1
                    if len(example_merges) < 5:
                        example_merges.append((type_name, entity.name, best.name))
                    best.merge(entity)
                else:
                    type_canonicals.append(entity)
            canonical_list.extend(type_canonicals)

        merged_count = len(merge_map)
        merge_rate = (merged_count / len(all_entities) * 100.0) if all_entities else 0.0
        logger.info(
            "Entity resolution: %d entities → %d canonical (%d merged, %.1f%% merge rate)",
            len(all_entities),
            len(canonical_list),
            merged_count,
            merge_rate,
        )
        if merges_by_type:
            # Only log types that actually had merges, sorted by count desc
            non_zero = sorted(
                ((t, c) for t, c in merges_by_type.items() if c > 0),
                key=lambda kv: -kv[1],
            )
            if non_zero:
                breakdown = ", ".join(f"{t}={c}" for t, c in non_zero)
                logger.info("Entity merges by type: %s", breakdown)
        for ex_type, merged_name, canonical_name in example_merges:
            logger.info(
                "Entity merge example [%s]: %r → %r",
                ex_type,
                merged_name,
                canonical_name,
            )

        if merged_count == 0:
            return extractions, merge_map

        # Rebuild extractions with canonical entities only
        canonical_by_id: Dict[str, Entity] = {e.id: e for e in canonical_list}
        # Any merged id → its canonical
        for mid, cid in merge_map.items():
            canonical_by_id[mid] = canonical_by_id[cid]

        # Each canonical entity should appear in only one ExtractionResult
        # (the first one that mentions it).  We track which canonical IDs have
        # already been placed to avoid duplicating entities in the flat list.
        seen_canonical_ids: set[str] = set()
        updated: List[ExtractionResult] = []
        for ex in extractions:
            unique_entities: List[Entity] = []
            for entity in ex.entities:
                canonical = canonical_by_id.get(entity.id, entity)
                if canonical.id not in seen_canonical_ids:
                    unique_entities.append(canonical)
                    seen_canonical_ids.add(canonical.id)
            updated.append(
                ExtractionResult(
                    conversation_id=ex.conversation_id,
                    entities=unique_entities,
                    concepts=ex.concepts,
                )
            )

        return updated, merge_map

    def remap_relationships(
        self,
        relationships: List[Relationship],
        merge_map: Dict[str, str],
    ) -> List[Relationship]:
        """Update relationship source/target IDs after entity resolution.

        Mirrors the same step in Enricher so the caller can keep graph
        consistency after a fuzzy-merge pass.
        """
        if not merge_map:
            return relationships

        for rel in relationships:
            if rel.source_id in merge_map:
                rel.source_id = merge_map[rel.source_id]
            if rel.target_id in merge_map:
                rel.target_id = merge_map[rel.target_id]

        # Drop self-loops created by merging
        return [r for r in relationships if r.source_id != r.target_id]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_canonical(
        self, entity: Entity, candidates: List[Entity]
    ) -> Optional[Entity]:
        """Return the best-matching candidate for entity, or None."""
        best_score = 0.0
        best_match: Optional[Entity] = None
        for candidate in candidates:
            score = self._max_name_similarity(entity, candidate)
            if score >= self.similarity_threshold and score > best_score:
                best_score = score
                best_match = candidate
        return best_match

    def _max_name_similarity(self, a: Entity, b: Entity) -> float:
        """Return the maximum SequenceMatcher ratio across all name pairs."""
        best = 0.0
        for name_a in a.all_names:
            for name_b in b.all_names:
                ratio = SequenceMatcher(
                    None, name_a.lower(), name_b.lower()
                ).ratio()
                if ratio > best:
                    best = ratio
        return best
