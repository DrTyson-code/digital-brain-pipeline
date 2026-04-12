"""Surface items needing human review and handle user corrections.

Generates a prioritised review queue from four sources:
1. Low-confidence concepts (below min_review_confidence threshold)
2. Entities that had duplicates merged into them (verify correctness)
3. Detected contradictions (both sides flagged)
4. Concepts whose status changed to superseded / completed / abandoned

Feedback mechanism:
- Look for a _corrections/ folder in the vault.
- Each file is a YAML document (or list of documents) with:
      original_id:      <pipeline object id>
      correction_type:  rename | merge | split | delete | status_change
      corrected_value:  <new value>
- Corrections are loaded and returned; they serve as context for the next run.

This module also defines CurationResult — the single data bag that flows
from the curation stages into the output writers (obsidian.py, moc.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from src.models.entity import Entity
from src.models.concept import Concept
from src.models.relationship import Relationship, RelationshipType
from src.process.temporal_tracker import ConceptStatus

logger = logging.getLogger(__name__)

_CORRECTIONS_FOLDER = "_corrections"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ReviewItem:
    """A single item flagged for human review."""

    object_id: str
    object_type: str  # "entity" | "concept" | "contradiction"
    reason: str
    priority: str = "normal"  # "high" | "normal" | "low"


@dataclass
class CurationResult:
    """Aggregated output from all curation stages.

    Passed from the pipeline into output writers (ObsidianWriter, MOCGenerator)
    so they can enrich notes with curation metadata.
    """

    # Stage 4a — source quality weights (conv_id → [0, 1])
    source_weights: Dict[str, float] = field(default_factory=dict)

    # Stage 4b — entity resolution (merged_id → canonical_id)
    merge_map: Dict[str, str] = field(default_factory=dict)

    # Stage 4c — temporal status (concept_id → ConceptStatus)
    concept_statuses: Dict[str, ConceptStatus] = field(default_factory=dict)

    # Stage 4d — contradiction relationships
    contradictions: List[Relationship] = field(default_factory=list)

    # Stage 4e — review queue
    review_items: List[ReviewItem] = field(default_factory=list)
    review_ids: Set[str] = field(default_factory=set)

    # Corrections loaded from vault _corrections/ folder
    corrections: List[Dict[str, Any]] = field(default_factory=list)

    # Stage 8 — cross-domain synthesis notes
    synthesis_notes: List[Any] = field(default_factory=list)  # List[SynthesisNote]


# ---------------------------------------------------------------------------
# Review queue generator
# ---------------------------------------------------------------------------


@dataclass
class ReviewQueueGenerator:
    """Generate a prioritised review queue from curation outputs."""

    min_review_confidence: float = 0.6

    def generate(
        self,
        entities: List[Entity],
        concepts: List[Concept],
        contradictions: List[Relationship],
        merge_map: Optional[Dict[str, str]] = None,
        concept_statuses: Optional[Dict[str, ConceptStatus]] = None,
    ) -> tuple[List[ReviewItem], Set[str]]:
        """Build the review queue.

        Returns:
            (review_items, review_ids) — review_ids is the set of object IDs
            that should have ``needs_review: true`` in their Obsidian frontmatter.
        """
        items: List[ReviewItem] = []
        review_ids: Set[str] = set()

        # 1. Low-confidence concepts
        for concept in concepts:
            if concept.confidence < self.min_review_confidence:
                priority = "low" if concept.confidence < 0.4 else "normal"
                items.append(
                    ReviewItem(
                        object_id=concept.id,
                        object_type="concept",
                        reason=f"Low confidence: {concept.confidence:.2f}",
                        priority=priority,
                    )
                )
                review_ids.add(concept.id)

        # 2. Canonical entities that absorbed merges
        if merge_map:
            canonical_merge_counts: Dict[str, int] = {}
            for canonical_id in merge_map.values():
                canonical_merge_counts[canonical_id] = (
                    canonical_merge_counts.get(canonical_id, 0) + 1
                )

            entity_ids = {e.id for e in entities}
            for canonical_id, count in canonical_merge_counts.items():
                if canonical_id in entity_ids and canonical_id not in review_ids:
                    items.append(
                        ReviewItem(
                            object_id=canonical_id,
                            object_type="entity",
                            reason=f"Entity merge target: {count} duplicate(s) merged",
                            priority="normal",
                        )
                    )
                    review_ids.add(canonical_id)

        # 3. Contradiction pairs — both sides flagged
        for rel in contradictions:
            if rel.relationship_type != RelationshipType.CONTRADICTS:
                continue
            for oid in (rel.source_id, rel.target_id):
                if oid not in review_ids:
                    other = rel.target_id if oid == rel.source_id else rel.source_id
                    items.append(
                        ReviewItem(
                            object_id=oid,
                            object_type="contradiction",
                            reason=f"Contradicts {other}",
                            priority="high",
                        )
                    )
                    review_ids.add(oid)

        # 4. Concepts with status changes
        if concept_statuses:
            for concept_id, status in concept_statuses.items():
                if (
                    status.status in ("superseded", "completed", "abandoned")
                    and concept_id not in review_ids
                ):
                    items.append(
                        ReviewItem(
                            object_id=concept_id,
                            object_type="concept",
                            reason=f"Status changed to: {status.status}",
                            priority="normal",
                        )
                    )
                    review_ids.add(concept_id)

        # 5. Stale knowledge — high-confidence items not re-referenced
        if concept_statuses:
            for concept_id, status in concept_statuses.items():
                if (
                    status.status == "stale"
                    and concept_id not in review_ids
                ):
                    items.append(
                        ReviewItem(
                            object_id=concept_id,
                            object_type="concept",
                            reason=f"Stale knowledge (since {status.stale_since})",
                            priority="low",
                        )
                    )
                    review_ids.add(concept_id)

        logger.info(
            "Review queue: %d items flagged (%d unique object IDs)",
            len(items),
            len(review_ids),
        )
        return items, review_ids

    def load_corrections(self, vault_path: Path) -> List[Dict[str, Any]]:
        """Load user corrections from <vault>/_corrections/*.yaml.

        Each correction file contains one YAML document (or a YAML list) with:
            original_id:      <pipeline object id>
            correction_type:  rename | merge | split | delete | status_change
            corrected_value:  <new value or target id>

        Returns:
            List of correction dicts. Empty list if folder doesn't exist.
        """
        corrections_dir = Path(vault_path).expanduser() / _CORRECTIONS_FOLDER
        if not corrections_dir.exists():
            return []

        corrections: List[Dict[str, Any]] = []
        for yaml_file in sorted(corrections_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    corrections.append(raw)
                elif isinstance(raw, list):
                    corrections.extend(item for item in raw if isinstance(item, dict))
            except Exception as exc:
                logger.warning(
                    "Skipping correction file %s: %s", yaml_file.name, exc
                )

        logger.info(
            "Loaded %d correction(s) from %s",
            len(corrections),
            corrections_dir,
        )
        return corrections
