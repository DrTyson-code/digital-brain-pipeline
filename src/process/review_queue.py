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


def _categorize_reason(reason: str) -> str:
    """Map a free-form reason string to a coarse category for breakdown logging."""
    if reason.startswith("Low confidence"):
        return "low_confidence"
    if reason.startswith("Entity merge target"):
        return "entity_merge_review"
    if reason.startswith("Contradicts"):
        return "contradiction"
    if reason.startswith("Status changed"):
        return "status_change"
    if reason.startswith("Stale knowledge"):
        return "stale"
    return "other"


def _confidence_bin_label(idx: int) -> str:
    """5-bin equal-width labels for confidence in [0.0, 1.0]."""
    return ["[0.0-0.2]", "[0.2-0.4]", "[0.4-0.6]", "[0.6-0.8]", "[0.8-1.0]"][idx]


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

        # Breakdown by priority, object_type, and reason category
        if items:
            priority_counts: Dict[str, int] = {}
            type_counts: Dict[str, int] = {}
            reason_categories: Dict[str, int] = {}
            for item in items:
                priority_counts[item.priority] = priority_counts.get(item.priority, 0) + 1
                type_counts[item.object_type] = type_counts.get(item.object_type, 0) + 1
                cat = _categorize_reason(item.reason)
                reason_categories[cat] = reason_categories.get(cat, 0) + 1

            priority_str = ", ".join(
                f"{p}={c}" for p, c in sorted(priority_counts.items(), key=lambda kv: -kv[1])
            )
            type_str = ", ".join(
                f"{t}={c}" for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1])
            )
            reason_str = ", ".join(
                f"{r}={c}" for r, c in sorted(reason_categories.items(), key=lambda kv: -kv[1])
            )
            logger.info("Review queue by priority: %s", priority_str)
            logger.info("Review queue by object_type: %s", type_str)
            logger.info("Review queue by reason category: %s", reason_str)

            # Confidence histogram of concept items.
            # This is the calibration signal for min_review_confidence: it
            # shows how the queue would shrink if the threshold were bumped.
            concept_conf_by_id = {c.id: c.confidence for c in concepts}
            confidence_bins = [0] * 5
            concept_items_with_conf = 0
            for item in items:
                if item.object_type != "concept":
                    continue
                conf = concept_conf_by_id.get(item.object_id)
                if conf is None:
                    # Status-change or stale items may reference concepts not
                    # in the input list (e.g., from prior runs); skip silently.
                    continue
                bin_idx = min(4, max(0, int(conf * 5)))
                confidence_bins[bin_idx] += 1
                concept_items_with_conf += 1

            if concept_items_with_conf > 0:
                bins_str = ", ".join(
                    f"{_confidence_bin_label(i)}={confidence_bins[i]}"
                    for i in range(5)
                )
                logger.info(
                    "Review queue confidence histogram (%d concept items, "
                    "current threshold=%.2f): %s",
                    concept_items_with_conf,
                    self.min_review_confidence,
                    bins_str,
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
