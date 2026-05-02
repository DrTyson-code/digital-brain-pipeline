"""Detect conflicting knowledge across extracted concepts.

Approach:
1. Restrict to actionable concept types (DECISION, INSIGHT) where contradictions
   are meaningful enough to surface.
2. Group by concept_type so we only compare decisions against decisions, etc.
3. Skip pairs from the same conversation (minor rephrasing is expected).
4. For pairs above a configurable similarity threshold, check for negation
   asymmetry — one text asserts something the other explicitly negates.
5. Create Relationship objects with RelationshipType.CONTRADICTS (already in
   the model) and return them for appending to the main relationships list.

The negation check uses specific positive/negative pattern pairs to reduce
false positives compared to a simple "one has NOT and the other doesn't".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from src.models.concept import Concept, ConceptType
from src.models.relationship import Relationship, RelationshipType

logger = logging.getLogger(__name__)


# Each tuple: (positive_pattern, negative_pattern).
# A contradiction is signalled when one text matches the positive form and the
# other matches the negative form for the same pattern pair.
_NEGATION_PAIRS: List[Tuple[re.Pattern[str], re.Pattern[str]]] = [
    (
        re.compile(r"\bshould\b", re.IGNORECASE),
        re.compile(r"\bshould\s+not\b|\bshouldn't\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\buse\b|\busing\b|\bused\b", re.IGNORECASE),
        re.compile(r"\bdon't\s+use\b|\bdo\s+not\s+use\b|\bavoid\b|\bnot\s+use\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\bwill\b|\bgoing\s+to\b", re.IGNORECASE),
        re.compile(r"\bwon't\b|\bwill\s+not\b|\bnot\s+going\s+to\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\bdo\b|\bdoes\b", re.IGNORECASE),
        re.compile(r"\bdon't\b|\bdoesn't\b|\bdo\s+not\b|\bdoes\s+not\b", re.IGNORECASE),
    ),
]


# Concept types eligible for contradiction checks
_CHECKED_TYPES = (ConceptType.DECISION, ConceptType.INSIGHT)


@dataclass
class ContradictionDetector:
    """Detect contradictory concepts and surface them as CONTRADICTS relationships."""

    similarity_threshold: float = 0.50

    def detect(self, concepts: List[Concept]) -> List[Relationship]:
        """Find contradictions and return them as CONTRADICTS Relationship objects.

        Args:
            concepts: All extracted concepts from the pipeline.

        Returns:
            List of Relationship objects with type CONTRADICTS.
            These should be appended to the main relationships list so the
            graph and output stages can reference them.
        """
        candidates = [c for c in concepts if c.concept_type in _CHECKED_TYPES]

        # Group by concept_type to avoid comparing apples to oranges
        by_type: Dict[str, List[Concept]] = {}
        for concept in candidates:
            by_type.setdefault(concept.concept_type.value, []).append(concept)

        # Track per-type pair counts for observability (denominator for hit rate)
        pairs_compared_by_type: Dict[str, int] = {t: 0 for t in by_type}
        contradictions: List[Relationship] = []
        # Keep a small sample of contradiction examples for the summary log
        examples: List[Tuple[str, str]] = []

        for type_name, type_concepts in by_type.items():
            for i, a in enumerate(type_concepts):
                for b in type_concepts[i + 1:]:
                    # Skip same-conversation pairs (rephrasing is normal)
                    if (
                        a.source_conversation_id
                        and a.source_conversation_id == b.source_conversation_id
                    ):
                        continue
                    pairs_compared_by_type[type_name] = (
                        pairs_compared_by_type.get(type_name, 0) + 1
                    )
                    similarity = SequenceMatcher(
                        None,
                        a.content.lower(),
                        b.content.lower(),
                    ).ratio()
                    if similarity < self.similarity_threshold:
                        continue
                    if self._has_contradiction(a.content, b.content):
                        rel = Relationship(
                            source_id=a.id,
                            target_id=b.id,
                            relationship_type=RelationshipType.CONTRADICTS,
                            weight=round(similarity, 3),
                            evidence=[
                                f"'{a.content[:80]}' vs '{b.content[:80]}'"
                            ],
                        )
                        contradictions.append(rel)
                        if len(examples) < 3:
                            examples.append(
                                (a.content[:80], b.content[:80])
                            )
                        logger.debug(
                            "Contradiction: %s ↔ %s (similarity=%.2f)",
                            a.id,
                            b.id,
                            similarity,
                        )

        total_pairs = sum(pairs_compared_by_type.values())
        if contradictions:
            hit_rate = (
                len(contradictions) / total_pairs * 100.0 if total_pairs else 0.0
            )
            logger.info(
                "Contradiction detection: %d candidates, %d cross-conversation "
                "pairs compared, %d contradictions found (%.3f%% hit rate)",
                len(candidates),
                total_pairs,
                len(contradictions),
                hit_rate,
            )
            for ex_a, ex_b in examples:
                logger.info("Contradiction example: %r ↔ %r", ex_a, ex_b)
        else:
            # Make absence-of-signal explicit so silent zero is distinguishable
            # from "stage didn't run." A consistent zero across runs is itself
            # a calibration data point — possibly threshold too restrictive,
            # or possibly working correctly on a corpus without conflicts.
            logger.info(
                "Contradiction detection: %d candidates, %d cross-conversation "
                "pairs compared, 0 contradictions found "
                "(threshold=%.2f; persistent zero may indicate restrictive threshold)",
                len(candidates),
                total_pairs,
                self.similarity_threshold,
            )

        return contradictions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_contradiction(self, text_a: str, text_b: str) -> bool:
        """Return True if one text affirms while the other negates via a known pattern.

        A contradiction is signalled when:
        - The two texts have opposite polarity for a pattern pair (one negates,
          the other does not).
        - At least one text contains the positive form of the keyword.
          We don't require *both* to contain it because negation words like
          "avoid" or "won't" implicitly reference the positive form.
        """
        for positive_pat, negative_pat in _NEGATION_PAIRS:
            a_neg = bool(negative_pat.search(text_a))
            b_neg = bool(negative_pat.search(text_b))
            if a_neg == b_neg:
                continue  # same polarity — not a contradiction for this pair

            # Require at least one text to contain the positive keyword so we
            # avoid flagging unrelated sentences that happen to share "not".
            a_pos = bool(positive_pat.search(text_a))
            b_pos = bool(positive_pat.search(text_b))
            if a_pos or b_pos:
                return True

        return False
