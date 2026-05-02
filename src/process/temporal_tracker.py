"""Lifecycle tracking for decisions and action items.

Adds status metadata to concepts so the vault reflects whether a decision
is still active, has been superseded by a newer one, or is completed.

Detection approach (rule-based):
- Group decisions and action items by type.
- Sort by date (conversation created_at or concept created_at as fallback).
- For each pair where the newer concept is topically similar (difflib >= threshold)
  AND contains supersession keywords ("switched to", "no longer", etc.),
  mark the older concept as superseded.

Placeholder hook is included for an LLM-enhanced detection pass.

New frontmatter fields produced (stored in ConceptStatus):
    status:         active | superseded | completed | abandoned
    valid_from:     date the concept was first seen
    superseded_by:  short title of the superseding concept (for wikilink)
    last_confirmed: last date this concept was referenced without contradiction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from src.models.concept import Concept, ConceptType
from src.models.message import Conversation

logger = logging.getLogger(__name__)


# Keywords whose presence in a newer concept suggests it supersedes an older one
SUPERSESSION_KEYWORDS = [
    "changed",
    "reversed",
    "decided instead",
    "no longer",
    "switched to",
    "instead of",
    "replaced",
    "updated decision",
    "actually",
    "nevermind",
    "going with",
    "pivot",
]


# Minimum content similarity for supersession comparison
_DEFAULT_SIMILARITY_THRESHOLD = 0.40


# Concept types to track
_TRACKED_TYPES = (ConceptType.DECISION, ConceptType.ACTION_ITEM)


@dataclass
class ConceptStatus:
    """Lifecycle metadata for a single concept."""

    concept_id: str
    status: str = "active"  # active | superseded | completed | abandoned | stale
    valid_from: Optional[datetime] = None
    superseded_by_id: Optional[str] = None
    superseded_by_title: Optional[str] = None  # truncated content for wikilink
    last_confirmed: Optional[datetime] = None
    stale_since: Optional[str] = None  # ISO date when marked stale


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Return *dt* normalized to tz-aware UTC; assume naive datetimes are UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class TemporalTracker:
    """Assign lifecycle status to decisions and action items."""

    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD

    def track(
        self,
        concepts: List[Concept],
        conversations: Optional[List[Conversation]] = None,
    ) -> Dict[str, ConceptStatus]:
        """Analyse concepts and return a ConceptStatus per concept.

        Args:
            concepts: All extracted concepts.
            conversations: Optional list; used to resolve creation dates by
                           conversation_id for accurate chronological ordering.

        Returns:
            Dict mapping concept_id → ConceptStatus.
        """
        # Build a date lookup: conversation_id → created_at
        conv_dates: Dict[str, Optional[datetime]] = {}
        if conversations:
            for conv in conversations:
                conv_dates[conv.id] = conv.created_at

        def concept_date(c: Concept) -> datetime:
            if c.source_conversation_id:
                d = conv_dates.get(c.source_conversation_id)
                if d is not None:
                    return _ensure_utc(d)
            return _ensure_utc(c.created_at)

        # Initialise statuses for ALL concepts (not just tracked types)
        statuses: Dict[str, ConceptStatus] = {}
        for concept in concepts:
            d = concept_date(concept)
            statuses[concept.id] = ConceptStatus(
                concept_id=concept.id,
                status="active",
                valid_from=d,
                last_confirmed=d,
            )

        # Supersession detection — only for decisions and action items
        tracked = [c for c in concepts if c.concept_type in _TRACKED_TYPES]
        tracked_sorted = sorted(tracked, key=concept_date)

        # Cap look-back to avoid O(n²) blow-up on large corpora.
        # Only compare each concept against the most-recent _MAX_LOOKBACK predecessors
        # of the same type that still have supersession signal potential.
        _MAX_LOOKBACK = 50

        for i, newer in enumerate(tracked_sorted):
            # Fast pre-filter: skip if newer has no supersession signal at all
            if not self._has_supersession_signal(newer.content):
                continue
            newer_lower = newer.content.lower()
            # Only look back at the last _MAX_LOOKBACK items of the same type
            window = [
                c for c in tracked_sorted[max(0, i - _MAX_LOOKBACK) : i]
                if c.concept_type == newer.concept_type
                and statuses[c.id].status != "superseded"
            ]
            for older in window:
                similarity = SequenceMatcher(
                    None,
                    older.content.lower(),
                    newer_lower,
                ).ratio()
                if similarity < self.similarity_threshold:
                    continue
                statuses[older.id].status = "superseded"
                statuses[older.id].superseded_by_id = newer.id
                statuses[older.id].superseded_by_title = newer.content[:80]
                logger.debug(
                    "Concept %r superseded by %r (similarity=%.2f)",
                    older.id,
                    newer.id,
                    similarity,
                )

        # Status distribution across all concepts (active/superseded/etc.)
        status_counts: Dict[str, int] = {}
        for s in statuses.values():
            status_counts[s.status] = status_counts.get(s.status, 0) + 1
        # Sort by count desc for legible logging
        sorted_status = sorted(status_counts.items(), key=lambda kv: -kv[1])
        breakdown = ", ".join(f"{name}={count}" for name, count in sorted_status)

        logger.info(
            "Temporal tracking: %d concepts in tracked types (DECISION/ACTION_ITEM), "
            "%d total concepts; status distribution: %s",
            len(tracked),
            len(statuses),
            breakdown,
        )

        return statuses

    # ------------------------------------------------------------------
    # Knowledge decay detection
    # ------------------------------------------------------------------

    def detect_stale_knowledge(
        self,
        concepts: List[Concept],
        statuses: Dict[str, ConceptStatus],
        stale_days: int = 90,
        min_confidence: float = 0.7,
    ) -> List[str]:
        """Flag high-confidence decisions/action items that haven't been
        re-referenced or superseded within *stale_days*.

        Adds ``stale_since`` (ISO date string) to the ConceptStatus of flagged
        items and sets their status to ``stale``.

        Args:
            concepts: All extracted concepts.
            statuses: Dict of concept_id → ConceptStatus (mutated in place).
            stale_days: Threshold in days since last_confirmed.
            min_confidence: Only flag items at or above this confidence.

        Returns:
            List of concept IDs that were newly marked stale.
        """
        now = datetime.now(timezone.utc)
        stale_ids: List[str] = []

        for concept in concepts:
            if concept.concept_type not in _TRACKED_TYPES:
                continue
            if concept.confidence < min_confidence:
                continue

            st = statuses.get(concept.id)
            if st is None or st.status in ("superseded", "completed", "abandoned"):
                continue

            ref_date = _ensure_utc(st.last_confirmed or st.valid_from)
            if ref_date is None:
                continue

            age_days = (now - ref_date).days
            if age_days >= stale_days:
                st.status = "stale"
                st.stale_since = now.isoformat()[:10]
                stale_ids.append(concept.id)
                logger.debug(
                    "Concept %r marked stale (age=%d days, confidence=%.2f)",
                    concept.id,
                    age_days,
                    concept.confidence,
                )

        logger.info(
            "Knowledge decay: %d items flagged as stale (>%d days, confidence>=%.2f)",
            len(stale_ids),
            stale_days,
            min_confidence,
        )
        return stale_ids

    # ------------------------------------------------------------------
    # Placeholder for LLM-enhanced detection
    # ------------------------------------------------------------------

    def track_with_llm(
        self,
        concepts: List[Concept],
        conversations: Optional[List[Conversation]] = None,
    ) -> Dict[str, ConceptStatus]:
        """LLM-enhanced supersession detection (not yet implemented).

        Falls back to rule-based detection until an LLM provider is wired in.
        Override this method to add LLM-powered semantic comparison.
        """
        logger.debug("LLM-enhanced temporal tracking not implemented; using rules.")
        return self.track(concepts, conversations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_supersession_signal(self, content: str) -> bool:
        """Return True if the content contains supersession keyword signals."""
        content_lower = content.lower()
        return any(kw in content_lower for kw in SUPERSESSION_KEYWORDS)
