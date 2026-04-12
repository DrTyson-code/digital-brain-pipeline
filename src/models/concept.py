"""Concept model — abstract ideas, decisions, and action items from conversations."""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from src.models.base import OntologyObject, StrEnum


class ConceptType(StrEnum):
    """Types of extractable concepts."""

    TOPIC = "topic"
    DECISION = "decision"
    ACTION_ITEM = "action_item"
    INSIGHT = "insight"
    QUESTION = "question"


class Concept(OntologyObject):
    """A concept, decision, or action item extracted from a conversation."""

    concept_type: ConceptType
    content: str
    context: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_conversation_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.concept_type in (ConceptType.ACTION_ITEM, ConceptType.DECISION)

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8
