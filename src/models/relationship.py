"""Relationship model — links between ontology objects in the knowledge graph."""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from src.models.base import OntologyObject, StrEnum


class RelationshipType(StrEnum):
    """Types of relationships between objects."""

    MENTIONS = "mentions"
    DECIDES = "decides"
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"


class Relationship(OntologyObject):
    """A directed or undirected link between two ontology objects."""

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    source_conversation_id: Optional[str] = None

    @property
    def is_directed(self) -> bool:
        """Some relationship types are inherently directed."""
        return self.relationship_type in (
            RelationshipType.MENTIONS,
            RelationshipType.DECIDES,
            RelationshipType.DEPENDS_ON,
        )

    def involves(self, object_id: str) -> bool:
        """Check if an object is part of this relationship."""
        return object_id in (self.source_id, self.target_id)
