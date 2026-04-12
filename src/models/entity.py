"""Entity model — named real-world or conceptual things extracted from conversations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field

from src.models.base import OntologyObject, StrEnum


class EntityType(StrEnum):
    """Types of extractable entities."""

    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TOOL = "tool"
    LOCATION = "location"


class Entity(OntologyObject):
    """A named entity extracted from one or more conversations."""

    entity_type: EntityType
    name: str
    aliases: List[str] = Field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_conversations: List[str] = Field(default_factory=list)

    @property
    def all_names(self) -> set[str]:
        """Return the canonical name plus all known aliases."""
        return {self.name} | set(self.aliases)

    def matches(self, name: str) -> bool:
        """Check if a name matches this entity (case-insensitive)."""
        return name.lower() in {n.lower() for n in self.all_names}

    def merge(self, other: Entity) -> None:
        """Merge another entity's data into this one."""
        super().merge(other)
        # Combine aliases (deduplicated)
        combined = self.all_names | other.all_names
        combined.discard(self.name)
        self.aliases = sorted(combined)
        # Update time bounds
        if other.first_seen:
            if not self.first_seen or other.first_seen < self.first_seen:
                self.first_seen = other.first_seen
        if other.last_seen:
            if not self.last_seen or other.last_seen > self.last_seen:
                self.last_seen = other.last_seen
        # Merge properties (other wins on conflicts)
        self.properties = {**self.properties, **other.properties}
        # Merge source conversations
        self.source_conversations = sorted(
            set(self.source_conversations) | set(other.source_conversations)
        )
