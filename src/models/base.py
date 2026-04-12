"""Base ontology types shared across all models."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11."""
        pass

from typing import Optional

from pydantic import BaseModel, Field


class Platform(StrEnum):
    """Supported AI chat platforms."""

    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    CALENDAR = "calendar"
    COWORK = "cowork"


class OntologyObject(BaseModel):
    """Base class for all objects in the knowledge graph ontology.

    Every object has a unique ID, creation timestamp, and a source platform.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=datetime.now)
    source_platform: Optional[Platform] = None

    def merge(self, other: OntologyObject) -> None:
        """Merge another object's data into this one (in-place).

        Subclasses should override to handle type-specific merge logic.
        """
        if other.source_platform and not self.source_platform:
            self.source_platform = other.source_platform
