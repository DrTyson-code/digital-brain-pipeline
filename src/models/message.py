"""Chat message and conversation models."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import Field

from src.models.base import OntologyObject, Platform, StrEnum


class Role(StrEnum):
    """Message author role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(OntologyObject):
    """A single message within a conversation."""

    conversation_id: str
    role: Role
    content: str
    timestamp: Optional[datetime] = None
    platform: Platform
    model: Optional[str] = None
    tokens: Optional[int] = None

    @property
    def is_user(self) -> bool:
        return self.role == Role.USER

    @property
    def is_assistant(self) -> bool:
        return self.role == Role.ASSISTANT

    @property
    def word_count(self) -> int:
        return len(self.content.split())


class Conversation(OntologyObject):
    """A complete conversation session with an AI assistant."""

    title: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    platform: Platform
    author: Optional[str] = None
    session_id: Optional[str] = None
    session_slug: Optional[str] = None
    created_at_iso: Optional[str] = None
    model_id: Optional[str] = None
    ingested_by: Optional[str] = None
    source: Optional[str] = None
    updated_at: Optional[datetime] = None
    topics: List[str] = Field(default_factory=list)
    summary: Optional[str] = None

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.is_user]

    @property
    def assistant_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.is_assistant]

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages if m.tokens)

    @property
    def duration(self) -> float | None:
        """Conversation duration in seconds, if timestamps are available."""
        timestamps = [m.timestamp for m in self.messages if m.timestamp]
        if len(timestamps) < 2:
            return None
        return (max(timestamps) - min(timestamps)).total_seconds()
