"""Parser for Google Gemini chat export format.

Gemini exports (via Google Takeout) produce per-conversation JSON files
under Gemini Apps/. Each file follows this structure:

    {
        "id": "...",
        "name": "Conversation Title",
        "create_time": "2024-01-15T10:30:00Z",
        "update_time": "2024-01-15T11:00:00Z",
        "entries": [
            {
                "id": "...",
                "create_time": "2024-01-15T10:30:00Z",
                "parts": [
                    { "text": "message content" }
                ],
                "role": "User" | "Model",
                ...
            },
            ...
        ]
    }

Google Takeout can also produce a bulk JSON list across conversations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from src.models.base import Platform
from src.models.message import ChatMessage, Conversation, Role
from src.ingest.base import BaseIngester

logger = logging.getLogger(__name__)

_ROLE_MAP = {
    "User": Role.USER,
    "user": Role.USER,
    "Model": Role.ASSISTANT,
    "model": Role.ASSISTANT,
}


class GeminiIngester(BaseIngester):
    """Parse Google Gemini / Google Takeout export JSON files."""

    platform_name = "gemini"

    def parse_export(
        self, data: Union[dict, list], source_file: Optional[Path] = None
    ) -> list[Conversation]:
        # Handle both single conversation and list-of-conversations formats
        if isinstance(data, list):
            conversations_data = data
        elif isinstance(data, dict):
            if "entries" in data:
                # Single conversation file
                conversations_data = [data]
            else:
                conversations_data = data.get("conversations", [data])
        else:
            logger.warning("Unexpected Gemini export format")
            return []

        conversations: list[Conversation] = []
        for conv_data in conversations_data:
            conv = self._parse_conversation(conv_data)
            if conv:
                conversations.append(conv)

        return conversations

    def _parse_conversation(self, data: dict) -> Conversation | None:
        conv_id = data.get("id", "")
        if not conv_id:
            return None

        created_at = self._parse_timestamp(data.get("create_time"))
        if not created_at:
            return None

        messages: list[ChatMessage] = []
        for entry in data.get("entries", []):
            msg = self._parse_entry(entry, conv_id)
            if msg:
                messages.append(msg)

        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp or datetime.min)

        return Conversation(
            id=conv_id,
            title=data.get("name"),
            messages=messages,
            platform=Platform.GEMINI,
            author="chat-gemini",
            session_id=conv_id,
            created_at_iso=created_at.isoformat(),
            model_id=self._conversation_model_id(data, messages),
            ingested_by="pipeline-gemini-ingester",
            created_at=created_at,
            updated_at=self._parse_timestamp(data.get("update_time")),
        )

    def _parse_entry(self, entry: dict, conversation_id: str) -> ChatMessage | None:
        role_str = entry.get("role", "")
        role = _ROLE_MAP.get(role_str)
        if role is None:
            return None

        # Gemini entries use a "parts" array with text fields
        parts = entry.get("parts", [])
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)

        content = "\n".join(text_parts)
        if not content.strip():
            return None

        return ChatMessage(
            id=entry.get("id", ""),
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=self._parse_timestamp(entry.get("create_time")),
            platform=Platform.GEMINI,
            model=self._entry_model_id(entry),
        )

    @staticmethod
    def _entry_model_id(entry: dict) -> str | None:
        metadata = entry.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("model_slug")
        return None

    @staticmethod
    def _conversation_model_id(data: dict, messages: list[ChatMessage]) -> str | None:
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("model_slug"):
            return metadata["model_slug"]
        return next((msg.model for msg in messages if msg.model), None)

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
