"""Parser for Claude chat export format.

Claude exports (from claude.ai account data export) produce a JSON file
with the following structure:

    [
        {
            "uuid": "...",
            "name": "Conversation Title",
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "updated_at": "2024-01-15T11:00:00.000000+00:00",
            "chat_messages": [
                {
                    "uuid": "...",
                    "text": "message content",
                    "sender": "human" | "assistant",
                    "created_at": "2024-01-15T10:30:00.000000+00:00",
                    "attachments": [...],
                    "files": [...]
                },
                ...
            ]
        },
        ...
    ]
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

# Claude uses "human" / "assistant" for sender field
_ROLE_MAP = {
    "human": Role.USER,
    "assistant": Role.ASSISTANT,
    "system": Role.SYSTEM,
}


class ClaudeIngester(BaseIngester):
    """Parse Claude chat export JSON files."""

    platform_name = "claude"

    def parse_export(
        self, data: Union[dict, list], source_file: Optional[Path] = None
    ) -> list[Conversation]:
        # Claude export is a top-level list of conversation objects
        if isinstance(data, dict):
            # Sometimes wrapped in a single object
            conversations_data = data.get("conversations", [data])
        elif isinstance(data, list):
            conversations_data = data
        else:
            logger.warning("Unexpected Claude export format")
            return []

        conversations: list[Conversation] = []
        for conv_data in conversations_data:
            conv = self._parse_conversation(conv_data)
            if conv:
                conversations.append(conv)

        return conversations

    def _parse_conversation(self, data: dict) -> Conversation | None:
        conv_id = data.get("uuid", data.get("id", ""))
        if not conv_id:
            return None

        created_at = self._parse_timestamp(data.get("created_at"))
        if not created_at:
            return None

        messages: list[ChatMessage] = []
        for msg_data in data.get("chat_messages", []):
            msg = self._parse_message(msg_data, conv_id)
            if msg:
                messages.append(msg)

        # Sort messages by timestamp if available
        messages.sort(key=lambda m: m.timestamp or datetime.min)

        return Conversation(
            id=conv_id,
            title=data.get("name"),
            messages=messages,
            platform=Platform.CLAUDE,
            author="chat-claude",
            session_id=conv_id,
            created_at_iso=created_at.isoformat(),
            model_id=next((m.model for m in messages if m.model), None),
            ingested_by="pipeline-claude-export",
            created_at=created_at,
            updated_at=self._parse_timestamp(data.get("updated_at")),
        )

    def _parse_message(self, data: dict, conversation_id: str) -> ChatMessage | None:
        sender = data.get("sender", "")
        role = _ROLE_MAP.get(sender)
        if role is None:
            return None

        # Claude messages can have text directly or in content blocks
        content = data.get("text", "")
        if not content and "content" in data:
            content_blocks = data["content"]
            if isinstance(content_blocks, list):
                content = "\n".join(
                    block.get("text", "")
                    for block in content_blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            elif isinstance(content_blocks, str):
                content = content_blocks

        if not content.strip():
            return None

        return ChatMessage(
            id=data.get("uuid", data.get("id", "")),
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=self._parse_timestamp(data.get("created_at")),
            platform=Platform.CLAUDE,
            model=data.get("model"),
        )

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            # Python < 3.11 doesn't support 'Z' suffix in fromisoformat
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
