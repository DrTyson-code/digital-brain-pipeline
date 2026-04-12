"""Parser for ChatGPT chat export format.

ChatGPT exports (from Settings → Data Controls → Export Data) produce a
conversations.json file with the following structure:

    [
        {
            "title": "Conversation Title",
            "create_time": 1705312200.0,
            "update_time": 1705315800.0,
            "mapping": {
                "<node_id>": {
                    "id": "<node_id>",
                    "message": {
                        "id": "...",
                        "author": { "role": "user" | "assistant" | "system" | "tool" },
                        "content": {
                            "content_type": "text",
                            "parts": ["message text here"]
                        },
                        "create_time": 1705312200.0,
                        "metadata": {
                            "model_slug": "gpt-4",
                            ...
                        }
                    },
                    "parent": "<parent_node_id>" | null,
                    "children": ["<child_node_id>", ...]
                },
                ...
            },
            "id": "..."
        },
        ...
    ]

The mapping is a tree structure; we walk from root to leaf to reconstruct
the linear conversation thread.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from src.models.base import Platform
from src.models.message import ChatMessage, Conversation, Role
from src.ingest.base import BaseIngester

logger = logging.getLogger(__name__)

_ROLE_MAP = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "system": Role.SYSTEM,
    "tool": Role.TOOL,
}


class ChatGPTIngester(BaseIngester):
    """Parse ChatGPT data export JSON files."""

    platform_name = "chatgpt"

    def parse_export(
        self, data: Union[dict, list], source_file: Optional[Path] = None
    ) -> list[Conversation]:
        if isinstance(data, dict):
            conversations_data = [data]
        elif isinstance(data, list):
            conversations_data = data
        else:
            logger.warning("Unexpected ChatGPT export format")
            return []

        conversations: list[Conversation] = []
        for conv_data in conversations_data:
            conv = self._parse_conversation(conv_data)
            if conv:
                conversations.append(conv)

        return conversations

    def _parse_conversation(self, data: dict) -> Conversation | None:
        conv_id = data.get("id", "")
        mapping = data.get("mapping", {})
        if not conv_id or not mapping:
            return None

        # Walk the tree from root to extract the linear thread
        messages = self._extract_messages_from_tree(mapping, conv_id)

        created_at = self._epoch_to_datetime(data.get("create_time"))
        if not created_at:
            return None

        return Conversation(
            id=conv_id,
            title=data.get("title"),
            messages=messages,
            platform=Platform.CHATGPT,
            created_at=created_at,
            updated_at=self._epoch_to_datetime(data.get("update_time")),
        )

    def _extract_messages_from_tree(
        self, mapping: dict, conversation_id: str
    ) -> list[ChatMessage]:
        """Walk the ChatGPT mapping tree and extract messages in order."""
        # Find the root node (the one with no parent or parent not in mapping)
        root_id = None
        for node_id, node in mapping.items():
            parent = node.get("parent")
            if parent is None or parent not in mapping:
                root_id = node_id
                break

        if root_id is None:
            return []

        # Walk from root, always following the first child (main thread)
        messages: list[ChatMessage] = []
        current_id = root_id
        while current_id:
            node = mapping.get(current_id)
            if not node:
                break

            msg = self._parse_node_message(node, conversation_id)
            if msg:
                messages.append(msg)

            children = node.get("children", [])
            current_id = children[0] if children else None

        return messages

    def _parse_node_message(
        self, node: dict, conversation_id: str
    ) -> ChatMessage | None:
        msg_data = node.get("message")
        if not msg_data:
            return None

        author = msg_data.get("author", {})
        role = _ROLE_MAP.get(author.get("role", ""))
        if role is None:
            return None

        # Extract text content from parts
        content_obj = msg_data.get("content", {})
        parts = content_obj.get("parts", [])
        text_parts = [p for p in parts if isinstance(p, str)]
        content = "\n".join(text_parts)

        if not content.strip():
            return None

        metadata = msg_data.get("metadata", {})
        model = metadata.get("model_slug")

        return ChatMessage(
            id=msg_data.get("id", node.get("id", "")),
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=self._epoch_to_datetime(msg_data.get("create_time")),
            platform=Platform.CHATGPT,
            model=model,
        )

    @staticmethod
    def _epoch_to_datetime(epoch: float | None) -> datetime | None:
        if epoch is None:
            return None
        try:
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            return None
