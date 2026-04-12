"""Abstract base class for chat export ingesters."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from src.models.message import Conversation

logger = logging.getLogger(__name__)


class BaseIngester(ABC):
    """Base class for platform-specific chat export parsers.

    Subclasses implement `parse_export` to handle each platform's JSON format.
    """

    platform_name: str = "unknown"

    def __init__(self, min_messages: int = 2) -> None:
        self.min_messages = min_messages

    def ingest(self, path: Path) -> list[Conversation]:
        """Ingest conversations from a file or directory.

        Args:
            path: Path to an export file or directory of export files.

        Returns:
            List of parsed Conversation objects.
        """
        if path.is_dir():
            return self._ingest_directory(path)
        return self._ingest_file(path)

    def _ingest_directory(self, directory: Path) -> list[Conversation]:
        """Recursively ingest all JSON files in a directory."""
        conversations: list[Conversation] = []
        for json_file in sorted(directory.rglob("*.json")):
            conversations.extend(self._ingest_file(json_file))
        logger.info(
            "%s: ingested %d conversations from %s",
            self.platform_name,
            len(conversations),
            directory,
        )
        return conversations

    def _ingest_file(self, file_path: Path) -> list[Conversation]:
        """Parse a single export file."""
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Failed to parse %s: %s", file_path, e)
            return []

        conversations = self.parse_export(data, source_file=file_path)

        # Filter out short conversations
        conversations = [
            c for c in conversations if c.message_count >= self.min_messages
        ]
        return conversations

    @abstractmethod
    def parse_export(
        self, data: Union[dict, list], source_file: Optional[Path] = None
    ) -> list[Conversation]:
        """Parse platform-specific export data into Conversation objects.

        Args:
            data: Parsed JSON data from the export file.
            source_file: Path to the source file (for metadata).

        Returns:
            List of Conversation objects.
        """
        ...
