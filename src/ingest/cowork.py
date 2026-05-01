"""Parser for Cowork/Dispatch session transcript JSONL files.

Cowork (Claude Agent Mode / Dispatch) stores session transcripts as JSONL files.
The default search root is:
  ~/Library/Application Support/Claude/local-agent-mode-sessions/

This ingester reads the inner session transcript files under
.claude/projects/.../*.jsonl. Outer audit.jsonl files describe Agent Mode
runtime bookkeeping and are skipped during discovery.

Each file contains one JSON object per line representing a message event:

  - type "user": user turn — content is a string or list of blocks
  - type "assistant": assistant turn — content is a list of blocks
      (text, thinking, tool_use)
  - type "queue-operation": scheduling metadata — skipped

Tool-call arguments (tool_use input) and text tool results are retained alongside
human-readable text blocks from user/assistant turns.
Internal sidechain records (isSidechain=true) are also skipped.

Session title is derived from the "cwd" field or directory path slug
(e.g. /sessions/eloquent-nice-knuth → "eloquent-nice-knuth").
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from src.models.base import Platform
from src.models.message import ChatMessage, Conversation, Role
from src.ingest.base import BaseIngester

logger = logging.getLogger(__name__)

# Regex to pull a scheduled-task name from queue-operation content
_TASK_NAME_RE = re.compile(r'<scheduled-task\s+name="([^"]+)"')


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _compact_json(value) -> str:
    """Return compact, stable JSON for tool arguments when possible."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _extract_tool_result_text(block: dict) -> list[str]:
    """Return text content from a tool_result block."""
    content = block.get("content")
    if isinstance(content, str):
        text = content.strip()
        return [text] if text else []
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text", "").strip()
            if text:
                parts.append(text)
        return parts
    return []


def _extract_text(content) -> str:
    """Return user-visible text and work content from a message content field.

    Skips thinking blocks and non-text tool_result content.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(text)
            elif block.get("type") == "tool_use":
                tool_args = block.get("input")
                if tool_args is None:
                    continue
                parts.append(f"tool_use args: {_compact_json(tool_args)}")
            elif block.get("type") == "tool_result":
                parts.extend(_extract_tool_result_text(block))
        return "\n\n".join(parts)
    return ""


def _session_title_from_path(file_path: Path) -> str | None:
    """Derive a slug title from a path component like '-sessions-eloquent-nice-knuth'."""
    for part in file_path.parts:
        if part.startswith("-sessions-"):
            return part[len("-sessions-"):]
    return None


def _session_title_from_cwd(cwd: str | None) -> str | None:
    """Derive a slug title from a cwd like '/sessions/eloquent-nice-knuth'."""
    if not cwd:
        return None
    # Last non-empty path component
    parts = [p for p in cwd.split("/") if p]
    if parts:
        return parts[-1]
    return None


def _first_model(messages: list[ChatMessage]) -> str | None:
    return next((msg.model for msg in messages if msg.model), None)


class CoworkIngester(BaseIngester):
    """Parse Cowork/Dispatch session JSONL transcript files."""

    platform_name = "cowork"

    def ingest(self, path: Path) -> list[Conversation]:
        """Ingest from a directory (recursively) or a single .jsonl file."""
        path = Path(path).expanduser()
        if path.is_dir():
            return self._ingest_directory(path)
        return self._ingest_jsonl_file(path)

    def _ingest_directory(self, directory: Path) -> list[Conversation]:
        """Recursively discover and parse all session .jsonl files."""
        conversations: list[Conversation] = []
        for jsonl_file in sorted(directory.rglob("*.jsonl")):
            # Skip audit logs and subagent traces. The outer audit.jsonl layer is
            # where isReplay appears; inner .claude/projects/... session JSONL
            # records do not carry that field, so no per-record isReplay filter
            # belongs in the inner transcript parser.
            if jsonl_file.name.startswith("audit") and jsonl_file.name.endswith(".jsonl"):
                continue
            if "subagents" in jsonl_file.parts:
                continue
            conversations.extend(self._ingest_jsonl_file(jsonl_file))

        # Deduplicate: if the same sessionId appears in multiple files,
        # keep the last-modified copy (files are sorted so last wins).
        seen: dict[str, Conversation] = {}
        for conv in conversations:
            seen[conv.id] = conv
        unique = list(seen.values())

        # Apply min_messages filter
        unique = [c for c in unique if c.message_count >= self.min_messages]

        logger.info(
            "%s: ingested %d conversations from %s",
            self.platform_name, len(unique), directory,
        )
        return unique

    def _ingest_jsonl_file(self, file_path: Path) -> list[Conversation]:
        """Parse a single .jsonl session transcript file."""
        try:
            raw = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return []

        records: list[dict] = []
        for i, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.debug("Skipping malformed line %d in %s: %s", i + 1, file_path, exc)

        if not records:
            return []

        conv = self._parse_session(records, file_path)
        if conv is None:
            return []
        # min_messages filtering happens in _ingest_directory; apply here too
        # for single-file calls
        if conv.message_count < self.min_messages:
            return []
        return [conv]

    def parse_export(
        self,
        data: Union[dict, list],
        source_file: Optional[Path] = None,
    ) -> list[Conversation]:
        """Required by BaseIngester ABC; not used directly for JSONL files."""
        return []

    def _parse_session(
        self, records: list[dict], file_path: Path
    ) -> Conversation | None:
        """Build a Conversation from the ordered list of JSONL records."""
        session_id: str | None = None
        task_name: str | None = None          # from queue-operation content
        first_cwd: str | None = None
        messages: list[ChatMessage] = []
        first_ts: datetime | None = None
        last_ts: datetime | None = None

        for record in records:
            rec_type = record.get("type")

            # Capture session ID from any record
            if not session_id:
                session_id = record.get("sessionId")

            # Try to extract a human-readable task name from enqueue operations
            if rec_type == "queue-operation" and record.get("operation") == "enqueue":
                content_str = record.get("content", "")
                m = _TASK_NAME_RE.search(content_str)
                if m and not task_name:
                    task_name = m.group(1)

            if rec_type not in ("user", "assistant"):
                continue

            # Skip agent-internal sidechain scratch work from research subagents;
            # it is not user-facing dialogue for brain-ingest semantics.
            if record.get("isSidechain"):
                continue

            msg_data = record.get("message", {})
            role_str = msg_data.get("role", "")
            if role_str == "user":
                role = Role.USER
            elif role_str == "assistant":
                role = Role.ASSISTANT
            else:
                continue

            content_raw = msg_data.get("content", "")

            content = _extract_text(content_raw)
            if not content:
                continue

            # Capture cwd for title derivation
            if not first_cwd:
                first_cwd = record.get("cwd")

            ts = _parse_timestamp(record.get("timestamp"))
            if ts:
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts

            # For assistant records, model is in message.model (API response shape)
            model: str | None = None
            if role == Role.ASSISTANT:
                model = msg_data.get("model")

            messages.append(
                ChatMessage(
                    id=record.get("uuid", ""),
                    conversation_id=session_id or file_path.stem,
                    role=role,
                    content=content,
                    timestamp=ts,
                    platform=Platform.COWORK,
                    model=model,
                )
            )

        if not session_id:
            session_id = file_path.stem

        if not messages:
            return None

        messages.sort(key=lambda m: m.timestamp or datetime.min.replace(tzinfo=timezone.utc))

        # Build title: prefer task name, then path slug, then session ID
        title = (
            task_name
            or _session_title_from_path(file_path)
            or _session_title_from_cwd(first_cwd)
            or session_id
        )
        created_at = first_ts or datetime.now(timezone.utc)

        return Conversation(
            id=session_id,
            title=title,
            messages=messages,
            platform=Platform.COWORK,
            author="cowork-claude",
            session_id=title,
            session_slug=title,
            created_at_iso=created_at.isoformat(),
            model_id=_first_model(messages),
            ingested_by="pipeline-cowork-ingester",
            created_at=created_at,
            updated_at=last_ts,
        )
