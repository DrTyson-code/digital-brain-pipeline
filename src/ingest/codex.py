"""Parser for Codex CLI session transcript JSONL files."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from src.ingest.base import BaseIngester
from src.models.base import Platform
from src.models.message import ChatMessage, Conversation, Role

logger = logging.getLogger(__name__)

_SHIM_PREFIXES = ("<environment_context>", "<permissions instructions>")
_SLUG_WORD_RE = re.compile(r"[a-z0-9]+")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _format_author_timestamp(value: datetime | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%d-%H%M%S")


def _is_shim_text(text: str) -> bool:
    return text.lstrip().startswith(_SHIM_PREFIXES)


def _extract_content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"input_text", "output_text", "text"}:
            text = str(block.get("text", "")).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _compact_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _extract_tool_event_text(payload: dict) -> str:
    payload_type = payload.get("type")
    if payload_type in {"function_call", "custom_tool_call"}:
        name = payload.get("name") or payload.get("call_id") or payload_type
        args = payload.get("arguments")
        if args is None:
            args = payload.get("input")
        if args in (None, ""):
            return str(name)
        return f"{name}: {_compact_json(args)}"

    if payload_type in {"function_call_output", "custom_tool_call_output"}:
        output = payload.get("output")
        if isinstance(output, str):
            return output.strip()
        if output is None:
            return ""
        return _compact_json(output)

    return ""


def _slugify_task(text: str, max_len: int = 64) -> str:
    words = _SLUG_WORD_RE.findall(text.lower())
    if not words:
        return ""
    slug = "-".join(words[:10])
    if len(slug) <= max_len:
        return slug
    return slug[:max_len].rstrip("-")


def _fallback_slug(cwd: str | None, originator: str | None, session_id: str) -> str:
    if cwd:
        basename = Path(cwd).name
        slug = _slugify_task(basename)
        if slug:
            return slug
    if originator:
        slug = _slugify_task(originator)
        if slug:
            return slug
    return session_id[:8] if session_id else "session"


def _role_from_codex(role: str | None) -> Role | None:
    if role == "assistant":
        return Role.ASSISTANT
    if role == "user":
        return Role.USER
    if role == "developer":
        return Role.SYSTEM
    return None


class CodexIngester(BaseIngester):
    """Parse Codex CLI session JSONL transcript files."""

    platform_name = "codex"

    def ingest(self, path: Path) -> list[Conversation]:
        path = Path(path).expanduser()
        if path.is_dir():
            return self._ingest_directory(path)
        return self._ingest_jsonl_file(path)

    def _ingest_directory(self, directory: Path) -> list[Conversation]:
        conversations: list[Conversation] = []
        for jsonl_file in sorted(directory.rglob("rollout-*.jsonl")):
            conversations.extend(self._ingest_jsonl_file(jsonl_file))

        seen: dict[str, Conversation] = {}
        for conv in conversations:
            seen[conv.id] = conv
        unique = [c for c in seen.values() if c.message_count >= self.min_messages]

        logger.info(
            "%s: ingested %d conversations from %s",
            self.platform_name,
            len(unique),
            directory,
        )
        return unique

    def _ingest_jsonl_file(self, file_path: Path) -> list[Conversation]:
        records: list[dict] = []
        try:
            raw = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return []

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
        if conv is None or conv.message_count < self.min_messages:
            return []
        return [conv]

    def parse_export(
        self,
        data: Union[dict, list],
        source_file: Optional[Path] = None,
    ) -> list[Conversation]:
        """Required by BaseIngester ABC; Codex sessions are line-oriented JSONL."""
        return []

    def _parse_session(self, records: list[dict], file_path: Path) -> Conversation | None:
        session_id = file_path.stem
        session_ts: datetime | None = None
        cwd: str | None = None
        originator: str | None = None
        model: str | None = None
        messages: list[ChatMessage] = []
        first_meaningful_user: str | None = None
        last_ts: datetime | None = None

        for record in records:
            record_type = record.get("type")
            payload = record.get("payload") or {}
            ts = _parse_timestamp(record.get("timestamp"))
            if ts and (last_ts is None or ts > last_ts):
                last_ts = ts

            if record_type == "session_meta":
                session_id = payload.get("id") or session_id
                session_ts = _parse_timestamp(payload.get("timestamp")) or ts
                cwd = payload.get("cwd") or cwd
                originator = payload.get("originator") or originator
                continue

            if record_type == "turn_context":
                model = payload.get("model") or model
                cwd = payload.get("cwd") or cwd
                continue

            if record_type != "response_item":
                continue

            payload_type = payload.get("type")
            message_ts = ts
            if payload_type == "message":
                role = _role_from_codex(payload.get("role"))
                if role is None:
                    continue
                content = _extract_content_text(payload.get("content"))
                if not content or _is_shim_text(content):
                    continue
                if role == Role.USER and first_meaningful_user is None:
                    first_meaningful_user = content
                messages.append(
                    ChatMessage(
                        id=payload.get("id") or record.get("call_id") or "",
                        conversation_id=session_id,
                        role=role,
                        content=content,
                        timestamp=message_ts,
                        platform=Platform.CODEX,
                        model=model if role == Role.ASSISTANT else None,
                    )
                )
                continue

            tool_content = _extract_tool_event_text(payload)
            if tool_content:
                messages.append(
                    ChatMessage(
                        id=payload.get("call_id") or payload.get("id") or "",
                        conversation_id=session_id,
                        role=Role.TOOL,
                        content=tool_content,
                        timestamp=message_ts,
                        platform=Platform.CODEX,
                        model=model,
                    )
                )

        if not messages or first_meaningful_user is None:
            return None

        created_at = session_ts or min(
            (m.timestamp for m in messages if m.timestamp),
            default=datetime.now(timezone.utc),
        )
        messages.sort(key=lambda m: m.timestamp or datetime.min.replace(tzinfo=timezone.utc))
        slug = _slugify_task(first_meaningful_user) or _fallback_slug(cwd, originator, session_id)
        author_id = f"codex-{slug}-{_format_author_timestamp(created_at)}"

        return Conversation(
            id=session_id,
            title=f"Codex: {slug}",
            messages=messages,
            platform=Platform.CODEX,
            author="codex-cli",
            session_id=author_id,
            session_slug=slug,
            created_at_iso=created_at.isoformat(),
            model_id=model,
            ingested_by="pipeline-codex-ingester",
            source=author_id,
            created_at=created_at,
            updated_at=last_ts,
        )
