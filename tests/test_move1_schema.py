"""Move 1 v2 schema tests across ingesters and writer integration."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from src.ingest.calendar import CalendarIngester
from src.ingest.chatgpt import ChatGPTIngester
from src.ingest.claude import ClaudeIngester
from src.ingest.codex import CodexIngester
from src.ingest.cowork import CoworkIngester
from src.ingest.gemini import GeminiIngester
from src.output.obsidian import ObsidianWriter


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
CODEX_FIXTURE = Path(__file__).parent / "fixtures" / "codex" / "redacted_session.jsonl"


def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _frontmatter(path: Path) -> dict:
    match = FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    assert match is not None
    return yaml.safe_load(match.group(1))


def _claude_export() -> list[dict]:
    return [
        {
            "uuid": "claude-conv",
            "name": "Claude Schema Test",
            "created_at": "2026-05-01T10:00:00.000000+00:00",
            "updated_at": "2026-05-01T10:01:00.000000+00:00",
            "chat_messages": [
                {
                    "uuid": "claude-msg-1",
                    "sender": "human",
                    "text": "Hello",
                    "created_at": "2026-05-01T10:00:00.000000+00:00",
                },
                {
                    "uuid": "claude-msg-2",
                    "sender": "assistant",
                    "text": "Hi",
                    "model": "claude-sonnet-4-6",
                    "created_at": "2026-05-01T10:00:10.000000+00:00",
                },
            ],
        }
    ]


def _chatgpt_export() -> list[dict]:
    return [
        {
            "id": "chatgpt-conv",
            "title": "ChatGPT Schema Test",
            "create_time": 1777633200.0,
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": ["u"]},
                "u": {
                    "id": "u",
                    "message": {
                        "id": "u",
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                        "create_time": 1777633200.0,
                        "metadata": {},
                    },
                    "parent": "root",
                    "children": ["a"],
                },
                "a": {
                    "id": "a",
                    "message": {
                        "id": "a",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Hi"]},
                        "create_time": 1777633201.0,
                        "metadata": {"model_slug": "gpt-4o"},
                    },
                    "parent": "u",
                    "children": [],
                },
            },
        }
    ]


def _gemini_export() -> list[dict]:
    return [
        {
            "id": "gemini-conv",
            "name": "Gemini Schema Test",
            "create_time": "2026-05-01T10:00:00Z",
            "metadata": {"model_slug": "gemini-2.5-pro"},
            "entries": [
                {
                    "id": "g-u",
                    "role": "User",
                    "parts": [{"text": "Hello"}],
                    "create_time": "2026-05-01T10:00:00Z",
                },
                {
                    "id": "g-a",
                    "role": "Model",
                    "parts": [{"text": "Hi"}],
                    "create_time": "2026-05-01T10:00:01Z",
                },
            ],
        }
    ]


def _cowork_session() -> list[dict]:
    return [
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": "2026-05-01T10:00:00.000Z",
            "sessionId": "cowork-session-id",
            "content": '<scheduled-task name="cowork-schema-test">Do it</scheduled-task>',
        },
        {
            "type": "user",
            "uuid": "cw-u",
            "timestamp": "2026-05-01T10:00:01.000Z",
            "sessionId": "cowork-session-id",
            "message": {"role": "user", "content": "Hello"},
        },
        {
            "type": "assistant",
            "uuid": "cw-a",
            "timestamp": "2026-05-01T10:00:02.000Z",
            "sessionId": "cowork-session-id",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Hi"}],
            },
        },
    ]


def _calendar_export() -> list[dict]:
    return [
        {
            "id": "event-1",
            "summary": "Planning Meeting",
            "status": "confirmed",
            "start": {"dateTime": "2026-05-01T10:00:00+00:00"},
            "end": {"dateTime": "2026-05-01T10:30:00+00:00"},
        }
    ]


def test_ingesters_populate_move1_metadata(tmp_path):
    cases = [
        (
            "claude",
            ClaudeIngester(min_messages=1).ingest(
                _write_json(tmp_path / "claude.json", _claude_export())
            )[0],
            "chat-claude",
            "pipeline-claude-export",
            "claude-conv",
            "claude-sonnet-4-6",
        ),
        (
            "chatgpt",
            ChatGPTIngester(min_messages=1).ingest(
                _write_json(tmp_path / "chatgpt.json", _chatgpt_export())
            )[0],
            "chat-chatgpt",
            "pipeline-chatgpt-ingester",
            "chatgpt-conv",
            "gpt-4o",
        ),
        (
            "gemini",
            GeminiIngester(min_messages=1).ingest(
                _write_json(tmp_path / "gemini.json", _gemini_export())
            )[0],
            "chat-gemini",
            "pipeline-gemini-ingester",
            "gemini-conv",
            "gemini-2.5-pro",
        ),
        (
            "cowork",
            CoworkIngester(min_messages=1).ingest(
                _write_jsonl(tmp_path / "cowork.jsonl", _cowork_session())
            )[0],
            "cowork-claude",
            "pipeline-cowork-ingester",
            "cowork-schema-test",
            "claude-sonnet-4-6",
        ),
        (
            "codex",
            CodexIngester(min_messages=2).ingest(CODEX_FIXTURE)[0],
            "codex-cli",
            "pipeline-codex-ingester",
            "codex-implement-codex-ingestion-for-the-digital-brain-pipeline-2026-05-01-192116",
            "gpt-5.5",
        ),
        (
            "calendar",
            CalendarIngester(min_messages=1).parse_export(_calendar_export())[0],
            "william",
            "pipeline-calendar-ingester",
            "2026-05-01",
            None,
        ),
    ]

    for _, conv, author, ingested_by, session_id, model_id in cases:
        assert conv.author == author
        assert conv.ingested_by == ingested_by
        assert conv.session_id == session_id
        assert conv.created_at_iso
        assert conv.model_id == model_id


def test_writer_frontmatter_for_each_ingester_output(tmp_path):
    conversations = [
        ClaudeIngester(min_messages=1).ingest(
            _write_json(tmp_path / "claude.json", _claude_export())
        )[0],
        ChatGPTIngester(min_messages=1).ingest(
            _write_json(tmp_path / "chatgpt.json", _chatgpt_export())
        )[0],
        GeminiIngester(min_messages=1).ingest(
            _write_json(tmp_path / "gemini.json", _gemini_export())
        )[0],
        CoworkIngester(min_messages=1).ingest(
            _write_jsonl(tmp_path / "cowork.jsonl", _cowork_session())
        )[0],
        CodexIngester(min_messages=2).ingest(CODEX_FIXTURE)[0],
        CalendarIngester(min_messages=1).parse_export(_calendar_export())[0],
    ]

    writer = ObsidianWriter(vault_path=tmp_path / "vault")
    written = writer.write_all(conversations, [], [], [])

    assert len(written) == len(conversations)
    by_title = {_frontmatter(path)["title"]: _frontmatter(path) for path in written}
    assert by_title["Claude Schema Test"]["author"] == "chat-claude"
    assert by_title["ChatGPT Schema Test"]["author"] == "chat-chatgpt"
    assert by_title["Gemini Schema Test"]["author"] == "chat-gemini"
    assert by_title["cowork-schema-test"]["author"] == "cowork-claude"
    assert by_title[
        "Codex: implement-codex-ingestion-for-the-digital-brain-pipeline"
    ]["author"] == "codex-cli"
    assert by_title["2026-05-01 Schedule"]["author"] == "william"
    assert all(fm["memory_type"] == "episodic" for fm in by_title.values())
    assert all(fm["session_id"] for fm in by_title.values())
    assert all(fm["created_at"] for fm in by_title.values())


def test_claude_ingest_to_writer_integration_has_expected_schema(tmp_path):
    export_path = _write_json(tmp_path / "claude.json", _claude_export())
    conversations = ClaudeIngester(min_messages=1).ingest(export_path)
    writer = ObsidianWriter(vault_path=tmp_path / "vault")

    written = writer.write_all(conversations, [], [], [])
    fm = _frontmatter(written[0])

    assert fm["memory_type"] == "episodic"
    assert fm["author"] == "chat-claude"
    assert fm["session_id"] == "claude-conv"
    assert fm["created_at"] == "2026-05-01T10:00:00+00:00"
    assert fm["model_id"] == "claude-sonnet-4-6"
    assert fm["ingested_by"] == "pipeline-claude-export"
