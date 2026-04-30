"""Tests for the Cowork/Dispatch session JSONL ingester."""

import json
import tempfile
from pathlib import Path

import pytest

from src.ingest.cowork import CoworkIngester
from src.models.base import Platform
from src.models.message import Role


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict], suffix: str = ".jsonl") -> Path:
    """Write a list of dicts as JSONL to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    for record in records:
        f.write(json.dumps(record) + "\n")
    f.close()
    return Path(f.name)


def _write_jsonl_in_dir(records: list[dict], dir_path: Path, filename: str) -> Path:
    """Write JSONL into an existing directory at a specific filename."""
    file_path = dir_path / filename
    file_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return file_path


# ---------------------------------------------------------------------------
# Sample JSONL data
# ---------------------------------------------------------------------------

SESSION_ID = "sess-abc123"

BASIC_SESSION = [
    {
        "type": "queue-operation",
        "operation": "enqueue",
        "timestamp": "2026-04-01T10:00:00.000Z",
        "sessionId": SESSION_ID,
        "content": '<scheduled-task name="check-something">Do a thing</scheduled-task>',
    },
    {
        "type": "queue-operation",
        "operation": "dequeue",
        "timestamp": "2026-04-01T10:00:00.100Z",
        "sessionId": SESSION_ID,
    },
    {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "uuid": "msg-u-001",
        "timestamp": "2026-04-01T10:00:00.200Z",
        "sessionId": SESSION_ID,
        "message": {
            "role": "user",
            "content": "Please check the server status.",
        },
    },
    {
        "parentUuid": "msg-u-001",
        "isSidechain": False,
        "type": "assistant",
        "uuid": "msg-a-001",
        "timestamp": "2026-04-01T10:00:02.000Z",
        "sessionId": SESSION_ID,
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [
                {"type": "thinking", "thinking": "Let me check the status."},
                {"type": "text", "text": "I'll check the server status now."},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Bash",
                    "input": {"command": "uptime"},
                },
            ],
        },
    },
    {
        "parentUuid": "msg-a-001",
        "isSidechain": False,
        "type": "user",
        "uuid": "msg-u-002",
        "timestamp": "2026-04-01T10:00:03.000Z",
        "sessionId": SESSION_ID,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": [{"type": "text", "text": "12:00  up 3 days"}],
                }
            ],
        },
    },
    {
        "parentUuid": "msg-u-002",
        "isSidechain": False,
        "type": "assistant",
        "uuid": "msg-a-002",
        "timestamp": "2026-04-01T10:00:04.000Z",
        "sessionId": SESSION_ID,
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [
                {"type": "text", "text": "The server has been up for 3 days."},
            ],
        },
    },
]

# A session where some records have isSidechain=True (should be ignored)
SIDECHAIN_SESSION = [
    {
        "type": "user",
        "uuid": "msg-u-main",
        "isSidechain": False,
        "timestamp": "2026-04-02T09:00:00.000Z",
        "sessionId": "sess-side-001",
        "message": {"role": "user", "content": "Main user message"},
    },
    {
        "type": "assistant",
        "uuid": "msg-a-side",
        "isSidechain": True,
        "timestamp": "2026-04-02T09:00:01.000Z",
        "sessionId": "sess-side-001",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Sidechain assistant message"}],
        },
    },
    {
        "type": "assistant",
        "uuid": "msg-a-main",
        "isSidechain": False,
        "timestamp": "2026-04-02T09:00:02.000Z",
        "sessionId": "sess-side-001",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Main assistant reply"}],
        },
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicParsing:
    def test_basic_session_returns_one_conversation(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(path)
        assert len(conversations) == 1

    def test_conversation_id_is_session_id(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        assert conv.id == SESSION_ID

    def test_platform_is_cowork(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        assert conv.platform == Platform.COWORK

    def test_title_extracted_from_scheduled_task_name(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        assert conv.title == "check-something"

    def test_tool_result_user_records_retain_text_content(self):
        """User records with text tool_result blocks should retain that output."""
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        user_msgs = conv.user_messages
        assert len(user_msgs) == 2
        assert user_msgs[0].content == "Please check the server status."
        assert user_msgs[1].content == "12:00  up 3 days"

    def test_thinking_blocks_stripped_from_assistant(self):
        """thinking blocks must not appear in assistant message content."""
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        first_assistant = conv.assistant_messages[0]
        assert "Let me check" not in first_assistant.content
        assert "check the server status now" in first_assistant.content

    def test_tool_use_args_retained_from_assistant(self):
        """tool_use argument dictionaries should appear in assistant message content."""
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        first_assistant = conv.assistant_messages[0]
        assert 'tool_use args: {"command":"uptime"}' in first_assistant.content

    def test_assistant_model_captured(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        assert conv.assistant_messages[0].model == "claude-opus-4-6"

    def test_message_roles(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        roles = [m.role for m in conv.messages]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

    def test_timestamps_are_parsed(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        assert conv.created_at is not None
        for msg in conv.messages:
            assert msg.timestamp is not None


class TestSidechainFiltering:
    def test_sidechain_records_excluded(self):
        path = _write_jsonl(SIDECHAIN_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        # Sidechain assistant message should be dropped
        contents = [m.content for m in conv.assistant_messages]
        assert "Sidechain assistant message" not in contents
        assert "Main assistant reply" in contents

    def test_message_count_without_sidechains(self):
        path = _write_jsonl(SIDECHAIN_SESSION)
        ingester = CoworkIngester(min_messages=1)
        conv = ingester.ingest(path)[0]
        assert conv.message_count == 2  # 1 user + 1 non-sidechain assistant


class TestMinMessagesFilter:
    def test_min_messages_filters_short_sessions(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=10)
        conversations = ingester.ingest(path)
        assert conversations == []

    def test_min_messages_passes_long_enough_session(self):
        path = _write_jsonl(BASIC_SESSION)
        ingester = CoworkIngester(min_messages=2)
        conversations = ingester.ingest(path)
        assert len(conversations) == 1


class TestDirectoryIngestion:
    def test_directory_finds_jsonl_files(self, tmp_path):
        session_dir = tmp_path / ".claude" / "projects" / "-sessions-my-test-session"
        session_dir.mkdir(parents=True)
        _write_jsonl_in_dir(BASIC_SESSION, session_dir, "abc123.jsonl")

        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(tmp_path)
        assert len(conversations) == 1

    def test_directory_title_from_path_slug(self, tmp_path):
        """When no queue-operation task name, title falls back to path slug."""
        session_dir = tmp_path / ".claude" / "projects" / "-sessions-my-slug"
        session_dir.mkdir(parents=True)
        # Session without a queue-operation task name
        no_task_records = [r for r in BASIC_SESSION if r.get("operation") != "enqueue"]
        _write_jsonl_in_dir(no_task_records, session_dir, "abc123.jsonl")

        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(tmp_path)
        assert len(conversations) == 1
        assert conversations[0].title == "my-slug"

    def test_directory_skips_audit_files(self, tmp_path):
        _write_jsonl_in_dir(BASIC_SESSION, tmp_path, "audit.jsonl")

        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(tmp_path)
        assert conversations == []

    def test_directory_skips_subagent_files(self, tmp_path):
        subagents_dir = tmp_path / "subagents"
        subagents_dir.mkdir()
        _write_jsonl_in_dir(BASIC_SESSION, subagents_dir, "agent-xyz.jsonl")

        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(tmp_path)
        assert conversations == []

    def test_directory_deduplicates_by_session_id(self, tmp_path):
        """Same sessionId in two files → only one conversation kept."""
        _write_jsonl_in_dir(BASIC_SESSION, tmp_path, "file1.jsonl")
        _write_jsonl_in_dir(BASIC_SESSION, tmp_path, "file2.jsonl")

        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(tmp_path)
        assert len(conversations) == 1

    def test_directory_multiple_distinct_sessions(self, tmp_path):
        """Two files with different sessionIds → two conversations."""
        session_b = [
            {**r, "sessionId": "sess-other-999"}
            if r.get("sessionId") else r
            for r in BASIC_SESSION
        ]
        # Fix conversation_id references in message records
        _write_jsonl_in_dir(BASIC_SESSION, tmp_path, "session_a.jsonl")
        _write_jsonl_in_dir(session_b, tmp_path, "session_b.jsonl")

        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(tmp_path)
        assert len(conversations) == 2


class TestEdgeCases:
    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        ingester = CoworkIngester(min_messages=1)
        assert ingester.ingest(f) == []

    def test_malformed_lines_skipped(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        lines = [
            "not valid json {{{",
            json.dumps(BASIC_SESSION[2]),  # valid user text record
            json.dumps(BASIC_SESSION[5]),  # valid assistant text record
        ]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(f)
        assert len(conversations) == 1
        assert conversations[0].message_count == 2

    def test_all_text_tool_result_records_returns_conversation(self):
        """A session with text tool_result user records retains those results."""
        only_tool_results = [
            {
                "type": "user",
                "uuid": "msg-x",
                "isSidechain": False,
                "timestamp": "2026-04-01T10:00:00.000Z",
                "sessionId": "sess-tools-only",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                    ],
                },
            }
        ]
        path = _write_jsonl(only_tool_results)
        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(path)
        assert len(conversations) == 1
        assert conversations[0].messages[0].content == "ok"

    def test_session_id_falls_back_to_filename_stem(self):
        """When no sessionId field exists, conversation id = file stem."""
        records_no_session_id = [
            {
                "type": "user",
                "uuid": "u1",
                "isSidechain": False,
                "timestamp": "2026-04-01T10:00:00.000Z",
                "message": {"role": "user", "content": "Hello"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "isSidechain": False,
                "timestamp": "2026-04-01T10:00:01.000Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there"}],
                },
            },
        ]
        path = _write_jsonl(records_no_session_id)
        ingester = CoworkIngester(min_messages=1)
        conversations = ingester.ingest(path)
        assert len(conversations) == 1
        assert conversations[0].id == path.stem
