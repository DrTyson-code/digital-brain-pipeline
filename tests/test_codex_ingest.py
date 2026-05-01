"""Tests for the Codex session JSONL ingester."""

from __future__ import annotations

import json
from pathlib import Path

from src.ingest.codex import CodexIngester
from src.models.base import Platform
from src.models.message import Role
from src.pipeline import PipelineConfig


FIXTURE = Path(__file__).parent / "fixtures" / "codex" / "redacted_session.jsonl"


def _write_jsonl(records: list[dict], path: Path) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _fixture_records() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def test_fixture_session_returns_conversation_with_codex_metadata():
    ingester = CodexIngester(min_messages=2)
    conversations = ingester.ingest(FIXTURE)

    assert len(conversations) == 1
    conv = conversations[0]
    assert conv.id == "019de4fc-redacted-0000-0000-000000000000"
    assert conv.platform == Platform.CODEX
    assert conv.title == "Codex: implement-codex-ingestion-for-the-digital-brain-pipeline"
    assert conv.author == (
        "codex-implement-codex-ingestion-for-the-digital-brain-pipeline-"
        "2026-05-01-192116"
    )
    assert conv.source == conv.author
    assert conv.created_at.isoformat() == "2026-05-01T19:21:16.621000+00:00"


def test_role_mapping_and_shim_filtering():
    conv = CodexIngester(min_messages=2).ingest(FIXTURE)[0]

    assert [message.role for message in conv.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.TOOL,
    ]
    contents = [message.content for message in conv.messages]
    assert "Use the repository's existing ingestion patterns." in contents
    assert "Implement Codex ingestion for the digital brain pipeline." in contents
    assert all("<environment_context>" not in content for content in contents)
    assert all("<permissions instructions>" not in content for content in contents)


def test_assistant_model_is_captured_from_turn_context():
    conv = CodexIngester(min_messages=2).ingest(FIXTURE)[0]

    assert conv.assistant_messages[0].model == "gpt-5.5"
    tool_models = [message.model for message in conv.messages if message.role == Role.TOOL]
    assert tool_models == ["gpt-5.5", "gpt-5.5"]


def test_min_messages_filter_removes_short_sessions(tmp_path):
    records = _fixture_records()
    path = _write_jsonl(records[:7], tmp_path / "rollout-short.jsonl")

    assert CodexIngester(min_messages=4).ingest(path) == []


def test_sessions_with_only_shim_user_turns_are_skipped(tmp_path):
    records = [
        record
        for record in _fixture_records()
        if not (
            record.get("type") == "response_item"
            and record.get("payload", {}).get("role") == "user"
            and "Implement Codex ingestion" in json.dumps(record)
        )
    ]
    path = _write_jsonl(records, tmp_path / "rollout-shim-only.jsonl")

    assert CodexIngester(min_messages=1).ingest(path) == []


def test_directory_ingestion_finds_rollout_jsonl_files(tmp_path):
    session_dir = tmp_path / "2026" / "05" / "01"
    session_dir.mkdir(parents=True)
    _write_jsonl(_fixture_records(), session_dir / "rollout-2026-05-01T19-21-16-redacted.jsonl")
    _write_jsonl(_fixture_records(), session_dir / "not-a-rollout.jsonl")

    conversations = CodexIngester(min_messages=2).ingest(tmp_path)

    assert len(conversations) == 1
    assert conversations[0].platform == Platform.CODEX


def test_settings_style_null_sources_are_skipped(tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """
vault:
  path: ~/Vault/Claude-Brain
ingest:
  sources:
    calendar: null
    codex: ~/.codex/sessions
  min_messages: 2
""",
        encoding="utf-8",
    )

    config = PipelineConfig.from_yaml(config_path)

    assert "calendar" not in config.source_dirs
    assert config.source_dirs["codex"] == Path("~/.codex/sessions").expanduser()
