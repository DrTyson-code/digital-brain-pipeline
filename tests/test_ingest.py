"""Tests for chat export ingesters."""

import json
import tempfile
from pathlib import Path

from src.ingest.claude import ClaudeIngester
from src.ingest.chatgpt import ChatGPTIngester
from src.ingest.gemini import GeminiIngester
from src.models.base import Platform


def _write_json(data, suffix=".json") -> Path:
    """Write data to a temp JSON file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    json.dump(data, f)
    f.close()
    return Path(f.name)


# --- Claude ---

CLAUDE_EXPORT = [
    {
        "uuid": "conv-001",
        "name": "Test Claude Conversation",
        "created_at": "2024-06-15T10:00:00.000000+00:00",
        "updated_at": "2024-06-15T11:00:00.000000+00:00",
        "chat_messages": [
            {
                "uuid": "msg-001",
                "text": "What is Python?",
                "sender": "human",
                "created_at": "2024-06-15T10:00:00.000000+00:00",
            },
            {
                "uuid": "msg-002",
                "text": "Python is a programming language.",
                "sender": "assistant",
                "created_at": "2024-06-15T10:00:30.000000+00:00",
            },
        ],
    }
]


def test_claude_ingester():
    path = _write_json(CLAUDE_EXPORT)
    ingester = ClaudeIngester(min_messages=1)
    conversations = ingester.ingest(path)
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv.id == "conv-001"
    assert conv.title == "Test Claude Conversation"
    assert conv.platform == Platform.CLAUDE
    assert conv.message_count == 2
    assert conv.messages[0].content == "What is Python?"
    assert conv.messages[1].content == "Python is a programming language."


def test_claude_min_messages_filter():
    path = _write_json(CLAUDE_EXPORT)
    ingester = ClaudeIngester(min_messages=5)
    conversations = ingester.ingest(path)
    assert len(conversations) == 0


# --- ChatGPT ---

CHATGPT_EXPORT = [
    {
        "id": "conv-gpt-001",
        "title": "Test ChatGPT Conversation",
        "create_time": 1718445600.0,
        "update_time": 1718449200.0,
        "mapping": {
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": ["msg1"],
            },
            "msg1": {
                "id": "msg1",
                "message": {
                    "id": "msg1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Tell me about Rust"]},
                    "create_time": 1718445600.0,
                    "metadata": {},
                },
                "parent": "root",
                "children": ["msg2"],
            },
            "msg2": {
                "id": "msg2",
                "message": {
                    "id": "msg2",
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "text",
                        "parts": ["Rust is a systems programming language."],
                    },
                    "create_time": 1718445630.0,
                    "metadata": {"model_slug": "gpt-4"},
                },
                "parent": "msg1",
                "children": [],
            },
        },
    }
]


def test_chatgpt_ingester():
    path = _write_json(CHATGPT_EXPORT)
    ingester = ChatGPTIngester(min_messages=1)
    conversations = ingester.ingest(path)
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv.id == "conv-gpt-001"
    assert conv.platform == Platform.CHATGPT
    assert conv.message_count == 2
    assert conv.messages[0].content == "Tell me about Rust"
    assert conv.messages[1].model == "gpt-4"


# --- Gemini ---

GEMINI_EXPORT = [
    {
        "id": "conv-gem-001",
        "name": "Test Gemini Conversation",
        "create_time": "2024-06-15T10:00:00Z",
        "update_time": "2024-06-15T11:00:00Z",
        "entries": [
            {
                "id": "entry-001",
                "role": "User",
                "parts": [{"text": "Explain Docker"}],
                "create_time": "2024-06-15T10:00:00Z",
            },
            {
                "id": "entry-002",
                "role": "Model",
                "parts": [{"text": "Docker is a containerization platform."}],
                "create_time": "2024-06-15T10:00:30Z",
            },
        ],
    }
]


def test_gemini_ingester():
    path = _write_json(GEMINI_EXPORT)
    ingester = GeminiIngester(min_messages=1)
    conversations = ingester.ingest(path)
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv.id == "conv-gem-001"
    assert conv.platform == Platform.GEMINI
    assert conv.message_count == 2
    assert conv.messages[0].content == "Explain Docker"


def test_invalid_json_returns_empty():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    f.write("not valid json {{{")
    f.close()
    ingester = ClaudeIngester()
    assert ingester.ingest(Path(f.name)) == []
