"""Tests for Move 1 additions to brain_query.py."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


BRAIN_QUERY_PATH = Path.home() / "code" / "brain-tools" / "brain_query.py"
FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "brain_query" / "vault"


def _load_brain_query():
    spec = importlib.util.spec_from_file_location("brain_query_move1", BRAIN_QUERY_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_core_flag_returns_shared_and_agent_context(monkeypatch, capsys):
    brain_query = _load_brain_query()
    monkeypatch.setattr(brain_query, "VAULT_DIR", FIXTURE_VAULT)
    monkeypatch.setattr(sys, "argv", ["brain_query.py", "--load-core", "chat-claude"])

    brain_query.main()

    output = capsys.readouterr().out
    assert "Shared Core Fixture" in output
    assert "Agent Core Fixture" in output


def test_author_filter_restricts_semantic_results(monkeypatch, tmp_path, capsys):
    brain_query = _load_brain_query()
    monkeypatch.setattr(brain_query, "VAULT_DIR", FIXTURE_VAULT)
    monkeypatch.setattr(brain_query, "QUERY_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(brain_query, "GRAPH_FILE", tmp_path / "missing-graph.json")

    def fake_query_vault(query_text, n_results=5):
        return [
            {
                "title": "Claude Note",
                "source": "AI-Conversations/claude-note.md",
                "note_type": "conversation",
                "tags": "",
                "snippet": "Relevant Claude-authored content.",
                "distance": 0.1,
                "salience": 0.9,
            },
            {
                "title": "Codex Note",
                "source": "AI-Conversations/codex-note.md",
                "note_type": "conversation",
                "tags": "",
                "snippet": "Relevant Codex-authored content.",
                "distance": 0.2,
                "salience": 0.8,
            },
        ][:n_results]

    monkeypatch.setitem(
        sys.modules,
        "vault_embedder",
        types.SimpleNamespace(query_vault=fake_query_vault),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "brain_query.py",
            "relevant",
            "--filter",
            "author=chat-claude",
            "--no-graph",
        ],
    )

    brain_query.main()

    output = capsys.readouterr().out
    assert "Claude Note" in output
    assert "Codex Note" not in output
