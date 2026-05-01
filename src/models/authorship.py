"""Canonical agent IDs for vault note authorship and pipeline provenance.

When adding a new agent ID:
- Add it to ``AUTHORING_AGENT_IDS`` or ``PIPELINE_AGENT_IDS`` below.
- Document the surface or role in the matching description map.
- If it is an authoring agent with its own continuity context, add a
  corresponding ``Core/agents/<agent-id>/`` directory in the vault.
"""

from __future__ import annotations

AUTHORING_AGENT_IDS = frozenset(
    {
        "chat-claude",
        "cowork-claude",
        "codex-cli",
        "chat-chatgpt",
        "chat-gemini",
        "william",
    }
)

PIPELINE_AGENT_IDS = frozenset(
    {
        "pipeline-synth-cluster",
        "pipeline-entity-extractor",
        "pipeline-moc-generator",
        "pipeline-claude-export",
        "pipeline-cowork-ingester",
        "pipeline-codex-ingester",
        "pipeline-chatgpt-ingester",
        "pipeline-gemini-ingester",
        "pipeline-calendar-ingester",
        "pipeline-concept-extractor",
    }
)

AGENT_IDS = AUTHORING_AGENT_IDS | PIPELINE_AGENT_IDS

AGENT_ID_DESCRIPTIONS = {
    "chat-claude": "claude.ai web/desktop chat exports",
    "cowork-claude": "Cowork / local agent mode Claude sessions",
    "codex-cli": "Codex CLI sessions",
    "chat-chatgpt": "ChatGPT web exports",
    "chat-gemini": "Gemini web exports",
    "william": "Human-authored sources",
    "pipeline-synth-cluster": "Cluster synthesis/reflection pipeline",
    "pipeline-entity-extractor": "Entity extraction pipeline",
    "pipeline-moc-generator": "Map-of-content generator",
    "pipeline-claude-export": "Claude export ingestion pipeline",
    "pipeline-cowork-ingester": "Cowork session ingestion pipeline",
    "pipeline-codex-ingester": "Codex session ingestion pipeline",
    "pipeline-chatgpt-ingester": "ChatGPT export ingestion pipeline",
    "pipeline-gemini-ingester": "Gemini export ingestion pipeline",
    "pipeline-calendar-ingester": "Calendar ingestion pipeline",
    "pipeline-concept-extractor": "Concept extraction pipeline",
}


def is_valid_agent_id(value: str) -> bool:
    """Return True when ``value`` is a registered agent or pipeline ID."""
    return value in AGENT_IDS
