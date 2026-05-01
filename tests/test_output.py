"""Tests for the Obsidian output writer."""

import tempfile
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.models.base import Platform
from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import ChatMessage, Conversation, Role
from src.models.relationship import Relationship, RelationshipType
from src.output.obsidian import ObsidianWriter


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _make_writer(tmp_dir: Path) -> ObsidianWriter:
    return ObsidianWriter(vault_path=tmp_dir)


def _frontmatter(path: Path) -> dict:
    match = FRONTMATTER_RE.match(path.read_text())
    assert match is not None
    return yaml.safe_load(match.group(1))


def test_write_conversation():
    with tempfile.TemporaryDirectory() as tmp:
        writer = _make_writer(Path(tmp))
        conv = Conversation(
            id="conv-test",
            title="Test Conversation",
            messages=[
                ChatMessage(
                    conversation_id="conv-test",
                    role=Role.USER,
                    content="Hello",
                    platform=Platform.CLAUDE,
                ),
                ChatMessage(
                    conversation_id="conv-test",
                    role=Role.ASSISTANT,
                    content="Hi there!",
                    platform=Platform.CLAUDE,
                ),
            ],
            platform=Platform.CLAUDE,
            created_at=datetime.now(timezone.utc),
            topics=["programming"],
        )
        path = writer.write_conversation(conv, [])
        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "---" in content  # frontmatter
        assert "Test Conversation" in content
        assert "[[programming]]" in content
        assert "Hello" in content
        fm = _frontmatter(path)
        assert fm["memory_type"] == "episodic"
        assert fm["author"] == "chat-claude"
        assert fm["session_id"] == "conv-test"
        assert fm["created_at"].endswith("+00:00")
        assert fm["ingested_by"] == "pipeline-claude-export"


def test_write_entity():
    with tempfile.TemporaryDirectory() as tmp:
        writer = _make_writer(Path(tmp))
        entity = Entity(
            entity_type=EntityType.TOOL,
            name="Python",
            aliases=["python3", "cpython"],
        )
        path = writer.write_entity(entity, [])
        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "Python" in content
        assert "tool" in content
        fm = _frontmatter(path)
        assert fm["memory_type"] == "resource"
        assert fm["author"] == "pipeline-entity-extractor"
        assert fm["ingested_by"] == "pipeline-entity-extractor"


def test_write_concept():
    with tempfile.TemporaryDirectory() as tmp:
        writer = _make_writer(Path(tmp))
        concept = Concept(
            concept_type=ConceptType.DECISION,
            content="Use PostgreSQL for the main database",
            confidence=0.85,
        )
        path = writer.write_concept(concept, [])
        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "PostgreSQL" in content
        assert "decision" in content
        fm = _frontmatter(path)
        assert fm["memory_type"] == "semantic"
        assert fm["author"] == "pipeline-concept-extractor"
        assert fm["ingested_by"] == "pipeline-concept-extractor"


def test_write_all():
    with tempfile.TemporaryDirectory() as tmp:
        writer = _make_writer(Path(tmp))
        conv = Conversation(
            id="c1",
            title="Convo",
            messages=[
                ChatMessage(
                    conversation_id="c1", role=Role.USER,
                    content="test", platform=Platform.CLAUDE,
                ),
                ChatMessage(
                    conversation_id="c1", role=Role.ASSISTANT,
                    content="reply", platform=Platform.CLAUDE,
                ),
            ],
            platform=Platform.CLAUDE,
            created_at=datetime.now(timezone.utc),
        )
        entity = Entity(entity_type=EntityType.PERSON, name="Alice")
        concept = Concept(
            concept_type=ConceptType.TOPIC, content="Testing",
        )
        rel = Relationship(
            source_id="c1",
            target_id=entity.id,
            relationship_type=RelationshipType.MENTIONS,
        )
        written = writer.write_all([conv], [entity], [concept], [rel])
        assert len(written) == 3


def test_write_all_concept_inherits_source_author():
    with tempfile.TemporaryDirectory() as tmp:
        writer = _make_writer(Path(tmp))
        created_at = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        conv = Conversation(
            id="c-source",
            title="Source Convo",
            messages=[
                ChatMessage(
                    conversation_id="c-source",
                    role=Role.USER,
                    content="Remember this concept",
                    platform=Platform.COWORK,
                )
            ],
            platform=Platform.COWORK,
            author="cowork-claude",
            session_id="cowork-session",
            ingested_by="pipeline-cowork-ingester",
            created_at=created_at,
        )
        concept = Concept(
            concept_type=ConceptType.TOPIC,
            content="Source-derived concept",
            source_conversation_id="c-source",
        )

        writer.write_all([conv], [], [concept], [])
        fm = _frontmatter(Path(tmp) / "Concepts" / "Source-derived concept.md")

        assert fm["memory_type"] == "semantic"
        assert fm["author"] == "cowork-claude"
        assert fm["session_id"] == "cowork-session"
        assert fm["ingested_by"] == "pipeline-concept-extractor"


def test_action_item_concepts_are_procedural():
    with tempfile.TemporaryDirectory() as tmp:
        writer = _make_writer(Path(tmp))
        concept = Concept(
            concept_type=ConceptType.ACTION_ITEM,
            content="Follow up on the migration",
        )

        path = writer.write_concept(concept, [])
        fm = _frontmatter(path)

        assert fm["memory_type"] == "procedural"


def test_folder_structure():
    with tempfile.TemporaryDirectory() as tmp:
        writer = _make_writer(Path(tmp))
        writer.write_entity(
            Entity(entity_type=EntityType.PERSON, name="Bob"), []
        )
        writer.write_entity(
            Entity(entity_type=EntityType.PROJECT, name="MyProject"), []
        )
        writer.write_concept(
            Concept(concept_type=ConceptType.DECISION, content="Use Rust for the backend"), []
        )

        assert (Path(tmp) / "Contacts" / "Bob.md").exists()
        assert (Path(tmp) / "Projects" / "MyProject.md").exists()
        assert (Path(tmp) / "Decisions").exists()
