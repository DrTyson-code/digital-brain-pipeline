"""Tests for the Obsidian output writer."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.models.base import Platform
from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import ChatMessage, Conversation, Role
from src.models.relationship import Relationship, RelationshipType
from src.output.obsidian import ObsidianWriter


def _make_writer(tmp_dir: Path) -> ObsidianWriter:
    return ObsidianWriter(vault_path=tmp_dir)


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
