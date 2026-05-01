"""Tests for ontology data models."""

from datetime import datetime, timezone

from src.models.base import OntologyObject, Platform
from src.models.authorship import AGENT_IDS, is_valid_agent_id
from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import ChatMessage, Conversation, Role
from src.models.relationship import Relationship, RelationshipType


def test_ontology_object_has_id():
    obj = OntologyObject()
    assert obj.id
    assert len(obj.id) == 12


def test_platform_enum():
    assert Platform.CLAUDE == "claude"
    assert Platform.CHATGPT == "chatgpt"
    assert Platform.GEMINI == "gemini"


def test_agent_id_taxonomy():
    expected = {
        "chat-claude",
        "cowork-claude",
        "codex-cli",
        "chat-chatgpt",
        "chat-gemini",
        "william",
        "pipeline-synth-cluster",
        "pipeline-entity-extractor",
        "pipeline-moc-generator",
        "pipeline-claude-export",
        "pipeline-cowork-ingester",
        "pipeline-codex-ingester",
        "pipeline-concept-extractor",
    }
    assert expected <= AGENT_IDS
    assert is_valid_agent_id("chat-claude")
    assert is_valid_agent_id("pipeline-concept-extractor")
    assert not is_valid_agent_id("unknown-agent")


def test_chat_message():
    msg = ChatMessage(
        conversation_id="conv1",
        role=Role.USER,
        content="Hello, world!",
        platform=Platform.CLAUDE,
    )
    assert msg.is_user
    assert not msg.is_assistant
    assert msg.word_count == 2


def test_conversation_properties():
    msgs = [
        ChatMessage(
            conversation_id="conv1",
            role=Role.USER,
            content="Hi there",
            platform=Platform.CLAUDE,
            tokens=10,
        ),
        ChatMessage(
            conversation_id="conv1",
            role=Role.ASSISTANT,
            content="Hello! How can I help?",
            platform=Platform.CLAUDE,
            tokens=20,
        ),
    ]
    conv = Conversation(
        id="conv1",
        title="Test Conversation",
        messages=msgs,
        platform=Platform.CLAUDE,
        created_at=datetime.now(timezone.utc),
    )
    assert conv.message_count == 2
    assert len(conv.user_messages) == 1
    assert len(conv.assistant_messages) == 1
    assert conv.total_tokens == 30


def test_conversation_duration():
    t1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
    msgs = [
        ChatMessage(
            conversation_id="c", role=Role.USER, content="a",
            platform=Platform.CLAUDE, timestamp=t1,
        ),
        ChatMessage(
            conversation_id="c", role=Role.ASSISTANT, content="b",
            platform=Platform.CLAUDE, timestamp=t2,
        ),
    ]
    conv = Conversation(
        id="c", messages=msgs, platform=Platform.CLAUDE, created_at=t1,
    )
    assert conv.duration == 1800.0  # 30 minutes in seconds


def test_entity_merge():
    e1 = Entity(
        entity_type=EntityType.TOOL,
        name="Python",
        aliases=["python3"],
        first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_conversations=["conv1"],
    )
    e2 = Entity(
        entity_type=EntityType.TOOL,
        name="python",
        aliases=["cpython"],
        first_seen=datetime(2023, 6, 1, tzinfo=timezone.utc),
        last_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
        source_conversations=["conv2"],
    )
    e1.merge(e2)
    assert "cpython" in e1.aliases
    assert "python" in e1.aliases
    assert e1.first_seen == datetime(2023, 6, 1, tzinfo=timezone.utc)
    assert e1.last_seen == datetime(2024, 6, 1, tzinfo=timezone.utc)
    assert "conv2" in e1.source_conversations


def test_entity_matches():
    entity = Entity(
        entity_type=EntityType.PERSON,
        name="John Doe",
        aliases=["JD", "johnd"],
    )
    assert entity.matches("John Doe")
    assert entity.matches("john doe")
    assert entity.matches("JD")
    assert not entity.matches("Jane")


def test_concept_properties():
    concept = Concept(
        concept_type=ConceptType.ACTION_ITEM,
        content="Implement the new API endpoint",
        confidence=0.9,
    )
    assert concept.is_actionable
    assert concept.is_high_confidence

    topic = Concept(
        concept_type=ConceptType.TOPIC,
        content="Machine Learning",
        confidence=0.5,
    )
    assert not topic.is_actionable
    assert not topic.is_high_confidence


def test_relationship_directed():
    rel = Relationship(
        source_id="a",
        target_id="b",
        relationship_type=RelationshipType.MENTIONS,
    )
    assert rel.is_directed
    assert rel.involves("a")
    assert rel.involves("b")
    assert not rel.involves("c")

    rel2 = Relationship(
        source_id="a",
        target_id="b",
        relationship_type=RelationshipType.RELATES_TO,
    )
    assert not rel2.is_directed
