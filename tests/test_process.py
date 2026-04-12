"""Tests for processing pipeline stages."""

from datetime import datetime, timezone

from src.models.base import Platform
from src.models.concept import ConceptType
from src.models.entity import EntityType
from src.models.message import ChatMessage, Conversation, Role
from src.process.classifier import ConversationClassifier
from src.process.extractor import EntityConceptExtractor
from src.process.linker import ObjectLinker
from src.process.enricher import Enricher


def _make_conversation(messages_text: list[str], conv_id: str = "test-conv") -> Conversation:
    """Helper to create a conversation with alternating user/assistant messages."""
    msgs = []
    for i, text in enumerate(messages_text):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        msgs.append(
            ChatMessage(
                conversation_id=conv_id,
                role=role,
                content=text,
                platform=Platform.CLAUDE,
            )
        )
    return Conversation(
        id=conv_id,
        messages=msgs,
        platform=Platform.CLAUDE,
        created_at=datetime.now(timezone.utc),
    )


def test_classifier_detects_programming():
    conv = _make_conversation([
        "How do I debug this Python function?",
        "You can use the debugger or add print statements to your code.",
    ])
    classifier = ConversationClassifier()
    result = classifier.classify(conv)
    assert "programming" in result.topics


def test_classifier_empty_conversation():
    conv = _make_conversation([])
    classifier = ConversationClassifier()
    result = classifier.classify(conv)
    assert result.topics == []


def test_extractor_finds_tools():
    conv = _make_conversation([
        "I'm building a project with Python and Docker",
        "Great choices! Python and Docker work well together.",
    ])
    extractor = EntityConceptExtractor()
    result = extractor.extract(conv)
    tool_names = {e.name for e in result.entities if e.entity_type == EntityType.TOOL}
    assert "python" in tool_names
    assert "docker" in tool_names


def test_extractor_finds_action_items():
    conv = _make_conversation([
        "TODO: implement the authentication middleware",
        "I'll help you with that.",
    ])
    extractor = EntityConceptExtractor()
    result = extractor.extract(conv)
    action_items = [c for c in result.concepts if c.concept_type == ConceptType.ACTION_ITEM]
    assert len(action_items) >= 1


def test_extractor_finds_decisions():
    conv = _make_conversation([
        "We decided to use PostgreSQL as the primary relational database for this project.",
        "Good decision, PostgreSQL is reliable.",
    ])
    extractor = EntityConceptExtractor()
    result = extractor.extract(conv)
    decisions = [c for c in result.concepts if c.concept_type == ConceptType.DECISION]
    assert len(decisions) >= 1


def test_linker_creates_relationships():
    conv = _make_conversation([
        "Let's use Python and Docker together",
        "Sure, they work well together.",
    ])
    extractor = EntityConceptExtractor()
    extraction = extractor.extract(conv)

    linker = ObjectLinker()
    relationships = linker.link([conv], [extraction])
    assert len(relationships) > 0


def test_enricher_deduplicates():
    conv1 = _make_conversation(["I love Python"], conv_id="c1")
    conv2 = _make_conversation(["Python is great"], conv_id="c2")

    extractor = EntityConceptExtractor()
    ext1 = extractor.extract(conv1)
    ext2 = extractor.extract(conv2)

    # Both should find "python"
    python_entities_before = [
        e for ext in [ext1, ext2] for e in ext.entities if e.name == "python"
    ]
    assert len(python_entities_before) == 2

    enricher = Enricher(deduplicate=True)
    enricher.enrich([ext1, ext2], [])

    python_entities_after = [
        e for ext in [ext1, ext2] for e in ext.entities if e.name == "python"
    ]
    assert len(python_entities_after) == 1
