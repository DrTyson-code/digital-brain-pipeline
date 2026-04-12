"""Tests for MOCGenerator — Maps of Content generation for Obsidian vault.

Tests cover MOC generation, file writing, filtering by domain, and dataview queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import ChatMessage, Conversation, Role
from src.output.moc import MOCGenerator, DOMAIN_KEYWORDS


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def vault_path(tmp_path):
    """Create a temporary vault path."""
    return tmp_path / "vault"


@pytest.fixture
def generator(vault_path):
    """Create a MOCGenerator with temp vault."""
    return MOCGenerator(vault_path=vault_path)


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    conversations = []

    # Software engineering conversation
    conv1 = Conversation(
        id="conv-se-1",
        title="Python Web Development",
        messages=[
            ChatMessage(
                conversation_id="conv-se-1",
                role=Role.USER,
                content="How do I build a REST API with Python?",
                platform="claude",
            ),
            ChatMessage(
                conversation_id="conv-se-1",
                role=Role.ASSISTANT,
                content="You can use Flask or FastAPI",
                platform="claude",
            ),
        ],
        platform="claude",
        created_at=datetime.now(timezone.utc),
        topics=["python", "api", "web development"],
        summary="Discussion about building APIs with Python",
    )
    conversations.append(conv1)

    # Medical conversation
    conv2 = Conversation(
        id="conv-med-1",
        title="Anesthesia Techniques",
        messages=[
            ChatMessage(
                conversation_id="conv-med-1",
                role=Role.USER,
                content="What are common anesthesia methods?",
                platform="claude",
            ),
            ChatMessage(
                conversation_id="conv-med-1",
                role=Role.ASSISTANT,
                content="Regional and general anesthesia are common",
                platform="claude",
            ),
        ],
        platform="claude",
        created_at=datetime.now(timezone.utc),
        topics=["anesthesia", "medical"],
        summary="Overview of anesthesia techniques",
    )
    conversations.append(conv2)

    # Business conversation
    conv3 = Conversation(
        id="conv-biz-1",
        title="Startup Funding",
        messages=[
            ChatMessage(
                conversation_id="conv-biz-1",
                role=Role.USER,
                content="How do I get SBA loans for my startup?",
                platform="claude",
            ),
        ],
        platform="claude",
        created_at=datetime.now(timezone.utc),
        topics=["startup", "funding", "sba"],
        summary="Discussion about startup funding",
    )
    conversations.append(conv3)

    return conversations


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            entity_type=EntityType.TOOL,
            name="Python",
            aliases=["py", "python3"],
            source_conversations=["conv-se-1"],
        ),
        Entity(
            entity_type=EntityType.TOOL,
            name="Flask",
            aliases=["flask-framework"],
            source_conversations=["conv-se-1"],
        ),
        Entity(
            entity_type=EntityType.PERSON,
            name="Alice",
            aliases=["Dr. Alice Smith"],
            source_conversations=["conv-med-1"],
        ),
        Entity(
            entity_type=EntityType.ORGANIZATION,
            name="SBA",
            aliases=["Small Business Administration"],
            source_conversations=["conv-biz-1"],
        ),
        Entity(
            entity_type=EntityType.PROJECT,
            name="MyStartup",
            source_conversations=["conv-biz-1"],
        ),
    ]


@pytest.fixture
def sample_concepts():
    """Create sample concepts for testing."""
    return [
        Concept(
            concept_type=ConceptType.TOPIC,
            content="API Design Patterns",
            confidence=0.95,
            source_conversation_id="conv-se-1",
            tags=["python", "api"],
        ),
        Concept(
            concept_type=ConceptType.DECISION,
            content="Use FastAPI for better performance",
            confidence=0.9,
            source_conversation_id="conv-se-1",
            tags=["decision"],
        ),
        Concept(
            concept_type=ConceptType.QUESTION,
            content="Should I use sync or async handlers?",
            confidence=0.8,
            source_conversation_id="conv-se-1",
            tags=["question"],
        ),
        Concept(
            concept_type=ConceptType.ACTION_ITEM,
            content="Research FastAPI documentation",
            confidence=0.85,
            source_conversation_id="conv-se-1",
            tags=["action"],
        ),
        Concept(
            concept_type=ConceptType.INSIGHT,
            content="Regional anesthesia reduces recovery time",
            confidence=0.92,
            source_conversation_id="conv-med-1",
            tags=["anesthesia"],
        ),
    ]


# ============================================================================
# Basic MOC generation tests
# ============================================================================


def test_generator_initializes_with_vault_path(vault_path):
    """Test generator initialization."""
    gen = MOCGenerator(vault_path=vault_path)

    assert gen.vault_path == vault_path
    assert gen.moc_folder == vault_path / "MOC"


def test_generator_custom_tag_prefix(vault_path):
    """Test custom tag prefix."""
    gen = MOCGenerator(vault_path=vault_path, tag_prefix="my-brain")

    assert gen.tag_prefix == "my-brain"


def test_generate_dashboard(generator, sample_conversations, sample_entities, sample_concepts):
    """Test dashboard generation."""
    path = generator.generate_dashboard(sample_conversations, sample_entities, sample_concepts)

    assert path.exists()
    assert path.name == "00-Dashboard.md"

    content = path.read_text()
    assert "Digital Brain Dashboard" in content
    assert "Vault Statistics" in content
    assert "Domain Maps" in content
    assert "Entity Indexes" in content
    assert "Concept Indexes" in content


def test_dashboard_contains_dataview_queries(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that dashboard contains dataview queries."""
    path = generator.generate_dashboard(sample_conversations, sample_entities, sample_concepts)
    content = path.read_text()

    # Should contain dataview blocks
    assert "```dataview" in content
    assert "FROM #" in content
    assert "SORT" in content


def test_dashboard_frontmatter_valid(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that dashboard frontmatter is valid YAML."""
    path = generator.generate_dashboard(sample_conversations, sample_entities, sample_concepts)
    content = path.read_text()

    # Extract frontmatter
    lines = content.split("\n")
    fm_end = lines.index("---", 1)
    fm_str = "\n".join(lines[1:fm_end])

    frontmatter = yaml.safe_load(fm_str)
    assert frontmatter["title"] == "Dashboard"
    assert frontmatter["type"] == "moc"
    assert "ai-brain" in frontmatter["tags"][0]


# ============================================================================
# Domain MOC tests
# ============================================================================


def test_generate_domain_moc(generator, sample_conversations, sample_entities, sample_concepts):
    """Test generation of a domain-specific MOC."""
    path = generator.generate_domain_moc(
        "Software Engineering",
        DOMAIN_KEYWORDS["software_engineering"],
        sample_conversations,
        sample_entities,
        sample_concepts,
    )

    assert path.exists()
    assert "MOC-Software-Engineering" in path.name

    content = path.read_text()
    assert "Software Engineering" in content
    assert "Map of Content" in content


def test_domain_moc_filters_by_keywords(
    generator, sample_conversations, sample_entities, sample_concepts
):
    """Test that domain MOC filters conversations by keywords."""
    # Generate software engineering MOC
    se_path = generator.generate_domain_moc(
        "Software Engineering",
        DOMAIN_KEYWORDS["software_engineering"],
        sample_conversations,
        sample_entities,
        sample_concepts,
    )

    se_content = se_path.read_text()

    # Should reference Python (SE keyword)
    assert "python" in se_content.lower() or "Contains" in se_content

    # Generate medicine MOC
    med_path = generator.generate_domain_moc(
        "Medicine",
        DOMAIN_KEYWORDS["medicine"],
        sample_conversations,
        sample_entities,
        sample_concepts,
    )

    med_content = med_path.read_text()

    # Should reference anesthesia (medicine keyword)
    assert "anesthesia" in med_content.lower() or "Contains" in med_content


def test_domain_moc_contains_sections(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that domain MOC contains expected sections."""
    path = generator.generate_domain_moc(
        "Business",
        DOMAIN_KEYWORDS["business"],
        sample_conversations,
        sample_entities,
        sample_concepts,
    )

    content = path.read_text()

    # Should have various sections if content matches
    # (may be empty if no matching conversations)
    assert "Business" in content


def test_generate_all_domain_mocs(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that all domain MOCs are generated."""
    paths = generator.generate_all(sample_conversations, sample_entities, sample_concepts)

    # Should generate multiple MOCs
    assert len(paths) > 5

    # Check that domain MOCs exist
    filenames = {p.name for p in paths}
    assert "MOC-Software-Engineering.md" in filenames
    assert "MOC-Medicine.md" in filenames
    assert "MOC-Business.md" in filenames


# ============================================================================
# Entity index tests
# ============================================================================


def test_generate_entity_index_tool(generator, sample_entities):
    """Test tool entity index generation."""
    path = generator.generate_entity_index("tool", sample_entities)

    assert path.exists()
    assert "MOC-Tools-and-Tech" in path.name

    content = path.read_text()
    assert "Tools" in content
    assert "Index" in content
    assert "```dataview" in content


def test_generate_entity_index_person(generator, sample_entities):
    """Test person entity index generation."""
    path = generator.generate_entity_index("person", sample_entities)

    assert path.exists()
    assert "MOC-People" in path.name

    content = path.read_text()
    assert "People" in content


def test_generate_entity_index_organization(generator, sample_entities):
    """Test organization entity index generation."""
    path = generator.generate_entity_index("organization", sample_entities)

    assert path.exists()
    assert "MOC-Organizations" in path.name


def test_entity_index_lists_entities(generator, sample_entities):
    """Test that entity index lists entities."""
    path = generator.generate_entity_index("tool", sample_entities)
    content = path.read_text()

    # Tools: Python and Flask
    assert "Python" in content or "python" in content.lower()
    assert "Flask" in content or "flask" in content.lower()


def test_entity_index_empty(generator):
    """Test entity index with no entities."""
    path = generator.generate_entity_index("tool", [])

    assert path.exists()
    content = path.read_text()
    assert "Tools" in content


# ============================================================================
# Concept index tests
# ============================================================================


def test_generate_concept_index_decision(generator, sample_concepts):
    """Test decision concept index generation."""
    path = generator.generate_concept_index("decision", sample_concepts)

    assert path.exists()
    assert "MOC-Active-Decisions" in path.name

    content = path.read_text()
    assert "Decision" in content


def test_generate_concept_index_question(generator, sample_concepts):
    """Test question concept index generation."""
    path = generator.generate_concept_index("question", sample_concepts)

    assert path.exists()
    assert "MOC-Open-Questions" in path.name


def test_generate_concept_index_action_item(generator, sample_concepts):
    """Test action item concept index generation."""
    path = generator.generate_concept_index("action_item", sample_concepts)

    assert path.exists()
    assert "MOC-Action-Items" in path.name


def test_concept_index_lists_concepts(generator, sample_concepts):
    """Test that concept index lists concepts."""
    path = generator.generate_concept_index("decision", sample_concepts)
    content = path.read_text()

    # Should mention the decision concept
    assert "FastAPI" in content or "performance" in content.lower()


def test_concept_index_empty(generator):
    """Test concept index with no concepts."""
    path = generator.generate_concept_index("decision", [])

    assert path.exists()
    content = path.read_text()
    assert "Decision" in content or "Active" in content


# ============================================================================
# File writing tests
# ============================================================================


def test_moc_files_created_in_correct_folder(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that MOC files are created in MOC folder."""
    paths = generator.generate_all(sample_conversations, sample_entities, sample_concepts)

    for path in paths:
        assert path.parent == generator.moc_folder
        assert path.suffix == ".md"


def test_moc_frontmatter_structure(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that all MOCs have valid YAML frontmatter."""
    path = generator.generate_dashboard(sample_conversations, sample_entities, sample_concepts)

    content = path.read_text()
    lines = content.split("\n")

    # Should start with ---
    assert lines[0] == "---"

    # Should have closing --- after frontmatter
    assert "---" in lines[1:]

    # Parse frontmatter
    fm_end = lines.index("---", 1)
    fm_str = "\n".join(lines[1:fm_end])
    frontmatter = yaml.safe_load(fm_str)

    assert "title" in frontmatter
    assert "type" in frontmatter
    assert "tags" in frontmatter
    assert "date" in frontmatter


def test_moc_filename_sanitization(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that filenames are sanitized."""
    # Generate with a title that has special characters
    path = generator.generate_domain_moc(
        "Test/Domain:With<Special>Chars",
        ["test"],
        sample_conversations,
        sample_entities,
        sample_concepts,
    )

    # Should have valid filename (no special chars)
    assert "/" not in path.name
    assert ":" not in path.name
    assert "<" not in path.name
    assert ">" not in path.name


# ============================================================================
# Dataview query tests
# ============================================================================


def test_dashboard_contains_conversation_query(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that dashboard contains conversation dataview query."""
    path = generator.generate_dashboard(sample_conversations, sample_entities, sample_concepts)
    content = path.read_text()

    assert "FROM #ai-brain/conversation" in content
    assert "SORT date DESC" in content


def test_dashboard_contains_concept_query(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that dashboard contains concept dataview query."""
    path = generator.generate_dashboard(sample_conversations, sample_entities, sample_concepts)
    content = path.read_text()

    assert "FROM #ai-brain/concept" in content


def test_domain_moc_contains_entity_query(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that domain MOC contains entity dataview query."""
    path = generator.generate_domain_moc(
        "Software Engineering",
        DOMAIN_KEYWORDS["software_engineering"],
        sample_conversations,
        sample_entities,
        sample_concepts,
    )

    content = path.read_text()

    # Should have entity query if entities match domain
    if "Python" in content or "Flask" in content:
        assert "FROM #ai-brain/entity" in content or "entity" in content.lower()


def test_entity_index_contains_entity_query(generator, sample_entities):
    """Test that entity index contains dataview query."""
    path = generator.generate_entity_index("tool", sample_entities)
    content = path.read_text()

    assert "FROM #ai-brain/tool" in content
    assert "SORT" in content


def test_concept_index_contains_concept_query(generator, sample_concepts):
    """Test that concept index contains dataview query."""
    path = generator.generate_concept_index("decision", sample_concepts)
    content = path.read_text()

    assert "FROM #ai-brain/decision" in content


# ============================================================================
# Integration tests
# ============================================================================


def test_generate_all_creates_all_mocs(generator, sample_conversations, sample_entities, sample_concepts):
    """Test that generate_all creates all MOCs."""
    paths = generator.generate_all(sample_conversations, sample_entities, sample_concepts)

    filenames = {p.name for p in paths}

    # Should have dashboard
    assert "00-Dashboard.md" in filenames

    # Should have domain MOCs
    assert "MOC-Software-Engineering.md" in filenames
    assert "MOC-Medicine.md" in filenames
    assert "MOC-Machine-Learning.md" in filenames
    assert "MOC-Business.md" in filenames
    assert "MOC-Personal.md" in filenames
    assert "MOC-Writing.md" in filenames

    # Should have entity indexes
    assert "MOC-Tools-and-Tech.md" in filenames
    assert "MOC-People.md" in filenames
    assert "MOC-Organizations.md" in filenames
    assert "MOC-Projects.md" in filenames

    # Should have concept indexes
    assert "MOC-Active-Decisions.md" in filenames
    assert "MOC-Open-Questions.md" in filenames
    assert "MOC-Action-Items.md" in filenames


def test_generate_all_moc_folder_created(generator):
    """Test that MOC folder is created."""
    assert not generator.moc_folder.exists()

    generator.generate_all([], [], [])

    assert generator.moc_folder.exists()
    assert generator.moc_folder.is_dir()


def test_generate_all_with_empty_data(generator):
    """Test generate_all with empty data."""
    paths = generator.generate_all([], [], [])

    # Should still generate all MOCs, just empty
    assert len(paths) > 0

    for path in paths:
        assert path.exists()
        content = path.read_text()
        assert "---" in content  # Should have frontmatter


# ============================================================================
# Helper function tests
# ============================================================================


def test_sanitize_filename():
    """Test filename sanitization."""
    from src.output.moc import _sanitize_filename

    assert _sanitize_filename("normal-file") == "normal-file"
    assert _sanitize_filename("file:with:colons") == "filewithcolons"
    assert _sanitize_filename("file<with>angles") == "filewithangles"
    assert _sanitize_filename("very" + "x" * 300 + "long").startswith("very")
    assert len(_sanitize_filename("very" + "x" * 300 + "long")) <= 200


def test_matches_keywords():
    """Test keyword matching."""
    from src.output.moc import _matches_keywords

    assert _matches_keywords("Python programming", ["python"])
    assert _matches_keywords("python programming", ["PYTHON"])
    assert not _matches_keywords("JavaScript code", ["python"])
    assert _matches_keywords(["python", "api"], ["python"])


def test_format_date():
    """Test date formatting."""
    from src.output.moc import _format_date

    dt = datetime(2026, 4, 12, tzinfo=timezone.utc)
    assert _format_date(dt, "%Y-%m-%d") == "2026-04-12"
    assert _format_date(None, "%Y-%m-%d") == ""
