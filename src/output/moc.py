"""Generate Maps of Content (MOCs) for organizing the Obsidian vault by domain.

Maps of Content are curated index notes that link to related notes in a topic area.
They use Dataview queries to auto-update as new notes are added.

MOCs generated:
- 00-Dashboard.md — master index linking to all other MOCs with overall stats
- MOC-Software-Engineering.md — programming, development, architecture topics
- MOC-Medicine.md — anesthesiology, pharmacology, physiology, medical topics
- MOC-Machine-Learning.md — ML/AI, models, training, embeddings topics
- MOC-Business.md — business, finance, entrepreneurship, strategy topics
- MOC-Personal.md — personal development, fitness, productivity, habits topics
- MOC-Writing.md — writing, content, creative, publishing topics
- MOC-Tools-and-Tech.md — index of all tool entities
- MOC-Active-Decisions.md — all high-confidence decisions
- MOC-Open-Questions.md — all extracted questions
- MOC-Action-Items.md — all action items (todo-like)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import Conversation
from src.process.review_queue import CurationResult
from src.utils.io import atomic_write

logger = logging.getLogger(__name__)

# Domain keyword mappings for filtering conversations and concepts by topic
DOMAIN_KEYWORDS = {
    "software_engineering": [
        "python",
        "javascript",
        "typescript",
        "rust",
        "react",
        "docker",
        "kubernetes",
        "api",
        "database",
        "git",
        "code",
        "programming",
        "software",
        "web",
        "frontend",
        "backend",
    ],
    "medicine": [
        "anesthesia",
        "anesthesiology",
        "medical",
        "patient",
        "surgery",
        "drug",
        "pharmacology",
        "physiology",
        "clinical",
        "hospital",
    ],
    "machine_learning": [
        "ml",
        "ai",
        "model",
        "neural",
        "transformer",
        "training",
        "embedding",
        "llm",
        "deep learning",
        "nlp",
        "fine-tune",
    ],
    "business": [
        "revenue",
        "startup",
        "business",
        "finance",
        "investment",
        "entrepreneurship",
        "marketing",
        "strategy",
        "loan",
        "sba",
    ],
    "personal": [
        "fitness",
        "health",
        "workout",
        "meditation",
        "productivity",
        "habit",
        "goal",
        "journal",
    ],
    "writing": [
        "writing",
        "blog",
        "article",
        "essay",
        "creative",
        "content",
        "publish",
        "book",
    ],
}


def _format_date(dt: datetime | None, fmt: str = "%Y-%m-%d") -> str:
    """Format a datetime object as a string."""
    if dt is None:
        return ""
    return dt.strftime(fmt)


def _matches_keywords(
    text: str | list[str], keywords: list[str]
) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    if isinstance(text, list):
        text = " ".join(text)
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def _sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    for ch in r'<>:"/\|?*':
        name = name.replace(ch, "")
    return name.strip()[:200]


class MOCGenerator:
    """Generate Maps of Content for organizing the Obsidian vault by domain."""

    def __init__(
        self,
        vault_path: Path,
        tag_prefix: str = "ai-brain",
        date_format: str = "%Y-%m-%d",
    ) -> None:
        """Initialize the MOC generator.

        Args:
            vault_path: Path to the Obsidian vault root
            tag_prefix: Prefix for all generated tags (default: "ai-brain")
            date_format: Format string for dates (default: "%Y-%m-%d")
        """
        self.vault_path = Path(vault_path).expanduser()
        self.tag_prefix = tag_prefix
        self.date_format = date_format
        self.moc_folder = self.vault_path / "MOC"

    def generate_all(
        self,
        conversations: list[Conversation],
        entities: list[Entity],
        concepts: list[Concept],
        relationships: Any = None,
        curation: Optional[CurationResult] = None,
    ) -> list[Path]:
        """Generate all MOCs. Returns list of created file paths."""
        self.moc_folder.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        logger.info("Generating MOCs in %s", self.moc_folder)

        # Generate dashboard first (links to all others)
        written.append(self.generate_dashboard(conversations, entities, concepts))

        # Generate domain-specific MOCs
        written.append(
            self.generate_domain_moc(
                "Software Engineering",
                DOMAIN_KEYWORDS["software_engineering"],
                conversations,
                entities,
                concepts,
            )
        )
        written.append(
            self.generate_domain_moc(
                "Medicine",
                DOMAIN_KEYWORDS["medicine"],
                conversations,
                entities,
                concepts,
            )
        )
        written.append(
            self.generate_domain_moc(
                "Machine Learning",
                DOMAIN_KEYWORDS["machine_learning"],
                conversations,
                entities,
                concepts,
            )
        )
        written.append(
            self.generate_domain_moc(
                "Business",
                DOMAIN_KEYWORDS["business"],
                conversations,
                entities,
                concepts,
            )
        )
        written.append(
            self.generate_domain_moc(
                "Personal",
                DOMAIN_KEYWORDS["personal"],
                conversations,
                entities,
                concepts,
            )
        )
        written.append(
            self.generate_domain_moc(
                "Writing",
                DOMAIN_KEYWORDS["writing"],
                conversations,
                entities,
                concepts,
            )
        )

        # Generate entity indexes
        written.append(self.generate_entity_index("tool", entities))
        written.append(self.generate_entity_index("person", entities))
        written.append(self.generate_entity_index("organization", entities))
        written.append(self.generate_entity_index("project", entities))

        # Generate concept indexes
        written.append(self.generate_concept_index("decision", concepts))
        written.append(self.generate_concept_index("question", concepts))
        written.append(self.generate_concept_index("action_item", concepts))

        # Generate review queue dashboard (always, even if curation is empty)
        written.append(self.generate_review_queue(curation))

        logger.info("Generated %d MOC files", len(written))
        return written

    def generate_dashboard(
        self,
        conversations: list[Conversation],
        entities: list[Entity],
        concepts: list[Concept],
    ) -> Path:
        """Generate the master 00-Dashboard.md MOC."""
        today = datetime.now()
        frontmatter = {
            "title": "Dashboard",
            "type": "moc",
            "tags": [f"{self.tag_prefix}/moc"],
            "date": _format_date(today, self.date_format),
        }

        lines: list[str] = []
        lines.append("# Digital Brain Dashboard\n")
        lines.append(
            "Master index to all Maps of Content in your AI-Brain vault.\n"
        )

        # Overall statistics
        lines.append("## Vault Statistics\n")
        lines.append("```dataview")
        lines.append(
            f'TABLE length(rows) as "{self.tag_prefix} Objects"'
        )
        lines.append(f"FROM #{self.tag_prefix}")
        lines.append("LIMIT 1")
        lines.append("```\n")

        lines.append(f"- **Conversations**: {len(conversations)}")
        lines.append(f"- **Entities**: {len(entities)}")
        lines.append(f"- **Concepts**: {len(concepts)}")
        lines.append(f"- **Last Updated**: {_format_date(today, self.date_format)}\n")

        # Domain MOCs
        lines.append("## Domain Maps\n")
        lines.append("- [[MOC-Software-Engineering]] — Programming and development")
        lines.append("- [[MOC-Medicine]] — Medical and healthcare topics")
        lines.append("- [[MOC-Machine-Learning]] — AI/ML and deep learning")
        lines.append("- [[MOC-Business]] — Business and entrepreneurship")
        lines.append("- [[MOC-Personal]] — Personal development and productivity")
        lines.append("- [[MOC-Writing]] — Writing and creative content\n")

        # Entity Indexes
        lines.append("## Entity Indexes\n")
        lines.append("- [[MOC-Tools-and-Tech]] — All tools and technologies")
        lines.append("- [[MOC-People]] — All people mentioned")
        lines.append("- [[MOC-Organizations]] — All organizations")
        lines.append("- [[MOC-Projects]] — All projects\n")

        # Concept Indexes
        lines.append("## Concept Indexes\n")
        lines.append("- [[MOC-Active-Decisions]] — High-confidence decisions")
        lines.append("- [[MOC-Open-Questions]] — Extracted questions")
        lines.append("- [[MOC-Action-Items]] — Action items and todos\n")

        # Curation
        lines.append("## Knowledge Quality\n")
        lines.append("- [[MOC-Review-Queue]] — Items needing human review\n")

        # Recent conversations (dataview query)
        lines.append("## Recent Conversations\n")
        lines.append("```dataview")
        lines.append(
            "TABLE date, platform, message_count"
        )
        lines.append(f"FROM #{self.tag_prefix}/conversation")
        lines.append("SORT date DESC")
        lines.append("LIMIT 10")
        lines.append("```\n")

        # High-confidence insights
        lines.append("## High-Confidence Insights\n")
        lines.append("```dataview")
        lines.append("TABLE confidence, type")
        lines.append(f"FROM #{self.tag_prefix}/concept")
        lines.append("WHERE confidence >= 0.8")
        lines.append("SORT confidence DESC")
        lines.append("LIMIT 15")
        lines.append("```\n")

        return self._write_moc(
            "00-Dashboard", frontmatter, "\n".join(lines)
        )

    def generate_domain_moc(
        self,
        domain: str,
        keywords: list[str],
        conversations: list[Conversation],
        entities: list[Entity],
        concepts: list[Concept],
    ) -> Path:
        """Generate a domain-specific MOC."""
        today = datetime.now()
        filename = f"MOC-{domain.replace(' ', '-')}"

        frontmatter = {
            "title": f"{domain} Map",
            "type": "moc",
            "tags": [f"{self.tag_prefix}/moc", f"{self.tag_prefix}/{domain.lower().replace(' ', '_')}"],
            "date": _format_date(today, self.date_format),
        }

        lines: list[str] = []
        lines.append(f"# {domain} Map of Content\n")
        lines.append(
            f"Curated index of {domain.lower()} related conversations, "
            "entities, and concepts.\n"
        )

        # Filter conversations by keywords
        domain_conversations = [
            c for c in conversations
            if _matches_keywords(
                (c.title or "") + " " + " ".join(c.topics), keywords
            )
        ]

        # Filter concepts by keywords
        domain_concepts = [
            c for c in concepts
            if _matches_keywords(c.content + " " + " ".join(c.tags), keywords)
        ]

        # Filter entities by keywords
        domain_entities = [
            e for e in entities
            if _matches_keywords(
                e.name + " " + " ".join(e.aliases), keywords
            )
        ]

        # Conversations section
        if domain_conversations:
            lines.append("## Conversations\n")
            lines.append("```dataview")
            lines.append("TABLE date, platform, message_count, topics")
            lines.append(f"FROM #{self.tag_prefix}/conversation")
            lines.append(
                f'WHERE contains(topics, "{keywords[0]}") OR contains(summary, "{keywords[0]}")'
            )
            lines.append("SORT date DESC")
            lines.append("```\n")

        # Concepts section
        if domain_concepts:
            lines.append("## Key Concepts & Insights\n")
            lines.append("```dataview")
            lines.append("TABLE type, confidence")
            lines.append(f"FROM #{self.tag_prefix}/concept")
            lines.append(
                f'WHERE contains(content, "{keywords[0]}") OR contains(tags, "{keywords[0]}")'
            )
            lines.append("SORT confidence DESC")
            lines.append("```\n")

        # Related entities
        if domain_entities:
            lines.append("## Related Entities\n")
            lines.append("```dataview")
            lines.append("TABLE type, first_seen, last_seen")
            lines.append(f"FROM #{self.tag_prefix}/entity")
            lines.append(
                f'WHERE contains(name, "{keywords[0]}") OR contains(aliases, "{keywords[0]}")'
            )
            lines.append("```\n")

        # Decisions in this domain
        domain_decisions = [
            c for c in domain_concepts
            if c.concept_type == ConceptType.DECISION
        ]
        if domain_decisions:
            lines.append("## Decisions\n")
            lines.append("```dataview")
            lines.append("TABLE confidence, source")
            lines.append(f"FROM #{self.tag_prefix}/decision")
            lines.append("WHERE confidence >= 0.7")
            lines.append("SORT confidence DESC")
            lines.append("```\n")

        # Questions in this domain
        domain_questions = [
            c for c in domain_concepts
            if c.concept_type == ConceptType.QUESTION
        ]
        if domain_questions:
            lines.append("## Open Questions\n")
            lines.append("```dataview")
            lines.append("TABLE source")
            lines.append(f"FROM #{self.tag_prefix}/question")
            lines.append("SORT date DESC")
            lines.append("```\n")

        return self._write_moc(filename, frontmatter, "\n".join(lines))

    def generate_entity_index(
        self, entity_type: str, entities: list[Entity]
    ) -> Path:
        """Generate an index for a specific entity type."""
        today = datetime.now()
        type_entities = [
            e for e in entities
            if e.entity_type.value == entity_type
        ]

        # Format title
        title_map = {
            "person": "People",
            "organization": "Organizations",
            "project": "Projects",
            "tool": "Tools and Tech",
            "location": "Locations",
        }
        title = title_map.get(entity_type, entity_type.title())
        filename = f"MOC-{title.replace(' ', '-')}"

        frontmatter = {
            "title": f"{title} Index",
            "type": "moc",
            "tags": [f"{self.tag_prefix}/moc", f"{self.tag_prefix}/{entity_type}"],
            "date": _format_date(today, self.date_format),
        }

        lines: list[str] = []
        lines.append(f"# {title} Index\n")
        lines.append(f"Complete index of all {entity_type.lower()} entities.\n")

        lines.append("```dataview")
        lines.append("TABLE type, first_seen, last_seen, aliases")
        lines.append(f"FROM #{self.tag_prefix}/{entity_type}")
        lines.append("SORT name ASC")
        lines.append("```\n")

        if type_entities:
            lines.append(f"## All {title}\n")
            for entity in sorted(type_entities, key=lambda e: e.name):
                lines.append(f"- [[{entity.name}]]")
            lines.append("")

        return self._write_moc(filename, frontmatter, "\n".join(lines))

    def generate_concept_index(
        self, concept_type: str, concepts: list[Concept]
    ) -> Path:
        """Generate an index for a specific concept type."""
        today = datetime.now()
        type_concepts = [
            c for c in concepts
            if c.concept_type.value == concept_type
        ]

        # Format title and filename
        title_map = {
            "decision": "Active Decisions",
            "question": "Open Questions",
            "action_item": "Action Items",
            "insight": "Insights",
            "topic": "Topics",
        }
        title = title_map.get(concept_type, concept_type.replace("_", " ").title())
        filename = f"MOC-{title.replace(' ', '-')}"

        frontmatter = {
            "title": title,
            "type": "moc",
            "tags": [f"{self.tag_prefix}/moc", f"{self.tag_prefix}/{concept_type}"],
            "date": _format_date(today, self.date_format),
        }

        lines: list[str] = []
        lines.append(f"# {title}\n")

        if concept_type == "decision":
            lines.append(
                "All extracted decisions with high confidence. "
                "Sorted by confidence and date.\n"
            )
        elif concept_type == "question":
            lines.append(
                "Open questions extracted from conversations. "
                "Use these as prompts for deeper exploration.\n"
            )
        elif concept_type == "action_item":
            lines.append(
                "Action items and todos extracted from conversations. "
                "Organize your next steps here.\n"
            )
        else:
            lines.append(
                f"All {concept_type.lower()} concepts from conversations.\n"
            )

        # Dataview query
        lines.append("```dataview")
        if concept_type == "decision":
            lines.append("TABLE confidence, source, tags")
            lines.append(f"FROM #{self.tag_prefix}/{concept_type}")
            lines.append("SORT confidence DESC, date DESC")
        elif concept_type == "action_item":
            lines.append("TABLE source, confidence, tags")
            lines.append(f"FROM #{self.tag_prefix}/{concept_type}")
            lines.append("SORT date DESC")
        else:
            lines.append("TABLE source, confidence")
            lines.append(f"FROM #{self.tag_prefix}/{concept_type}")
            lines.append("SORT date DESC")
        lines.append("LIMIT 100")
        lines.append("```\n")

        # Manual list if few items
        if type_concepts and len(type_concepts) <= 20:
            lines.append("## Items\n")
            for concept in sorted(
                type_concepts, key=lambda c: c.created_at, reverse=True
            ):
                confidence_str = (
                    f" ({concept.confidence:.0%})" if concept.confidence < 1.0 else ""
                )
                lines.append(f"- {concept.content[:80]}{confidence_str}")
            lines.append("")

        return self._write_moc(filename, frontmatter, "\n".join(lines))

    def generate_review_queue(
        self,
        curation: Optional[CurationResult] = None,
    ) -> Path:
        """Generate MOC-Review-Queue.md with Dataview queries for human review."""
        today = datetime.now()
        frontmatter = {
            "title": "Review Queue",
            "type": "moc",
            "tags": [f"{self.tag_prefix}/moc", f"{self.tag_prefix}/review"],
            "date": _format_date(today, self.date_format),
        }

        lines: list[str] = []
        lines.append("# Review Queue\n")
        lines.append(
            "Items flagged by the curation pipeline that need human attention.\n"
        )

        # Summary stats from the last run
        if curation:
            review_count = len(curation.review_ids)
            contradiction_count = len(curation.contradictions)
            merge_count = len(curation.merge_map)
            lines.append("## Last Run Summary\n")
            lines.append(f"- **Items needing review**: {review_count}")
            lines.append(f"- **Contradictions detected**: {contradiction_count}")
            lines.append(f"- **Entity merges performed**: {merge_count}")
            if curation.corrections:
                lines.append(
                    f"- **User corrections loaded**: {len(curation.corrections)}"
                )
            lines.append("")

        # Dataview: all items needing review
        lines.append("## All Items Needing Review\n")
        lines.append("```dataview")
        lines.append("TABLE type, confidence, needs_review")
        lines.append(f"FROM #{self.tag_prefix}")
        lines.append("WHERE needs_review = true")
        lines.append("SORT confidence ASC")
        lines.append("LIMIT 50")
        lines.append("```\n")

        # Dataview: low-confidence extractions
        lines.append("## Low-Confidence Extractions\n")
        lines.append("```dataview")
        lines.append("TABLE type, confidence, source")
        lines.append(f"FROM #{self.tag_prefix}/concept")
        lines.append("WHERE confidence < 0.6")
        lines.append("SORT confidence ASC")
        lines.append("LIMIT 30")
        lines.append("```\n")

        # Dataview: superseded decisions
        lines.append("## Recently Superseded Decisions\n")
        lines.append("```dataview")
        lines.append("TABLE status, valid_from, superseded_by")
        lines.append(f"FROM #{self.tag_prefix}/decision")
        lines.append('WHERE status = "superseded"')
        lines.append("SORT valid_from DESC")
        lines.append("LIMIT 20")
        lines.append("```\n")

        # Dataview: contradictions
        lines.append("## Detected Contradictions\n")
        lines.append("```dataview")
        lines.append("TABLE type, confidence, needs_review")
        lines.append(f"FROM #{self.tag_prefix}/concept")
        lines.append('WHERE needs_review = true AND type = "decision"')
        lines.append("SORT date DESC")
        lines.append("LIMIT 20")
        lines.append("```\n")

        # How to submit corrections
        lines.append("## How to Submit Corrections\n")
        lines.append(
            "Create a `.yaml` file in `_corrections/` folder with one of these formats:\n"
        )
        lines.append("```yaml")
        lines.append("# Rename an entity")
        lines.append("original_id: <pipeline-object-id>")
        lines.append("correction_type: rename")
        lines.append("corrected_value: New Canonical Name")
        lines.append("---")
        lines.append("# Mark a decision as completed")
        lines.append("original_id: <pipeline-object-id>")
        lines.append("correction_type: status_change")
        lines.append("corrected_value: completed")
        lines.append("```\n")

        return self._write_moc("MOC-Review-Queue", frontmatter, "\n".join(lines))

    def _write_moc(
        self,
        filename: str,
        frontmatter: dict[str, Any],
        body: str,
    ) -> Path:
        """Write a MOC markdown file with YAML frontmatter."""
        safe_filename = _sanitize_filename(filename) + ".md"
        file_path = self.moc_folder / safe_filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Build frontmatter YAML
        fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        content = f"---\n{fm_str}---\n\n{body}"

        atomic_write(file_path, content, root=self.vault_path)
        logger.debug("Wrote MOC: %s", file_path)
        return file_path
