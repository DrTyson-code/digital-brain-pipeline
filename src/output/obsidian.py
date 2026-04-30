"""Generate Obsidian-compatible markdown notes from the knowledge graph.

Produces notes with:
- YAML frontmatter (tags, aliases, dates, Dataview fields)
- Wikilinks [[like this]] throughout
- Organized into folders matching the vault structure
- Backlinks section at bottom of each note
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import Conversation
from src.models.relationship import Relationship
from src.process.review_queue import CurationResult
from src.utils.io import atomic_write

logger = logging.getLogger(__name__)

# Map entity/concept types to vault folders
FOLDER_MAP: dict[str, str] = {
    "person": "Contacts",
    "organization": "Contacts",
    "project": "Projects",
    "tool": "Tools",
    "location": "Locations",
    "topic": "Concepts",
    "decision": "Decisions",
    "action_item": "Action-Items",
    "insight": "Concepts",
    "question": "Concepts",
    "conversation": "AI-Conversations",
}


def _sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    # Remove or replace characters that are problematic in filenames
    for ch in r'<>:"/\|?*':
        name = name.replace(ch, "")
    return name.strip()[:200]


def _format_date(dt: datetime | None, fmt: str = "%Y-%m-%d") -> str:
    if dt is None:
        return ""
    return dt.strftime(fmt)


class ObsidianWriter:
    """Write knowledge graph objects as Obsidian markdown notes."""

    def __init__(
        self,
        vault_path: Path,
        tag_prefix: str = "ai-brain",
        date_format: str = "%Y-%m-%d",
        dataview_fields: bool = True,
        backlinks_section: bool = True,
    ) -> None:
        self.vault_path = Path(vault_path).expanduser()
        self.tag_prefix = tag_prefix
        self.date_format = date_format
        self.dataview_fields = dataview_fields
        self.backlinks_section = backlinks_section
        # Set during write_all; gives individual write_* methods access to
        # curation metadata without changing their public signatures.
        self._curation: Optional[CurationResult] = None

    def write_all(
        self,
        conversations: list[Conversation],
        entities: list[Entity],
        concepts: list[Concept],
        relationships: list[Relationship],
        curation: Optional[CurationResult] = None,
    ) -> list[Path]:
        """Write all objects to the vault. Returns list of created file paths."""
        self._curation = curation
        written: list[Path] = []

        for conv in conversations:
            path = self.write_conversation(conv, relationships)
            if path:
                written.append(path)

        for entity in entities:
            path = self.write_entity(entity, relationships)
            if path:
                written.append(path)

        for concept in concepts:
            path = self.write_concept(concept, relationships)
            if path:
                written.append(path)

        self._curation = None  # clear after write
        logger.info("Wrote %d notes to %s", len(written), self.vault_path)
        return written

    def write_conversation(
        self, conv: Conversation, relationships: list[Relationship]
    ) -> Path | None:
        """Write a conversation as an Obsidian note."""
        title = conv.title or f"Conversation {conv.id[:8]}"
        folder = FOLDER_MAP["conversation"]

        frontmatter = {
            "title": title,
            "date": _format_date(conv.created_at, self.date_format),
            "platform": conv.platform.value,
            "tags": [f"{self.tag_prefix}/conversation", f"{self.tag_prefix}/{conv.platform.value}"],
            "message_count": conv.message_count,
        }
        if conv.topics:
            frontmatter["topics"] = conv.topics
        if conv.summary:
            frontmatter["summary"] = conv.summary
        if self._curation and conv.id in self._curation.source_weights:
            frontmatter["source_weight"] = round(
                self._curation.source_weights[conv.id], 3
            )

        lines: list[str] = []
        lines.append(f"# {title}\n")

        if conv.summary:
            lines.append(f"> {conv.summary}\n")

        if self.dataview_fields:
            lines.append(f"platform:: {conv.platform.value}")
            lines.append(f"date:: {_format_date(conv.created_at, self.date_format)}")
            lines.append(f"messages:: {conv.message_count}")
            if conv.topics:
                topic_links = ", ".join(f"[[{t}]]" for t in conv.topics)
                lines.append(f"topics:: {topic_links}")
            lines.append("")

        # Write messages
        lines.append("## Messages\n")
        for msg in conv.messages:
            role_label = msg.role.value.capitalize()
            lines.append(f"### {role_label}")
            lines.append(f"{msg.content}\n")

        # Backlinks
        if self.backlinks_section:
            related = self._get_related_ids(conv.id, relationships)
            if related:
                lines.append("## Related\n")
                for rid in related:
                    lines.append(f"- [[{rid}]]")
                lines.append("")

        return self._write_note(folder, title, frontmatter, "\n".join(lines))

    def write_entity(
        self, entity: Entity, relationships: list[Relationship]
    ) -> Path | None:
        """Write an entity as an Obsidian note."""
        folder = FOLDER_MAP.get(entity.entity_type.value, "Entities")

        frontmatter: dict = {
            "title": entity.name,
            "type": entity.entity_type.value,
            "tags": [
                f"{self.tag_prefix}/entity",
                f"{self.tag_prefix}/{entity.entity_type.value}",
            ],
        }
        if entity.aliases:
            frontmatter["aliases"] = entity.aliases
        if entity.first_seen:
            frontmatter["first_seen"] = _format_date(entity.first_seen, self.date_format)
        if entity.last_seen:
            frontmatter["last_seen"] = _format_date(entity.last_seen, self.date_format)
        if self._curation and entity.id in self._curation.review_ids:
            frontmatter["needs_review"] = True

        lines: list[str] = []
        lines.append(f"# {entity.name}\n")

        if self.dataview_fields:
            lines.append(f"type:: {entity.entity_type.value}")
            if entity.aliases:
                lines.append(f"aliases:: {', '.join(entity.aliases)}")
            if entity.first_seen:
                lines.append(f"first_seen:: {_format_date(entity.first_seen, self.date_format)}")
            lines.append("")

        if entity.properties:
            lines.append("## Properties\n")
            for key, value in entity.properties.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if entity.source_conversations:
            lines.append("## Conversations\n")
            for conv_id in entity.source_conversations:
                lines.append(f"- [[{conv_id}]]")
            lines.append("")

        if self.backlinks_section:
            related = self._get_related_ids(entity.id, relationships)
            if related:
                lines.append("## Related\n")
                for rid in related:
                    lines.append(f"- [[{rid}]]")
                lines.append("")

        return self._write_note(folder, entity.name, frontmatter, "\n".join(lines))

    def write_concept(
        self, concept: Concept, relationships: list[Relationship]
    ) -> Path | None:
        """Write a concept as an Obsidian note."""
        folder = FOLDER_MAP.get(concept.concept_type.value, "Concepts")
        title = concept.content[:80]

        frontmatter: dict = {
            "title": title,
            "type": concept.concept_type.value,
            "tags": [
                f"{self.tag_prefix}/concept",
                f"{self.tag_prefix}/{concept.concept_type.value}",
            ],
            "confidence": concept.confidence,
        }
        if concept.tags:
            frontmatter["tags"].extend(concept.tags)
        if concept.source_conversation_id:
            frontmatter["source"] = concept.source_conversation_id

        # Curation fields
        if self._curation:
            # effective_confidence = confidence weighted by source quality
            if concept.source_conversation_id in self._curation.source_weights:
                sw = self._curation.source_weights[concept.source_conversation_id]
                frontmatter["effective_confidence"] = round(
                    concept.confidence * sw, 3
                )
            # Review flag
            if concept.id in self._curation.review_ids:
                frontmatter["needs_review"] = True
            # Temporal status (decisions and action items only)
            status = self._curation.concept_statuses.get(concept.id)
            if status:
                frontmatter["status"] = status.status
                if status.valid_from:
                    frontmatter["valid_from"] = _format_date(
                        status.valid_from, self.date_format
                    )
                if status.superseded_by_title:
                    frontmatter["superseded_by"] = (
                        f"[[{_sanitize_filename(status.superseded_by_title)}]]"
                    )
                if status.last_confirmed:
                    frontmatter["last_confirmed"] = _format_date(
                        status.last_confirmed, self.date_format
                    )

        lines: list[str] = []
        lines.append(f"# {title}\n")

        if self.dataview_fields:
            lines.append(f"type:: {concept.concept_type.value}")
            lines.append(f"confidence:: {concept.confidence}")
            if concept.source_conversation_id:
                lines.append(f"source:: [[{concept.source_conversation_id}]]")
            lines.append("")

        lines.append(f"{concept.content}\n")

        if concept.context:
            lines.append("## Context\n")
            lines.append(f"> {concept.context}\n")

        if self.backlinks_section:
            related = self._get_related_ids(concept.id, relationships)
            if related:
                lines.append("## Related\n")
                for rid in related:
                    lines.append(f"- [[{rid}]]")
                lines.append("")

        return self._write_note(folder, title, frontmatter, "\n".join(lines))

    def _write_note(
        self,
        folder: str,
        title: str,
        frontmatter: dict,
        body: str,
    ) -> Path:
        """Write a markdown note with YAML frontmatter to the vault."""
        folder_path = self.vault_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)

        filename = _sanitize_filename(title) + ".md"
        file_path = folder_path / filename

        # Build frontmatter YAML
        fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)

        content = f"---\n{fm_str}---\n\n{body}"
        atomic_write(file_path, content, root=self.vault_path)
        return file_path

    @staticmethod
    def _get_related_ids(
        object_id: str, relationships: list[Relationship]
    ) -> list[str]:
        """Get IDs of objects related to the given object."""
        related: list[str] = []
        for rel in relationships:
            if rel.source_id == object_id:
                related.append(rel.target_id)
            elif rel.target_id == object_id:
                related.append(rel.source_id)
        return sorted(set(related))
