"""Cross-reference and link objects in the knowledge graph.

Builds relationships between entities, concepts, and conversations based on
co-occurrence, shared topics, and explicit references.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.models.concept import Concept
from src.models.entity import Entity
from src.models.message import Conversation
from src.models.relationship import Relationship, RelationshipType
from src.process.extractor import ExtractionResult

logger = logging.getLogger(__name__)


class ObjectLinker:
    """Build relationships between extracted objects.

    Links are created based on:
    - Co-occurrence: entities/concepts appearing in the same conversation
    - Topic overlap: conversations sharing similar topics
    - Explicit references: mentions of entities within concept text
    """

    def link(
        self,
        conversations: list[Conversation],
        extractions: list[ExtractionResult],
    ) -> list[Relationship]:
        """Generate all relationships from conversations and extractions."""
        relationships: list[Relationship] = []

        # Index extractions by conversation
        extraction_map = {e.conversation_id: e for e in extractions}

        # 1. Link entities to their source conversations
        for extraction in extractions:
            for entity in extraction.entities:
                relationships.append(
                    Relationship(
                        source_id=extraction.conversation_id,
                        target_id=entity.id,
                        relationship_type=RelationshipType.MENTIONS,
                        source_conversation_id=extraction.conversation_id,
                    )
                )

        # 2. Link concepts to their source conversations
        for extraction in extractions:
            for concept in extraction.concepts:
                relationships.append(
                    Relationship(
                        source_id=extraction.conversation_id,
                        target_id=concept.id,
                        relationship_type=RelationshipType.RELATES_TO,
                        source_conversation_id=extraction.conversation_id,
                    )
                )

        # 3. Link entities that co-occur in the same conversation
        relationships.extend(self._link_co_occurring_entities(extractions))

        # 4. Link conversations with overlapping topics
        relationships.extend(self._link_conversations_by_topic(conversations))

        # 5. Link entities mentioned in concept text
        all_entities = [e for ex in extractions for e in ex.entities]
        all_concepts = [c for ex in extractions for c in ex.concepts]
        relationships.extend(
            self._link_entity_concept_mentions(all_entities, all_concepts)
        )

        logger.info("Generated %d relationships", len(relationships))
        return relationships

    def _link_co_occurring_entities(
        self, extractions: list[ExtractionResult]
    ) -> list[Relationship]:
        """Link entities that appear in the same conversation."""
        relationships: list[Relationship] = []
        for extraction in extractions:
            entities = extraction.entities
            for i, entity_a in enumerate(entities):
                for entity_b in entities[i + 1 :]:
                    relationships.append(
                        Relationship(
                            source_id=entity_a.id,
                            target_id=entity_b.id,
                            relationship_type=RelationshipType.RELATES_TO,
                            weight=0.5,
                            source_conversation_id=extraction.conversation_id,
                        )
                    )
        return relationships

    def _link_conversations_by_topic(
        self, conversations: list[Conversation]
    ) -> list[Relationship]:
        """Link conversations that share topics."""
        relationships: list[Relationship] = []
        # Build topic → conversation index
        topic_index: dict[str, list[str]] = defaultdict(list)
        for conv in conversations:
            for topic in conv.topics:
                topic_index[topic].append(conv.id)

        # Link conversations sharing a topic
        for topic, conv_ids in topic_index.items():
            for i, id_a in enumerate(conv_ids):
                for id_b in conv_ids[i + 1 :]:
                    relationships.append(
                        Relationship(
                            source_id=id_a,
                            target_id=id_b,
                            relationship_type=RelationshipType.RELATES_TO,
                            weight=0.3,
                            evidence=[f"Shared topic: {topic}"],
                        )
                    )
        return relationships

    def _link_entity_concept_mentions(
        self, entities: list[Entity], concepts: list[Concept]
    ) -> list[Relationship]:
        """Link entities that are mentioned in concept text."""
        relationships: list[Relationship] = []
        for concept in concepts:
            content_lower = concept.content.lower()
            for entity in entities:
                if entity.matches(entity.name) and entity.name.lower() in content_lower:
                    relationships.append(
                        Relationship(
                            source_id=concept.id,
                            target_id=entity.id,
                            relationship_type=RelationshipType.MENTIONS,
                            weight=0.7,
                            evidence=[concept.content[:200]],
                        )
                    )
        return relationships
