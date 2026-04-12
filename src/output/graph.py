"""Optional graph database export for the knowledge graph.

Exports the ontology to a JSON graph format compatible with common
graph visualization tools (e.g., D3.js, Cytoscape, Gephi).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.models.concept import Concept
from src.models.entity import Entity
from src.models.message import Conversation
from src.models.relationship import Relationship

logger = logging.getLogger(__name__)


class GraphExporter:
    """Export the knowledge graph to JSON graph format."""

    def export(
        self,
        conversations: list[Conversation],
        entities: list[Entity],
        concepts: list[Concept],
        relationships: list[Relationship],
        output_path: Path,
    ) -> Path:
        """Export the full graph as a JSON file.

        Format:
            {
                "nodes": [ { "id": ..., "type": ..., "label": ..., "properties": {...} } ],
                "edges": [ { "source": ..., "target": ..., "type": ..., "weight": ... } ]
            }
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        # Conversations as nodes
        for conv in conversations:
            nodes.append({
                "id": conv.id,
                "type": "conversation",
                "label": conv.title or f"Conv {conv.id[:8]}",
                "properties": {
                    "platform": conv.platform.value,
                    "message_count": conv.message_count,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                    "topics": conv.topics,
                },
            })

        # Entities as nodes
        for entity in entities:
            nodes.append({
                "id": entity.id,
                "type": entity.entity_type.value,
                "label": entity.name,
                "properties": {
                    "aliases": entity.aliases,
                    **entity.properties,
                },
            })

        # Concepts as nodes
        for concept in concepts:
            nodes.append({
                "id": concept.id,
                "type": concept.concept_type.value,
                "label": concept.content[:80],
                "properties": {
                    "confidence": concept.confidence,
                    "full_content": concept.content,
                },
            })

        # Relationships as edges
        for rel in relationships:
            edges.append({
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.relationship_type.value,
                "weight": rel.weight,
            })

        graph = {"nodes": nodes, "edges": edges}

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(graph, indent=2, default=str), encoding="utf-8"
        )

        logger.info(
            "Exported graph: %d nodes, %d edges → %s",
            len(nodes),
            len(edges),
            output_path,
        )
        return output_path
