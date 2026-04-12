"""Relationship mapping prompt and response schema.

Given a list of extracted entities and the original conversation text, the LLM
identifies directed relationships between entities.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a relationship extraction system for a personal knowledge base.

Given a conversation and a list of entities already extracted from it, identify
relationships between entities.

Relationship types (from ontology):
- USES: entity uses or depends on another (e.g. "project USES pytorch")
- CREATED_BY: entity was created by another
- PART_OF: entity is a component of another
- RELATED_TO: general topical relationship
- COMPARED_WITH: entities are explicitly compared
- IMPLEMENTS: entity implements a concept or pattern
- EXTENDS: entity builds on another

Rules:
1. Only extract relationships with clear evidence in the conversation text
2. Each relationship requires a direction (subject → predicate → object)
3. Assign confidence based on how explicitly the relationship is stated
4. Include a brief evidence quote from the conversation

Return JSON matching the provided schema exactly."""

USER_PROMPT_TEMPLATE = """\
Given these entities extracted from a {source} conversation:

{entities_json}

And this conversation:
--- CONVERSATION ---
{conversation_text}
--- END ---

Extract all relationships between these entities."""


class ExtractedRelationship(BaseModel):
    """A single relationship between two entities."""

    subject: str = Field(description="Name of the subject entity (must match an extracted entity)")
    predicate: str = Field(description="Relationship type from the ontology")
    object: str = Field(description="Name of the object entity (must match an extracted entity)")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(description="Short quote from the conversation supporting this relationship")
    bidirectional: bool = Field(default=False)


class LLMRelationshipExtraction(BaseModel):
    """Top-level response schema for relationship extraction."""

    relationships: list[ExtractedRelationship] = Field(default_factory=list)
