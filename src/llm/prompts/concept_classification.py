"""Concept classification prompt and response schema.

Classifies a conversation into the knowledge ontology and extracts high-level
concepts discussed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a concept classification system for a personal knowledge base.

Given a conversation, classify it into the knowledge ontology and extract
high-level concepts discussed.

Top-level domains:
- software_engineering
- machine_learning
- data_science
- medicine  (anesthesiology, pharmacology, physiology)
- productivity
- writing
- personal

Rules:
1. Assign a primary domain and up to 3 secondary domains
2. Extract specific concepts within those domains
   (e.g. "transformer architecture" within machine_learning)
3. Identify the conversation PURPOSE:
   learning | problem_solving | brainstorming | debugging | research | planning | creative
4. Rate conversation DEPTH: surface | moderate | deep
5. Concept names should be at a granularity useful for cross-conversation linking —
   not too broad ("programming"), not too narrow ("line 42 bug")
6. Write a 2-3 sentence summary suitable for an Obsidian note

Return JSON matching the provided schema exactly."""

USER_PROMPT_TEMPLATE = """\
Classify this {source} conversation and extract its key concepts.

--- CONVERSATION ---
{conversation_text}
--- END ---"""


class ExtractedConcept(BaseModel):
    """A single concept extracted from a conversation."""

    name: str
    domain: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class ConversationClassification(BaseModel):
    """Top-level response schema for concept classification."""

    primary_domain: str
    secondary_domains: list[str] = Field(default_factory=list)
    purpose: str
    depth: str = Field(description="surface | moderate | deep")
    concepts: list[ExtractedConcept] = Field(default_factory=list)
    summary: str = Field(description="2-3 sentence summary for the Obsidian note")
