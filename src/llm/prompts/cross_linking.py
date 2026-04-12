"""Cross-linking and deduplication prompt and response schema.

Given entities/concepts from *multiple* conversations, the LLM identifies
duplicates (same real-world thing, different names) and cross-conversation links.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a knowledge graph deduplication and linking system.

Given two sets of entities/concepts from different conversations, identify:
1. DUPLICATES: Same real-world entity with different names or slight variations
2. CROSS-LINKS: Entities from different conversations that are meaningfully related
3. MERGE CANDIDATES: Entities that should be consolidated into one note

Rules:
1. Be conservative with merges — only merge when clearly the same thing
2. Cross-links should be substantive, not just "both mention Python"
3. For each proposed link or merge, explain the reasoning
4. Consider aliases, abbreviations, and common variations

Return JSON matching the provided schema exactly."""

USER_PROMPT_TEMPLATE = """\
Conversation A entities:
{entities_a_json}

Conversation B entities:
{entities_b_json}

Identify duplicates, cross-links, and merge candidates."""


class DuplicatePair(BaseModel):
    entity_a: str
    entity_b: str
    confidence: float = Field(ge=0.0, le=1.0)
    canonical_name: str = Field(description="Preferred name after merge")
    reason: str


class CrossLink(BaseModel):
    entity_a: str
    entity_b: str
    relationship: str
    reason: str


class MergeCandidate(BaseModel):
    entities: list[str] = Field(description="Names of entities to merge")
    canonical_name: str
    reason: str


class DeduplicationResult(BaseModel):
    """Top-level response schema for cross-linking / deduplication."""

    duplicates: list[DuplicatePair] = Field(default_factory=list)
    cross_links: list[CrossLink] = Field(default_factory=list)
    merge_candidates: list[MergeCandidate] = Field(default_factory=list)
