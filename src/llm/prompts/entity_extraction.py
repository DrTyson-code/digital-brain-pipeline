"""Entity extraction prompt and response schema.

The LLM is asked to return a ``LLMEntityExtraction`` object, which the
provider validates with Pydantic before returning to the caller.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Prompt version — bump when changing any prompt text so the cache
# automatically invalidates stale entries.
# ---------------------------------------------------------------------------
PROMPT_VERSION = "v1"

# ---------------------------------------------------------------------------
# Prompt text
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an entity extraction system for a personal knowledge base.

Given an AI chat conversation, extract all meaningful entities mentioned.

Entity types (from ontology):
- PERSON: Named individuals (authors, researchers, colleagues)
- TOOL: Software tools, libraries, frameworks, APIs, languages
- ORGANIZATION: Companies, institutions, teams
- PROJECT: Named projects or repositories
- LOCATION: Physical or virtual locations

Rules:
1. Extract the canonical name (e.g. "PyTorch" not "pytorch" or "torch")
2. Include aliases when the conversation uses multiple names for the same entity
3. Only extract entities actually discussed, not passing boilerplate mentions
4. Provide a one-sentence description grounded in how the entity appears in THIS conversation
5. Assign a confidence score (0.0–1.0) based on how clearly the entity is identified
6. Include up to 3 direct quotes from the conversation as grounding evidence

Return JSON matching the provided schema exactly."""

USER_PROMPT_TEMPLATE = """\
Extract entities from this {source} conversation.

Topic: {topic}
Messages: {message_count}

--- CONVERSATION ---
{conversation_text}
--- END ---

Extract all entities as a JSON object."""

# ---------------------------------------------------------------------------
# Response schemas (validated by Pydantic before being used downstream)
# ---------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    """A single entity extracted from a conversation by the LLM."""

    name: str = Field(description="Canonical entity name")
    entity_type: str = Field(description="One of: PERSON, TOOL, ORGANIZATION, PROJECT, LOCATION")
    description: str = Field(description="One-sentence description grounded in the conversation")
    aliases: list[str] = Field(default_factory=list, description="Alternative names used in the text")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence score")
    source_quotes: list[str] = Field(
        default_factory=list, description="Up to 3 supporting quotes from the conversation"
    )


class LLMEntityExtraction(BaseModel):
    """Top-level response schema for entity extraction."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
