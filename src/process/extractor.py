"""Extract entities and concepts from conversations.

Provides rule-based extraction of named entities (people, orgs, tools, projects)
and concepts (decisions, action items, insights) from conversation text.

For production accuracy, extend with an LLM extraction call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import Conversation

logger = logging.getLogger(__name__)

# Patterns for action item detection
ACTION_PATTERNS = [
    re.compile(r"(?:TODO|FIXME|HACK|ACTION):\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:need to|should|must|have to|going to)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:let's|let me|I'll|we'll|we should)\s+(.+?)(?:\.|$)", re.IGNORECASE),
]

# Patterns for decision detection
DECISION_PATTERNS = [
    re.compile(r"(?:decided to|decision:|we decided|I decided)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:going with|chose|chosen|selected|picking)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:the plan is to|approach:|solution:)\s+(.+?)(?:\.|$)", re.IGNORECASE),
]

# Patterns for question/insight detection
QUESTION_PATTERNS = [
    re.compile(r"(?:how (?:do|can|should|would)|what (?:is|are|if)|why (?:does|is|do)|when (?:should|do))\s+(.+?\?)", re.IGNORECASE),
]

INSIGHT_PATTERNS = [
    re.compile(r"(?:key (?:insight|takeaway|learning)|TIL|learned that|realized that)\s*:?\s*(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:the (?:important|key|main) (?:thing|point|idea) is)\s+(.+?)(?:\.|$)", re.IGNORECASE),
]

# Common tool/technology names to look for
KNOWN_TOOLS = {
    "python", "javascript", "typescript", "rust", "go", "java", "kotlin",
    "react", "vue", "angular", "svelte", "nextjs", "django", "flask",
    "fastapi", "express", "docker", "kubernetes", "terraform", "aws",
    "gcp", "azure", "postgresql", "mysql", "mongodb", "redis", "kafka",
    "git", "github", "gitlab", "obsidian", "notion", "linear", "figma",
    "pytorch", "tensorflow", "pandas", "numpy", "langchain", "openai",
    "anthropic", "claude", "chatgpt", "gemini", "llama", "mistral",
    "palantir", "foundry", "vscode", "neovim", "vim",
}


@dataclass
class ExtractionResult:
    """Result of entity/concept extraction from a conversation."""

    conversation_id: str
    entities: List[Entity] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)


class EntityConceptExtractor:
    """Extract entities and concepts from conversations using pattern matching."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def extract(self, conversation: Conversation) -> ExtractionResult:
        """Extract all entities and concepts from a conversation."""
        result = ExtractionResult(conversation_id=conversation.id)

        full_text = "\n".join(m.content for m in conversation.messages)

        result.entities.extend(self._extract_tools(full_text, conversation.id))
        result.concepts.extend(self._extract_action_items(full_text, conversation.id))
        result.concepts.extend(self._extract_decisions(full_text, conversation.id))
        result.concepts.extend(self._extract_questions(full_text, conversation.id))
        result.concepts.extend(self._extract_insights(full_text, conversation.id))

        # Extract topic-level concept from conversation title
        if conversation.title:
            result.concepts.append(
                Concept(
                    concept_type=ConceptType.TOPIC,
                    content=conversation.title,
                    source_conversation_id=conversation.id,
                    confidence=0.9,
                )
            )

        return result

    def _extract_tools(self, text: str, conversation_id: str) -> list[Entity]:
        """Extract known tool/technology mentions."""
        entities: list[Entity] = []
        text_lower = text.lower()
        for tool in KNOWN_TOOLS:
            # Match whole words only
            if re.search(rf"\b{re.escape(tool)}\b", text_lower):
                entities.append(
                    Entity(
                        entity_type=EntityType.TOOL,
                        name=tool,
                        source_conversations=[conversation_id],
                    )
                )
        return entities

    def _extract_action_items(
        self, text: str, conversation_id: str
    ) -> list[Concept]:
        """Extract action items from text."""
        return self._extract_by_patterns(
            text, ACTION_PATTERNS, ConceptType.ACTION_ITEM, conversation_id
        )

    def _extract_decisions(
        self, text: str, conversation_id: str
    ) -> list[Concept]:
        """Extract decisions from text."""
        return self._extract_by_patterns(
            text, DECISION_PATTERNS, ConceptType.DECISION, conversation_id
        )

    def _extract_questions(
        self, text: str, conversation_id: str
    ) -> list[Concept]:
        """Extract open questions from text."""
        return self._extract_by_patterns(
            text, QUESTION_PATTERNS, ConceptType.QUESTION, conversation_id
        )

    def _extract_insights(
        self, text: str, conversation_id: str
    ) -> list[Concept]:
        """Extract insights and key learnings from text."""
        return self._extract_by_patterns(
            text, INSIGHT_PATTERNS, ConceptType.INSIGHT, conversation_id
        )

    def _extract_by_patterns(
        self,
        text: str,
        patterns: list[re.Pattern[str]],
        concept_type: ConceptType,
        conversation_id: str,
    ) -> list[Concept]:
        """Generic pattern-based extraction."""
        concepts: list[Concept] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in pattern.finditer(text):
                content = match.group(1).strip()
                # Basic quality filters
                if len(content) < 10 or len(content) > 500:
                    continue
                content_key = content.lower()
                if content_key in seen:
                    continue
                seen.add(content_key)
                concepts.append(
                    Concept(
                        concept_type=concept_type,
                        content=content,
                        context=match.group(0).strip(),
                        confidence=0.6,
                        source_conversation_id=conversation_id,
                    )
                )
        return concepts

    def extract_batch(
        self, conversations: list[Conversation]
    ) -> list[ExtractionResult]:
        """Extract entities and concepts from a batch of conversations."""
        return [self.extract(conv) for conv in conversations]
