"""Classify conversations by topic and type.

This module provides rule-based classification of conversations into topics
and categories. It uses keyword matching and heuristics as a baseline;
a future LLM-powered classifier can extend or replace this.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.models.message import Conversation

logger = logging.getLogger(__name__)

# Topic keyword lists — each maps a topic label to its trigger words
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "programming": [
        "code", "function", "class", "api", "debug", "error", "python",
        "javascript", "typescript", "rust", "golang", "git", "docker",
        "database", "sql", "algorithm", "refactor", "deploy", "ci/cd",
    ],
    "data-science": [
        "dataset", "model", "training", "neural", "machine learning",
        "deep learning", "pandas", "numpy", "tensorflow", "pytorch",
        "regression", "classification", "clustering", "statistics",
    ],
    "writing": [
        "essay", "article", "blog", "draft", "edit", "proofread",
        "paragraph", "outline", "narrative", "creative writing",
    ],
    "research": [
        "paper", "study", "literature", "citation", "hypothesis",
        "experiment", "methodology", "findings", "peer review",
    ],
    "business": [
        "strategy", "market", "revenue", "startup", "pitch",
        "investor", "customer", "product", "roadmap", "okr",
    ],
    "personal": [
        "health", "fitness", "recipe", "travel", "hobby",
        "relationship", "advice", "journal", "meditation",
    ],
}


@dataclass
class ClassificationResult:
    """Result of classifying a single conversation."""

    conversation_id: str
    topics: List[str] = field(default_factory=list)
    primary_topic: Optional[str] = None
    confidence: float = 0.0


class ConversationClassifier:
    """Classify conversations into topics using keyword matching.

    This is the baseline classifier. For production use, swap in an
    LLM-based classifier that calls the extraction API.
    """

    def __init__(
        self,
        topic_keywords: dict[str, list[str]] | None = None,
        max_topics: int = 5,
    ) -> None:
        self.topic_keywords = topic_keywords or TOPIC_KEYWORDS
        self.max_topics = max_topics

    def classify(self, conversation: Conversation) -> ClassificationResult:
        """Classify a conversation and return matched topics."""
        # Build a single text blob from all messages
        full_text = " ".join(m.content.lower() for m in conversation.messages)
        word_count = len(full_text.split())
        if word_count == 0:
            return ClassificationResult(conversation_id=conversation.id)

        # Score each topic by keyword frequency
        scores: dict[str, float] = {}
        for topic, keywords in self.topic_keywords.items():
            count = sum(
                len(re.findall(rf"\b{re.escape(kw)}\b", full_text))
                for kw in keywords
            )
            if count > 0:
                # Normalize by word count to avoid length bias
                scores[topic] = count / word_count

        if not scores:
            return ClassificationResult(conversation_id=conversation.id)

        # Sort by score descending, take top N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_topics = [t for t, _ in ranked[: self.max_topics]]
        top_score = ranked[0][1]

        return ClassificationResult(
            conversation_id=conversation.id,
            topics=top_topics,
            primary_topic=top_topics[0],
            confidence=min(top_score * 100, 1.0),  # rough confidence
        )

    def classify_batch(
        self, conversations: list[Conversation]
    ) -> list[ClassificationResult]:
        """Classify a batch of conversations."""
        results = []
        for conv in conversations:
            result = self.classify(conv)
            conv.topics = result.topics
            results.append(result)
        return results
