"""Score each conversation's depth and quality for downstream weighting.

Signals used:
- Message count (more messages = deeper discussion)
- Total word count across all messages
- Average message length (longer = deeper per-turn engagement)
- Whether any messages have a model field (indicates tool/function call use)
- Session duration (if timestamps available)
- Number of topics assigned by the classifier
- Whether decisions or action items were extracted (indicates conclusions reached)

Output: a source_weight float in [0.0, 1.0] per conversation.
Quick dispatch tasks tend to score ~0.3, deep work sessions ~0.9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional

from src.models.concept import ConceptType
from src.models.message import Conversation
from src.process.extractor import ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class SourceScorer:
    """Compute a quality weight for each conversation.

    Each signal is normalized to [0, 1] then combined via weighted average.
    Weights must sum to 1.0.
    """

    weight_message_count: float = 0.20
    weight_word_count: float = 0.20
    weight_avg_length: float = 0.15
    weight_tool_calls: float = 0.10
    weight_duration: float = 0.10
    weight_topics: float = 0.10
    weight_conclusions: float = 0.15

    # Normalization caps (class-level, not dataclass fields)
    MAX_MESSAGES: ClassVar[int] = 30
    MAX_WORDS: ClassVar[int] = 6000
    MAX_AVG_LENGTH: ClassVar[int] = 200
    MAX_DURATION_SECS: ClassVar[float] = 7200.0
    MAX_TOPICS: ClassVar[int] = 5

    def score(
        self,
        conversation: Conversation,
        extraction: Optional[ExtractionResult] = None,
    ) -> float:
        """Compute a source weight in [0.0, 1.0] for a single conversation."""
        # Signal 1: message count
        msg_score = min(conversation.message_count / self.MAX_MESSAGES, 1.0)

        # Signal 2: total word count
        total_words = sum(len(m.content.split()) for m in conversation.messages)
        word_score = min(total_words / self.MAX_WORDS, 1.0)

        # Signal 3: average message length
        if conversation.message_count > 0:
            avg_len = total_words / conversation.message_count
        else:
            avg_len = 0.0
        avg_score = min(avg_len / self.MAX_AVG_LENGTH, 1.0)

        # Signal 4: tool/model use (model field set on any message)
        has_tools = any(m.model is not None for m in conversation.messages)
        tool_score = 1.0 if has_tools else 0.0

        # Signal 5: session duration
        duration = conversation.duration
        if duration is not None:
            dur_score = min(duration / self.MAX_DURATION_SECS, 1.0)
        else:
            dur_score = 0.0

        # Signal 6: topic richness from classifier
        topic_score = min(len(conversation.topics) / self.MAX_TOPICS, 1.0)

        # Signal 7: extracted decisions or action items
        if extraction is not None:
            has_conclusions = any(
                c.concept_type in (ConceptType.DECISION, ConceptType.ACTION_ITEM)
                for c in extraction.concepts
            )
        else:
            has_conclusions = False
        conclusion_score = 1.0 if has_conclusions else 0.0

        raw = (
            self.weight_message_count * msg_score
            + self.weight_word_count * word_score
            + self.weight_avg_length * avg_score
            + self.weight_tool_calls * tool_score
            + self.weight_duration * dur_score
            + self.weight_topics * topic_score
            + self.weight_conclusions * conclusion_score
        )

        weight = max(0.0, min(1.0, raw))
        logger.debug(
            "Conversation %s scored %.3f "
            "(msgs=%d, words=%d, avg_len=%.0f, tools=%s, dur=%s, topics=%d, conclusions=%s)",
            conversation.id,
            weight,
            conversation.message_count,
            total_words,
            avg_len,
            has_tools,
            f"{duration:.0f}s" if duration else "n/a",
            len(conversation.topics),
            has_conclusions,
        )
        return weight

    def score_batch(
        self,
        conversations: List[Conversation],
        extractions: Optional[List[ExtractionResult]] = None,
    ) -> Dict[str, float]:
        """Score a batch of conversations.

        Args:
            conversations: All conversations to score.
            extractions: Optional corresponding ExtractionResults used for
                         the conclusions signal.

        Returns:
            Dict mapping conversation_id → source_weight.
        """
        extraction_map: Dict[str, ExtractionResult] = {}
        if extractions:
            extraction_map = {ex.conversation_id: ex for ex in extractions}

        weights: Dict[str, float] = {}
        for conv in conversations:
            ext = extraction_map.get(conv.id)
            weights[conv.id] = self.score(conv, ext)

        if weights:
            avg = sum(weights.values()) / len(weights)
            logger.info(
                "Source scoring: %d conversations, avg weight=%.3f, "
                "min=%.3f, max=%.3f",
                len(weights),
                avg,
                min(weights.values()),
                max(weights.values()),
            )
        return weights
