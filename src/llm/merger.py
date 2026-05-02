"""Merge rule-based and LLM extraction results.

This module combines entity and concept extractions from two sources:
- Rule-based extraction (pattern matching from src.process.extractor)
- LLM-based extraction (structured API calls)

The merger deduplicates entities and concepts that refer to the same thing,
prioritizing LLM results (higher quality, richer metadata) while preserving
unique findings from rule-based extraction.

Key strategy:
- Entity dedup: LLM entities are primary; rule-based entities add data if they
  find something the LLM missed. Uses normalized string similarity for matching.
- Concept dedup: Keep LLM version (better phrasing, higher confidence) but note
  rule-based confirmation when both sources extract the same thing.
- For non-matching entities/concepts: keep both for broader coverage.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from typing import List

from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.process.extractor import ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class MergeStatistics:
    """Statistics about a merge operation."""

    total_rule_entities: int = 0
    total_llm_entities: int = 0
    matched_entities: int = 0
    unique_rule_entities: int = 0
    unique_llm_entities: int = 0
    final_entity_count: int = 0

    total_rule_concepts: int = 0
    total_llm_concepts: int = 0
    matched_concepts: int = 0
    unique_rule_concepts: int = 0
    unique_llm_concepts: int = 0
    final_concept_count: int = 0

    def __str__(self) -> str:
        """Return human-readable summary."""
        return (
            f"Entities: {self.matched_entities} matched, "
            f"{self.unique_rule_entities} unique from rules, "
            f"{self.unique_llm_entities} unique from LLM → {self.final_entity_count} total | "
            f"Concepts: {self.matched_concepts} matched, "
            f"{self.unique_rule_concepts} unique from rules, "
            f"{self.unique_llm_concepts} unique from LLM → {self.final_concept_count} total"
        )


class ExtractionMerger:
    """Merge rule-based and LLM extraction results."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_concepts_per_conversation: int = 200,
    ) -> None:
        """Initialize the merger.

        Args:
            similarity_threshold: Minimum normalized similarity ratio (0.0-1.0) for
                considering two entity names as duplicates. Defaults to 0.85 (85%).
                Uses simple character-based similarity (difflib.SequenceMatcher),
                not fuzzy matching, to avoid extra dependencies.
            max_concepts_per_conversation: Defensive upper bound on per-conversation
                concept count. The merge loop is O(n*m) over rule and LLM concept
                lists; without a cap, a long Cowork-style conversation that
                produces 1000+ regex matches dominates pipeline runtime. The
                primary cap is applied at the extractor (see EntityConceptExtractor.
                max_concepts_per_conversation); this is a defensive backup for
                cases where extractors produce volumes (e.g., future paths).
                Set <= 0 to disable.
        """
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        self.similarity_threshold = similarity_threshold
        self.max_concepts_per_conversation = max_concepts_per_conversation

    def merge(
        self, rule_result: ExtractionResult, llm_result: ExtractionResult
    ) -> ExtractionResult:
        """Merge rule-based and LLM extraction results for the same conversation.

        Strategy:
        1. Start with LLM entities (higher quality, richer metadata)
        2. For each rule-based entity, check if it matches any LLM entity
           - If match: merge the rule-based data into the LLM entity
           - If no match: add the rule-based entity as-is
        3. For concepts: keep all from both, deduplicate by content similarity
        4. LLM concepts are assigned higher confidence than rule-based ones

        Args:
            rule_result: ExtractionResult from rule-based extraction
            llm_result: ExtractionResult from LLM extraction

        Returns:
            Merged ExtractionResult with deduplicated entities and concepts.
        """
        stats = MergeStatistics(
            total_rule_entities=len(rule_result.entities),
            total_llm_entities=len(llm_result.entities),
            total_rule_concepts=len(rule_result.concepts),
            total_llm_concepts=len(llm_result.concepts),
        )

        # Defensive cap on per-conversation concept counts. The merge below is
        # O(n*m) over rule_result.concepts and llm_result.concepts; uncapped
        # 1000+ concept lists from a single conversation dominate pipeline runtime.
        rule_result, llm_result = self._cap_concepts(rule_result, llm_result)

        # Merge entities
        merged_entities = list(llm_result.entities)  # Start with LLM entities
        matched_entity_indices = set()

        for rule_entity in rule_result.entities:
            matched_idx = None
            for i, llm_entity in enumerate(merged_entities):
                if self._entities_match(rule_entity, llm_entity):
                    matched_idx = i
                    break

            if matched_idx is not None:
                # Merge rule-based data into the LLM entity
                merged_entities[matched_idx].merge(rule_entity)
                stats.matched_entities += 1
                matched_entity_indices.add(matched_idx)
            else:
                # Rule-based entity not found in LLM results; add it
                merged_entities.append(rule_entity)
                stats.unique_rule_entities += 1

        stats.unique_llm_entities = len(merged_entities) - stats.unique_rule_entities
        stats.final_entity_count = len(merged_entities)

        # Merge concepts
        merged_concepts = list(llm_result.concepts)  # Start with LLM concepts
        matched_concept_indices = set()

        for rule_concept in rule_result.concepts:
            matched_idx = None
            for i, llm_concept in enumerate(merged_concepts):
                if self._concepts_match(rule_concept, llm_concept):
                    matched_idx = i
                    break

            if matched_idx is not None:
                # Concept matched: keep LLM version but note rule confirmation
                # Increase confidence slightly if rule-based also found it
                llm_concept = merged_concepts[matched_idx]
                if llm_concept.confidence < 0.95:
                    llm_concept.confidence = min(0.95, llm_concept.confidence + 0.05)
                # Add rule source if not already present
                if rule_concept.source_conversation_id:
                    if (
                        rule_concept.source_conversation_id
                        not in llm_concept.tags
                    ):
                        llm_concept.tags.append(
                            f"confirmed_by_rules_{rule_concept.source_conversation_id}"
                        )
                stats.matched_concepts += 1
                matched_concept_indices.add(matched_idx)
            else:
                # Rule-based concept not found in LLM results; add it
                # Lower its confidence slightly since it's rule-based only
                if rule_concept.confidence > 0.7:
                    rule_concept.confidence = 0.7
                merged_concepts.append(rule_concept)
                stats.unique_rule_concepts += 1

        stats.unique_llm_concepts = len(merged_concepts) - stats.unique_rule_concepts
        stats.final_concept_count = len(merged_concepts)

        # Create merged result
        merged = ExtractionResult(
            conversation_id=llm_result.conversation_id or rule_result.conversation_id,
            entities=merged_entities,
            concepts=merged_concepts,
        )

        # Log merge statistics
        logger.info(
            f"Merged results for conversation {merged.conversation_id}: {stats}"
        )

        return merged

    def _cap_concepts(
        self,
        rule_result: ExtractionResult,
        llm_result: ExtractionResult,
    ) -> tuple[ExtractionResult, ExtractionResult]:
        """Truncate oversized concept lists with a warning.

        Returns possibly-truncated copies. Original objects are not mutated.
        """
        cap = self.max_concepts_per_conversation
        if cap <= 0:
            return rule_result, llm_result

        capped_rule = rule_result
        capped_llm = llm_result

        if len(rule_result.concepts) > cap:
            logger.warning(
                "Merger defensive cap: rule_result for conversation %s had %d "
                "concepts, truncating to %d. Primary cap should apply at the "
                "extractor; this fallback caught the overage.",
                rule_result.conversation_id,
                len(rule_result.concepts),
                cap,
            )
            capped_rule = ExtractionResult(
                conversation_id=rule_result.conversation_id,
                entities=rule_result.entities,
                concepts=rule_result.concepts[:cap],
            )

        if len(llm_result.concepts) > cap:
            logger.warning(
                "Merger defensive cap: llm_result for conversation %s had %d "
                "concepts, truncating to %d.",
                llm_result.conversation_id,
                len(llm_result.concepts),
                cap,
            )
            capped_llm = ExtractionResult(
                conversation_id=llm_result.conversation_id,
                entities=llm_result.entities,
                concepts=llm_result.concepts[:cap],
            )

        return capped_rule, capped_llm

    def merge_batch(
        self,
        rule_results: list[ExtractionResult],
        llm_results: list[ExtractionResult],
    ) -> list[ExtractionResult]:
        """Merge paired extraction results by conversation ID.

        For each rule-based result, finds the matching LLM result and merges them.
        If an LLM result is missing for a conversation, uses the rule-based result
        as-is. If a rule-based result is missing, uses the LLM result as-is.

        Args:
            rule_results: List of ExtractionResults from rule-based extraction
            llm_results: List of ExtractionResults from LLM extraction

        Returns:
            List of merged ExtractionResults, one per unique conversation_id.
        """
        # Index LLM results by conversation_id for O(1) lookup
        llm_by_conv = {r.conversation_id: r for r in llm_results}
        rule_by_conv = {r.conversation_id: r for r in rule_results}

        merged_results = []
        processed_convs = set()

        # Process all rule-based results
        for rule_result in rule_results:
            conv_id = rule_result.conversation_id
            processed_convs.add(conv_id)

            if conv_id in llm_by_conv:
                # Both sources exist; merge them
                merged = self.merge(rule_result, llm_by_conv[conv_id])
                merged_results.append(merged)
            else:
                # Only rule-based result exists
                logger.info(
                    f"No LLM result for conversation {conv_id}; using rule-based only"
                )
                merged_results.append(rule_result)

        # Process LLM results that don't have rule-based counterparts
        for llm_result in llm_results:
            conv_id = llm_result.conversation_id
            if conv_id not in processed_convs:
                logger.info(
                    f"No rule-based result for conversation {conv_id}; using LLM only"
                )
                merged_results.append(llm_result)

        logger.info(
            f"Merged batch: {len(merged_results)} conversations "
            f"({len(rule_by_conv)} from rules, {len(llm_by_conv)} from LLM)"
        )

        return merged_results

    def _entities_match(self, a: Entity, b: Entity) -> bool:
        """Check if two entities refer to the same thing.

        Two entities match if:
        1. They have the same entity_type
        2. AND one of:
           - They have exact name/alias match (case-insensitive)
           - Their normalized names exceed the similarity threshold

        Args:
            a: First entity to compare
            b: Second entity to compare

        Returns:
            True if entities refer to the same thing, False otherwise.
        """
        # Must be the same type
        if a.entity_type != b.entity_type:
            return False

        # Check for exact/alias matches
        if a.matches(b.name) or b.matches(a.name):
            return True

        # Check normalized similarity
        sim = self._similarity(
            self._normalize(a.name), self._normalize(b.name)
        )
        return sim >= self.similarity_threshold

    def _concepts_match(self, a: Concept, b: Concept) -> bool:
        """Check if two concepts are duplicates.

        Two concepts match if:
        1. They have the same concept_type
        2. AND their normalized content is similar (exceeds similarity threshold)

        Note: For concepts, we use the full normalized content comparison
        rather than exact matching, since LLMs and rules may phrase the same
        idea differently.

        Args:
            a: First concept to compare
            b: Second concept to compare

        Returns:
            True if concepts represent the same idea, False otherwise.
        """
        # Must be the same type
        if a.concept_type != b.concept_type:
            return False

        # Check content similarity
        sim = self._similarity(
            self._normalize(a.content), self._normalize(b.content)
        )
        return sim >= self.similarity_threshold

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison.

        Converts to lowercase, strips leading/trailing whitespace,
        and collapses consecutive whitespace to single spaces.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        return " ".join(text.lower().split())

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Calculate character-based similarity ratio between two strings.

        Uses difflib.SequenceMatcher to compute the ratio of matching
        characters. Values range from 0.0 (completely different) to 1.0
        (identical).

        Args:
            a: First string to compare
            b: Second string to compare

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not a or not b:
            return 1.0 if a == b else 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()
