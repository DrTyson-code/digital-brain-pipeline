"""LLM-powered entity, relationship, and concept extraction.

Orchestrates multi-stage extraction using an LLM provider:
1. Entity extraction (PERSON, TOOL, ORGANIZATION, PROJECT, LOCATION)
2. Relationship mapping (USES, CREATED_BY, PART_OF, etc.)
3. Concept classification (topics, decisions, insights, etc.)

Integrates caching, cost tracking, and budget enforcement.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.llm.cache import ExtractionCache
from src.llm.cost import BudgetExceeded, CostTracker
from src.llm.prompts import concept_classification, entity_extraction, relationship_mapping
from src.llm.provider import LLMProvider, LLMResponse
from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
from src.models.message import Conversation
from src.models.relationship import Relationship, RelationshipType
from src.process.extractor import ExtractionResult

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Extract entities, relationships, and concepts using an LLM provider.

    Supports caching, cost tracking, and budget enforcement across multiple
    extraction stages.
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache: Optional[ExtractionCache] = None,
        cost_tracker: Optional[CostTracker] = None,
        max_conversation_tokens: int = 8000,
    ) -> None:
        """Initialize the LLM extractor.

        Args:
            provider: LLM provider instance (Claude, OpenAI, Ollama, etc.)
            cache: Optional extraction cache (SQLite-backed)
            cost_tracker: Optional cost/budget tracker
            max_conversation_tokens: Max tokens to include in context
                                     (estimated as chars / 4)
        """
        self.provider = provider
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.max_conversation_tokens = max_conversation_tokens

    async def extract(self, conversation: Conversation) -> ExtractionResult:
        """Extract entities, relationships, and concepts from a conversation.

        Stages:
        1. Entity extraction
        2. Relationship mapping (uses entities from stage 1)
        3. Concept classification

        Skips a stage if budget is exceeded; raises BudgetExceeded only on hard limit.

        Args:
            conversation: The conversation to extract from

        Returns:
            ExtractionResult with extracted entities and concepts

        Raises:
            BudgetExceeded: If any stage would exceed the hard budget limit
        """
        result = ExtractionResult(conversation_id=conversation.id)

        # Build conversation text and hash it
        conversation_text = self._build_conversation_text(conversation)
        conversation_hash = ExtractionCache.hash_conversation(conversation_text)

        logger.info(
            "Extracting from conversation %s (hash: %s, %d messages)",
            conversation.id,
            conversation_hash[:12],
            conversation.message_count,
        )

        try:
            # Stage 1: Entity extraction
            entities_llm = await self._extract_entities(
                conversation_hash=conversation_hash,
                conversation_text=conversation_text,
                source=conversation.platform.value,
                topic=conversation.title or "Unknown",
                message_count=conversation.message_count,
            )
            if entities_llm is not None:
                result.entities = self._convert_entities(
                    entities_llm, conversation.id
                )
                logger.debug(
                    "Extracted %d entities from %s",
                    len(result.entities),
                    conversation.id,
                )

            # Stage 2: Relationship extraction (needs entities as input)
            if result.entities:
                entities_json = json.dumps(
                    [{"name": e.name, "type": e.entity_type.value}
                     for e in result.entities],
                    indent=2,
                )
                relationships_llm = await self._extract_relationships(
                    conversation_hash=conversation_hash,
                    conversation_text=conversation_text,
                    source=conversation.platform.value,
                    entities_json=entities_json,
                )
                if relationships_llm is not None:
                    # Store relationships as concepts for now
                    # (relationship model integration happens in downstream stages)
                    logger.debug(
                        "Extracted %d relationships from %s",
                        len(relationships_llm),
                        conversation.id,
                    )

            # Stage 3: Concept classification
            classification = await self._classify_concepts(
                conversation_hash=conversation_hash,
                conversation_text=conversation_text,
                source=conversation.platform.value,
            )
            if classification is not None:
                result.concepts = self._convert_concepts(
                    classification, conversation.id
                )
                logger.debug(
                    "Extracted %d concepts from %s",
                    len(result.concepts),
                    conversation.id,
                )

        except BudgetExceeded as exc:
            logger.error(
                "Budget exceeded for conversation %s: %s",
                conversation.id,
                exc,
            )
            raise

        return result

    async def extract_batch(
        self,
        conversations: list[Conversation],
        batch_size: int = 10,
    ) -> list[ExtractionResult]:
        """Extract entities and concepts from multiple conversations.

        Processes in batches with progress logging. Errors in individual
        conversations are logged but do not stop batch processing.

        Args:
            conversations: Conversations to extract from
            batch_size: Number of concurrent extractions

        Returns:
            List of ExtractionResult objects (one per input conversation)
        """
        results: list[ExtractionResult] = []

        for i in range(0, len(conversations), batch_size):
            batch = conversations[i : i + batch_size]
            logger.info(
                "Processing batch %d/%d (%d conversations)",
                (i // batch_size) + 1,
                (len(conversations) + batch_size - 1) // batch_size,
                len(batch),
            )

            for conversation in batch:
                try:
                    result = await self.extract(conversation)
                    results.append(result)
                except BudgetExceeded:
                    # Hard budget exceeded; stop processing
                    logger.error(
                        "Hard budget exceeded; stopping batch processing "
                        "(%d/%d conversations processed)",
                        len(results),
                        len(conversations),
                    )
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Error extracting from conversation %s: %s",
                        conversation.id,
                        exc,
                        exc_info=True,
                    )
                    # Log and continue
                    results.append(ExtractionResult(conversation_id=conversation.id))

        return results

    # =========================================================================
    # Private: Conversation formatting
    # =========================================================================

    def _build_conversation_text(self, conversation: Conversation) -> str:
        """Format conversation messages as "User: ...\nAssistant: ..." text.

        Truncates to max_conversation_tokens (estimated as chars / 4).

        Args:
            conversation: The conversation to format

        Returns:
            Formatted conversation text, truncated if necessary
        """
        lines: list[str] = []

        for msg in conversation.messages:
            role_name = msg.role.value.capitalize()
            lines.append(f"{role_name}: {msg.content}")

        full_text = "\n".join(lines)

        # Truncate by character count (rough token estimate: chars / 4)
        max_chars = self.max_conversation_tokens * 4
        if len(full_text) > max_chars:
            logger.warning(
                "Conversation %s truncated from %d to %d chars",
                conversation.id,
                len(full_text),
                max_chars,
            )
            full_text = full_text[:max_chars] + "\n[... truncated ...]"

        return full_text

    # =========================================================================
    # Private: Stage implementations
    # =========================================================================

    async def _extract_entities(
        self,
        conversation_hash: str,
        conversation_text: str,
        source: str,
        topic: str,
        message_count: int,
    ) -> Optional[list]:
        """Extract entities using the LLM.

        Uses caching and budget checks. Returns None if budget exceeded.

        Args:
            conversation_hash: SHA-256 hash of conversation text
            conversation_text: Formatted conversation text
            source: Platform name (e.g., "claude")
            topic: Conversation topic/title
            message_count: Number of messages

        Returns:
            List of ExtractedEntity objects, or None if skipped due to budget
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(
                conversation_hash=conversation_hash,
                stage="entity",
                prompt_version=entity_extraction.PROMPT_VERSION,
                model=self.provider.model_name,
                response_model=entity_extraction.LLMEntityExtraction,
            )
            if cached is not None:
                logger.debug(
                    "Cache hit for entity extraction (%s)",
                    conversation_hash[:12],
                )
                if self.cost_tracker:
                    self.cost_tracker.record(
                        cached.response,
                        stage="entity",
                        conversation_id=conversation_hash[:12],
                    )
                return cached.result.entities

        # Format prompt
        user_prompt = entity_extraction.USER_PROMPT_TEMPLATE.format(
            source=source,
            topic=topic,
            message_count=message_count,
            conversation_text=conversation_text,
        )

        # Estimate cost and check budget
        estimated_tokens_in = len(user_prompt) // 4 + len(
            entity_extraction.SYSTEM_PROMPT
        ) // 4
        estimated_tokens_out = 1000  # Conservative estimate
        estimated_cost = self.provider.estimate_cost(
            estimated_tokens_in, estimated_tokens_out
        )

        if self.cost_tracker:
            if not self.cost_tracker.can_afford(estimated_cost):
                logger.warning(
                    "Skipping entity extraction: insufficient budget "
                    "(need $%.4f, have $%.4f)",
                    estimated_cost,
                    self.cost_tracker.remaining,
                )
                return None
            if not self.cost_tracker.can_afford_conversation(estimated_cost):
                logger.warning(
                    "Skipping entity extraction: per-conversation limit exceeded "
                    "(need $%.4f, limit $%.4f)",
                    estimated_cost,
                    self.cost_tracker.budget.max_cost_per_conversation,
                )
                return None

        # Call LLM
        try:
            extraction, response = await self.provider.extract_structured(
                system_prompt=entity_extraction.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=entity_extraction.LLMEntityExtraction,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Entity extraction failed: %s",
                exc,
                exc_info=True,
            )
            raise

        # Cache and record
        if self.cache:
            self.cache.put(
                conversation_hash=conversation_hash,
                stage="entity",
                prompt_version=entity_extraction.PROMPT_VERSION,
                model=self.provider.model_name,
                result=extraction,
                response=response,
            )

        if self.cost_tracker:
            self.cost_tracker.record(
                response,
                stage="entity",
                conversation_id=conversation_hash[:12],
            )

        return extraction.entities

    async def _extract_relationships(
        self,
        conversation_hash: str,
        conversation_text: str,
        source: str,
        entities_json: str,
    ) -> Optional[list]:
        """Extract relationships between entities.

        Uses entities from stage 1 as input. Skipped if budget exceeded.

        Args:
            conversation_hash: SHA-256 hash of conversation text
            conversation_text: Formatted conversation text
            source: Platform name
            entities_json: JSON string of extracted entities

        Returns:
            List of ExtractedRelationship objects, or None if skipped
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(
                conversation_hash=conversation_hash,
                stage="relationship",
                prompt_version=relationship_mapping.PROMPT_VERSION,
                model=self.provider.model_name,
                response_model=relationship_mapping.LLMRelationshipExtraction,
            )
            if cached is not None:
                logger.debug(
                    "Cache hit for relationship extraction (%s)",
                    conversation_hash[:12],
                )
                if self.cost_tracker:
                    self.cost_tracker.record(
                        cached.response,
                        stage="relationship",
                        conversation_id=conversation_hash[:12],
                    )
                return cached.result.relationships

        # Format prompt
        user_prompt = relationship_mapping.USER_PROMPT_TEMPLATE.format(
            source=source,
            entities_json=entities_json,
            conversation_text=conversation_text,
        )

        # Estimate cost and check budget
        estimated_tokens_in = (
            len(user_prompt) // 4
            + len(relationship_mapping.SYSTEM_PROMPT) // 4
        )
        estimated_tokens_out = 500
        estimated_cost = self.provider.estimate_cost(
            estimated_tokens_in, estimated_tokens_out
        )

        if self.cost_tracker:
            if not self.cost_tracker.can_afford(estimated_cost):
                logger.warning(
                    "Skipping relationship extraction: insufficient budget"
                )
                return None
            if not self.cost_tracker.can_afford_conversation(estimated_cost):
                logger.warning(
                    "Skipping relationship extraction: per-conversation limit"
                )
                return None

        # Call LLM
        try:
            extraction, response = await self.provider.extract_structured(
                system_prompt=relationship_mapping.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=relationship_mapping.LLMRelationshipExtraction,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Relationship extraction failed: %s",
                exc,
                exc_info=True,
            )
            raise

        # Cache and record
        if self.cache:
            self.cache.put(
                conversation_hash=conversation_hash,
                stage="relationship",
                prompt_version=relationship_mapping.PROMPT_VERSION,
                model=self.provider.model_name,
                result=extraction,
                response=response,
            )

        if self.cost_tracker:
            self.cost_tracker.record(
                response,
                stage="relationship",
                conversation_id=conversation_hash[:12],
            )

        return extraction.relationships

    async def _classify_concepts(
        self,
        conversation_hash: str,
        conversation_text: str,
        source: str,
    ) -> Optional:
        """Classify conversation and extract high-level concepts.

        Skipped if budget exceeded.

        Args:
            conversation_hash: SHA-256 hash of conversation text
            conversation_text: Formatted conversation text
            source: Platform name

        Returns:
            ConversationClassification object, or None if skipped
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(
                conversation_hash=conversation_hash,
                stage="concept",
                prompt_version=concept_classification.PROMPT_VERSION,
                model=self.provider.model_name,
                response_model=concept_classification.ConversationClassification,
            )
            if cached is not None:
                logger.debug(
                    "Cache hit for concept classification (%s)",
                    conversation_hash[:12],
                )
                if self.cost_tracker:
                    self.cost_tracker.record(
                        cached.response,
                        stage="concept",
                        conversation_id=conversation_hash[:12],
                    )
                return cached.result

        # Format prompt
        user_prompt = concept_classification.USER_PROMPT_TEMPLATE.format(
            source=source,
            conversation_text=conversation_text,
        )

        # Estimate cost and check budget
        estimated_tokens_in = (
            len(user_prompt) // 4
            + len(concept_classification.SYSTEM_PROMPT) // 4
        )
        estimated_tokens_out = 800
        estimated_cost = self.provider.estimate_cost(
            estimated_tokens_in, estimated_tokens_out
        )

        if self.cost_tracker:
            if not self.cost_tracker.can_afford(estimated_cost):
                logger.warning(
                    "Skipping concept classification: insufficient budget"
                )
                return None
            if not self.cost_tracker.can_afford_conversation(estimated_cost):
                logger.warning(
                    "Skipping concept classification: per-conversation limit"
                )
                return None

        # Call LLM
        try:
            classification, response = await self.provider.extract_structured(
                system_prompt=concept_classification.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=concept_classification.ConversationClassification,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Concept classification failed: %s",
                exc,
                exc_info=True,
            )
            raise

        # Cache and record
        if self.cache:
            self.cache.put(
                conversation_hash=conversation_hash,
                stage="concept",
                prompt_version=concept_classification.PROMPT_VERSION,
                model=self.provider.model_name,
                result=classification,
                response=response,
            )

        if self.cost_tracker:
            self.cost_tracker.record(
                response,
                stage="concept",
                conversation_id=conversation_hash[:12],
            )

        return classification

    # =========================================================================
    # Private: Conversion functions (LLM schema → pipeline schema)
    # =========================================================================

    def _convert_entities(
        self, llm_entities: list, conversation_id: str
    ) -> list[Entity]:
        """Convert ExtractedEntity objects to Entity model.

        Args:
            llm_entities: List of ExtractedEntity from LLM
            conversation_id: ID of source conversation

        Returns:
            List of Entity objects
        """
        entities: list[Entity] = []

        for llm_entity in llm_entities:
            # Map LLM entity_type string to EntityType enum
            try:
                entity_type = EntityType(llm_entity.entity_type.lower())
            except ValueError:
                logger.warning(
                    "Unknown entity type %r; treating as TOOL",
                    llm_entity.entity_type,
                )
                entity_type = EntityType.TOOL

            entity = Entity(
                entity_type=entity_type,
                name=llm_entity.name,
                aliases=llm_entity.aliases,
                source_conversations=[conversation_id],
                properties={
                    "description": llm_entity.description,
                    "confidence": llm_entity.confidence,
                    "source_quotes": llm_entity.source_quotes,
                },
            )
            entities.append(entity)

        return entities

    def _convert_relationships(
        self, llm_relationships: list, conversation_id: str
    ) -> list[Relationship]:
        """Convert ExtractedRelationship objects to Relationship model.

        Args:
            llm_relationships: List of ExtractedRelationship from LLM
            conversation_id: ID of source conversation

        Returns:
            List of Relationship objects
        """
        relationships: list[Relationship] = []

        for llm_rel in llm_relationships:
            # Map LLM relationship type to RelationshipType enum
            # (LLM uses USES, CREATED_BY, etc.; pipeline uses MENTIONS, RELATES_TO, etc.)
            rel_type_map = {
                "USES": RelationshipType.RELATES_TO,
                "CREATED_BY": RelationshipType.RELATES_TO,
                "PART_OF": RelationshipType.RELATES_TO,
                "RELATED_TO": RelationshipType.RELATES_TO,
                "COMPARED_WITH": RelationshipType.RELATES_TO,
                "IMPLEMENTS": RelationshipType.RELATES_TO,
                "EXTENDS": RelationshipType.RELATES_TO,
            }
            rel_type = rel_type_map.get(
                llm_rel.predicate.upper(), RelationshipType.RELATES_TO
            )

            relationship = Relationship(
                source_id=llm_rel.subject,  # Would be resolved to entity ID in enrichment
                target_id=llm_rel.object,
                relationship_type=rel_type,
                weight=llm_rel.confidence,
                evidence=[llm_rel.evidence],
                source_conversation_id=conversation_id,
            )
            relationships.append(relationship)

        return relationships

    def _convert_concepts(
        self, classification, conversation_id: str
    ) -> list[Concept]:
        """Convert ConversationClassification to Concept objects.

        Creates concepts from:
        - Extracted concepts (with type inference)
        - Primary/secondary domains as topics
        - Purpose as a concept

        Args:
            classification: ConversationClassification from LLM
            conversation_id: ID of source conversation

        Returns:
            List of Concept objects
        """
        concepts: list[Concept] = []

        # Add classified concepts with proper typing
        for llm_concept in classification.concepts:
            # Infer concept type from domain context
            concept_type = ConceptType.TOPIC
            concepts.append(
                Concept(
                    concept_type=concept_type,
                    content=llm_concept.name,
                    context=llm_concept.description,
                    confidence=llm_concept.confidence,
                    source_conversation_id=conversation_id,
                    tags=[llm_concept.domain],
                )
            )

        # Add primary domain as a topic concept
        concepts.append(
            Concept(
                concept_type=ConceptType.TOPIC,
                content=f"Domain: {classification.primary_domain}",
                context=classification.summary,
                confidence=0.95,
                source_conversation_id=conversation_id,
                tags=["domain", "classification"],
            )
        )

        # Add secondary domains as topic concepts
        for secondary in classification.secondary_domains:
            concepts.append(
                Concept(
                    concept_type=ConceptType.TOPIC,
                    content=f"Domain: {secondary}",
                    confidence=0.85,
                    source_conversation_id=conversation_id,
                    tags=["domain", "secondary"],
                )
            )

        # Add purpose as an insight
        concepts.append(
            Concept(
                concept_type=ConceptType.INSIGHT,
                content=f"Purpose: {classification.purpose}",
                confidence=0.90,
                source_conversation_id=conversation_id,
                tags=["purpose", "classification"],
            )
        )

        # Add depth observation
        concepts.append(
            Concept(
                concept_type=ConceptType.INSIGHT,
                content=f"Depth: {classification.depth}",
                confidence=0.85,
                source_conversation_id=conversation_id,
                tags=["depth", "classification"],
            )
        )

        # Add summary as insight
        if classification.summary:
            concepts.append(
                Concept(
                    concept_type=ConceptType.INSIGHT,
                    content=classification.summary,
                    confidence=0.90,
                    source_conversation_id=conversation_id,
                    tags=["summary"],
                )
            )

        return concepts
