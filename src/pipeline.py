"""Main pipeline orchestrator.

Coordinates the full ingest → classify → extract → link → enrich → output flow.

Supports four extraction modes:
- rules_only: Phase 1 behavior, no LLM calls (free)
- llm_augmented: Run both and merge results (default, recommended)
- llm_primary: LLM first, rule-based fills gaps on failure
- llm_only: Skip rule-based extraction entirely
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.ingest import ClaudeIngester, ChatGPTIngester, GeminiIngester
from src.models.base import Platform
from src.models.concept import Concept
from src.models.entity import Entity
from src.models.message import Conversation
from src.models.relationship import Relationship
from src.output.obsidian import ObsidianWriter
from src.output.graph import GraphExporter
from src.output.moc import MOCGenerator
from src.process.classifier import ConversationClassifier
from src.process.contradiction_detector import ContradictionDetector
from src.process.enricher import Enricher
from src.process.entity_resolver import EntityResolver
from src.process.extractor import EntityConceptExtractor, ExtractionResult
from src.process.linker import ObjectLinker
from src.process.cross_domain import CrossDomainSynthesizer
from src.process.review_queue import CurationResult, ReviewQueueGenerator
from src.export.anki import AnkiCardGenerator, AnkiExporter
from src.export.anki_scheduler import AnkiScheduler
from src.process.source_scorer import SourceScorer
from src.process.temporal_tracker import TemporalTracker

logger = logging.getLogger(__name__)


class ExtractionMode(str, Enum):
    """How to run the extraction stage."""

    RULES_ONLY = "rules_only"
    LLM_AUGMENTED = "llm_augmented"
    LLM_PRIMARY = "llm_primary"
    LLM_ONLY = "llm_only"


@dataclass
class LLMConfig:
    """Configuration for the LLM extraction layer."""

    extraction_mode: ExtractionMode = ExtractionMode.RULES_ONLY
    provider_name: str = "claude"
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_retries: int = 2
    timeout_seconds: int = 30
    max_cost_per_run_usd: float = 1.00
    max_cost_per_conversation_usd: float = 0.05
    merge_similarity_threshold: float = 0.85
    max_conversation_tokens: int = 8000
    batch_size: int = 10
    enable_cache: bool = True
    cache_path: Path = Path(".dbp_cache/")

    @classmethod
    def from_dict(cls, data: dict) -> LLMConfig:
        """Load from the llm section of a config dict."""
        if not data:
            return cls()

        provider = data.get("provider", {})
        budget = data.get("budget", {})
        quality = data.get("quality", {})
        processing = data.get("processing", {})

        mode_str = data.get("extraction_mode", "rules_only")
        try:
            mode = ExtractionMode(mode_str)
        except ValueError:
            logger.warning("Unknown extraction_mode %r; using rules_only", mode_str)
            mode = ExtractionMode.RULES_ONLY

        return cls(
            extraction_mode=mode,
            provider_name=provider.get("name", "claude"),
            model=provider.get("model"),
            api_key=provider.get("api_key"),
            api_key_env=provider.get("api_key_env", "ANTHROPIC_API_KEY"),
            base_url=provider.get("base_url"),
            temperature=provider.get("temperature", 0.0),
            max_retries=provider.get("max_retries", 2),
            timeout_seconds=provider.get("timeout_seconds", 30),
            max_cost_per_run_usd=budget.get("max_cost_per_run_usd", 1.00),
            max_cost_per_conversation_usd=budget.get("max_cost_per_conversation_usd", 0.05),
            merge_similarity_threshold=quality.get("merge_similarity_threshold", 0.85),
            max_conversation_tokens=processing.get("max_conversation_tokens", 8000),
            batch_size=processing.get("batch_size", 10),
            enable_cache=processing.get("enable_cache", True),
            cache_path=Path(processing.get("cache_path", ".dbp_cache/")),
        )


@dataclass
class CurationConfig:
    """Configuration for the Knowledge Curation Layer (stages 4a–4e)."""

    enable_source_scoring: bool = True
    enable_entity_resolution: bool = True
    enable_temporal_tracking: bool = True
    enable_contradiction_detection: bool = True
    min_review_confidence: float = 0.6
    entity_similarity_threshold: float = 0.85

    @property
    def any_enabled(self) -> bool:
        return any([
            self.enable_source_scoring,
            self.enable_entity_resolution,
            self.enable_temporal_tracking,
            self.enable_contradiction_detection,
        ])

    @classmethod
    def from_dict(cls, data: dict) -> CurationConfig:
        if not data:
            return cls()
        return cls(
            enable_source_scoring=data.get("enable_source_scoring", True),
            enable_entity_resolution=data.get("enable_entity_resolution", True),
            enable_temporal_tracking=data.get("enable_temporal_tracking", True),
            enable_contradiction_detection=data.get(
                "enable_contradiction_detection", True
            ),
            min_review_confidence=data.get("min_review_confidence", 0.6),
            entity_similarity_threshold=data.get("entity_similarity_threshold", 0.85),
        )


@dataclass
class AnkiConfig:
    """Configuration for the Anki export stage (Stage 9)."""

    enabled: bool = False
    output_path: Path = Path("data/anki_export.txt")
    format: str = "tsv"
    batch_size: int = 50
    stale_only: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> AnkiConfig:
        if not data:
            return cls()
        return cls(
            enabled=data.get("enabled", False),
            output_path=Path(data.get("output_path", "data/anki_export.txt")),
            format=data.get("format", "tsv"),
            batch_size=data.get("batch_size", 50),
            stale_only=data.get("stale_only", False),
        )


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    vault_path: Path = Path("~/Vault")
    source_dirs: Dict[str, Path] = field(default_factory=dict)
    min_messages: int = 2
    confidence_threshold: float = 0.5
    max_topics: int = 5
    deduplicate_entities: bool = True
    tag_prefix: str = "ai-brain"
    date_format: str = "%Y-%m-%d"
    dataview_fields: bool = True
    export_graph: bool = False
    graph_output_path: Path = Path("output/graph.json")
    generate_mocs: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    anki: AnkiConfig = field(default_factory=AnkiConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load configuration from a YAML settings file."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        vault_cfg = data.get("vault", {})
        ingest_cfg = data.get("ingest", {})
        proc_cfg = data.get("processing", {})
        out_cfg = data.get("output", {}).get("obsidian", {})
        llm_data = data.get("llm", {})
        curation_data = data.get("curation", {})
        anki_data = data.get("anki", {})

        source_dirs = {}
        for platform, dir_path in ingest_cfg.get("sources", {}).items():
            source_dirs[platform] = Path(dir_path).expanduser()

        return cls(
            vault_path=Path(vault_cfg.get("path", "~/Vault")).expanduser(),
            source_dirs=source_dirs,
            min_messages=ingest_cfg.get("min_messages", 2),
            confidence_threshold=proc_cfg.get("confidence_threshold", 0.5),
            max_topics=proc_cfg.get("max_topics_per_conversation", 5),
            deduplicate_entities=proc_cfg.get("deduplicate_entities", True),
            tag_prefix=out_cfg.get("tag_prefix", "ai-brain"),
            date_format=out_cfg.get("date_format", "%Y-%m-%d"),
            dataview_fields=out_cfg.get("dataview_fields", True),
            llm=LLMConfig.from_dict(llm_data),
            curation=CurationConfig.from_dict(curation_data),
            anki=AnkiConfig.from_dict(anki_data),
        )


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""

    conversations: List[Conversation] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    written_files: List[Path] = field(default_factory=list)
    cost_report: Optional[str] = None

    @property
    def summary(self) -> str:
        s = (
            f"Pipeline complete: {len(self.conversations)} conversations, "
            f"{len(self.entities)} entities, {len(self.concepts)} concepts, "
            f"{len(self.relationships)} relationships, "
            f"{len(self.written_files)} notes written"
        )
        if self.cost_report:
            s += f"\n{self.cost_report}"
        return s


# Map platform keys to ingester classes
INGESTERS = {
    "claude": ClaudeIngester,
    "chatgpt": ChatGPTIngester,
    "gemini": GeminiIngester,
}


class Pipeline:
    """Orchestrate the full Digital Brain Pipeline.

    Stages:
        1. Ingest  — parse platform exports into Conversation objects
        2. Classify — assign topics to each conversation
        3. Extract — pull entities and concepts from text (rule-based + optional LLM)
        4. Link    — build relationships between objects
        5. Enrich  — deduplicate and add metadata
        6. Output  — write to Obsidian vault (and optionally graph DB)
        7. MOCs    — generate Maps of Content index notes
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        source_files: dict[str, Path] | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            source_files: Optional override mapping platform → file/dir path.
                          Falls back to config source_dirs.
        """
        sources = source_files or self.config.source_dirs

        # Stage 1: Ingest
        logger.info("Stage 1: Ingesting conversations...")
        conversations = self._ingest(sources)
        logger.info("Ingested %d conversations", len(conversations))

        if not conversations:
            logger.warning("No conversations found. Exiting pipeline.")
            return PipelineResult()

        # Stage 2: Classify
        logger.info("Stage 2: Classifying conversations...")
        classifier = ConversationClassifier(max_topics=self.config.max_topics)
        classifier.classify_batch(conversations)

        # Stage 3: Extract (rule-based)
        logger.info("Stage 3: Extracting entities and concepts...")
        extractor = EntityConceptExtractor(
            confidence_threshold=self.config.confidence_threshold
        )

        mode = self.config.llm.extraction_mode

        if mode == ExtractionMode.LLM_ONLY:
            # Skip rule-based entirely
            rule_extractions = [
                ExtractionResult(conversation_id=c.id) for c in conversations
            ]
        else:
            rule_extractions = extractor.extract_batch(conversations)

        # Stage 3b: LLM extraction (if enabled)
        llm_extractions = None
        cost_report_str = None

        if mode != ExtractionMode.RULES_ONLY:
            logger.info("Stage 3b: LLM extraction (mode=%s)...", mode.value)
            llm_extractions, cost_report_str = self._run_llm_extraction(conversations)

        # Stage 3c: Merge if we have both sources
        if llm_extractions is not None:
            logger.info("Stage 3c: Merging rule-based and LLM extractions...")
            from src.llm.merger import ExtractionMerger

            merger = ExtractionMerger(
                similarity_threshold=self.config.llm.merge_similarity_threshold
            )
            extractions = merger.merge_batch(rule_extractions, llm_extractions)
        else:
            extractions = rule_extractions

        # Stage 4: Link
        logger.info("Stage 4: Linking objects...")
        linker = ObjectLinker()
        relationships = linker.link(conversations, extractions)

        # Stage 5: Enrich
        logger.info("Stage 5: Enriching and deduplicating...")
        enricher = Enricher(deduplicate=self.config.deduplicate_entities)
        extractions, relationships = enricher.enrich(extractions, relationships)

        # Flatten entities and concepts
        all_entities = [e for ex in extractions for e in ex.entities]
        all_concepts = [c for ex in extractions for c in ex.concepts]

        # ---------------------------------------------------------------
        # Stages 5a–5e: Knowledge Curation Layer
        # ---------------------------------------------------------------
        curation = CurationResult()

        # Stage 5a: Source quality scoring
        if self.config.curation.enable_source_scoring:
            logger.info("Stage 5a: Source quality scoring...")
            scorer = SourceScorer()
            curation.source_weights = scorer.score_batch(conversations, extractions)

        # Stage 5b: Corpus-level fuzzy entity resolution
        if self.config.curation.enable_entity_resolution:
            logger.info("Stage 5b: Corpus entity resolution...")
            resolver = EntityResolver(
                similarity_threshold=self.config.curation.entity_similarity_threshold
            )
            extractions, curation.merge_map = resolver.resolve(extractions)
            # Re-flatten after resolution (may have changed entity set)
            all_entities = [e for ex in extractions for e in ex.entities]
            # Remap any relationships that pointed at now-merged entity IDs
            if curation.merge_map:
                relationships = resolver.remap_relationships(
                    relationships, curation.merge_map
                )

        # Stage 5c: Temporal status tracking
        if self.config.curation.enable_temporal_tracking:
            logger.info("Stage 5c: Temporal status tracking...")
            tracker = TemporalTracker()
            curation.concept_statuses = tracker.track(all_concepts, conversations)

        # Stage 5d: Contradiction detection
        if self.config.curation.enable_contradiction_detection:
            logger.info("Stage 5d: Contradiction detection...")
            detector = ContradictionDetector()
            curation.contradictions = detector.detect(all_concepts)
            relationships.extend(curation.contradictions)

        # Stage 5e: Review queue generation
        logger.info("Stage 5e: Building review queue...")
        review_gen = ReviewQueueGenerator(
            min_review_confidence=self.config.curation.min_review_confidence
        )
        curation.review_items, curation.review_ids = review_gen.generate(
            all_entities,
            all_concepts,
            curation.contradictions,
            curation.merge_map or None,
            curation.concept_statuses or None,
        )
        # Load any user corrections from the vault
        if self.config.vault_path.exists():
            curation.corrections = review_gen.load_corrections(self.config.vault_path)

        # Stage 8: Cross-domain synthesis
        logger.info("Stage 8: Cross-domain synthesis...")
        synthesizer = CrossDomainSynthesizer()
        bridges = synthesizer.find_bridges(all_entities, all_concepts)
        curation.synthesis_notes = synthesizer.generate_synthesis_notes(bridges)

        # Stage 9: Anki export
        if self.config.anki.enabled:
            self._run_anki_export(all_concepts, curation)

        # Stage 6: Output
        logger.info("Stage 6: Writing to Obsidian vault...")
        writer = ObsidianWriter(
            vault_path=self.config.vault_path,
            tag_prefix=self.config.tag_prefix,
            date_format=self.config.date_format,
            dataview_fields=self.config.dataview_fields,
        )
        written = writer.write_all(
            conversations, all_entities, all_concepts, relationships,
            curation=curation,
        )

        # Stage 7: MOCs
        if self.config.generate_mocs:
            logger.info("Stage 7: Generating Maps of Content...")
            moc_gen = MOCGenerator(
                vault_path=self.config.vault_path,
                tag_prefix=self.config.tag_prefix,
                date_format=self.config.date_format,
            )
            moc_files = moc_gen.generate_all(
                conversations, all_entities, all_concepts, relationships,
                curation=curation,
            )
            written.extend(moc_files)

        # Optional graph export
        if self.config.export_graph:
            logger.info("Exporting graph...")
            graph_exporter = GraphExporter()
            graph_exporter.export(
                conversations, all_entities, all_concepts, relationships,
                self.config.graph_output_path,
            )

        result = PipelineResult(
            conversations=conversations,
            entities=all_entities,
            concepts=all_concepts,
            relationships=relationships,
            written_files=written,
            cost_report=cost_report_str,
        )
        logger.info(result.summary)
        return result

    def _run_anki_export(
        self, concepts: list[Concept], curation: CurationResult
    ) -> None:
        """Run Anki export (Stage 9)."""
        anki_cfg = self.config.anki
        logger.info("Stage 9: Anki export...")

        scheduler = AnkiScheduler(batch_size=anki_cfg.batch_size)
        generator = AnkiCardGenerator()
        exporter = AnkiExporter()

        stale_ids: set[str] | None = None
        if anki_cfg.stale_only and curation.concept_statuses:
            stale_ids = {
                cid for cid, st in curation.concept_statuses.items()
                if st.status == "stale"
            }

        cards = generator.generate_from_concepts(
            concepts,
            stale_ids=stale_ids,
            concept_statuses=curation.concept_statuses or None,
        )
        cards += generator.generate_from_synthesis_notes(curation.synthesis_notes or [])

        new_cards = scheduler.filter_new(cards)
        if not new_cards:
            logger.info("Anki export: no new cards to export")
            return

        output_path = anki_cfg.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = scheduler.start_run(anki_cfg.format, output_path)
        exporter.export(new_cards, output_path, fmt=anki_cfg.format)
        scheduler.mark_exported_batch(new_cards)
        scheduler.complete_run(run_id, len(new_cards))
        logger.info("Anki export: wrote %d cards to %s", len(new_cards), output_path)

    def _run_llm_extraction(
        self, conversations: list[Conversation]
    ) -> tuple[list[ExtractionResult] | None, str | None]:
        """Run LLM extraction on conversations.

        Returns (extractions, cost_report_str) or (None, None) on failure.
        """
        try:
            from src.llm.provider import ProviderConfig, create_provider
            from src.llm.cost import CostTracker, TokenBudget
            from src.llm.cache import ExtractionCache
            from src.llm.extractor import LLMExtractor
        except ImportError as exc:
            logger.error(
                "LLM dependencies not installed (pip install anthropic openai httpx): %s", exc
            )
            return None, None

        llm_cfg = self.config.llm

        # Create provider
        provider_config = ProviderConfig(
            provider=llm_cfg.provider_name,
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            api_key_env=llm_cfg.api_key_env,
            base_url=llm_cfg.base_url,
            temperature=llm_cfg.temperature,
            max_retries=llm_cfg.max_retries,
            timeout_seconds=llm_cfg.timeout_seconds,
        )

        try:
            provider = create_provider(provider_config)
        except Exception as exc:
            logger.error("Failed to create LLM provider: %s", exc)
            return None, None

        # Create cost tracker
        budget = TokenBudget(
            max_cost_usd=llm_cfg.max_cost_per_run_usd,
            max_cost_per_conversation=llm_cfg.max_cost_per_conversation_usd,
        )
        cost_tracker = CostTracker(budget)

        # Create cache
        cache = None
        if llm_cfg.enable_cache:
            cache = ExtractionCache(cache_path=llm_cfg.cache_path / "extractions.db")

        # Create extractor
        llm_extractor = LLMExtractor(
            provider=provider,
            cache=cache,
            cost_tracker=cost_tracker,
            max_conversation_tokens=llm_cfg.max_conversation_tokens,
        )

        # Run extraction (async → sync bridge)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    llm_extractor.extract_batch(
                        conversations, batch_size=llm_cfg.batch_size
                    )
                )
            finally:
                loop.close()
        except Exception as exc:
            logger.error("LLM extraction failed: %s", exc)
            if cache:
                cache.close()
            return None, None

        # Get cost report
        report = cost_tracker.report()
        cost_str = str(report)
        logger.info("LLM extraction complete:\n%s", cost_str)

        if cache:
            cache.close()

        return results, cost_str

    def _ingest(self, sources: dict[str, Path]) -> list[Conversation]:
        """Run all ingesters on their respective source directories."""
        all_conversations: list[Conversation] = []

        for platform_key, source_path in sources.items():
            source_path = Path(source_path).expanduser()
            if not source_path.exists():
                logger.warning(
                    "Source path does not exist: %s (platform: %s)",
                    source_path,
                    platform_key,
                )
                continue

            ingester_cls = INGESTERS.get(platform_key)
            if ingester_cls is None:
                logger.warning("Unknown platform: %s", platform_key)
                continue

            ingester = ingester_cls(min_messages=self.config.min_messages)
            conversations = ingester.ingest(source_path)
            all_conversations.extend(conversations)

        return all_conversations
