"""Main pipeline orchestrator.

Coordinates the full ingest → classify → extract → link → enrich → output flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from src.ingest import ClaudeIngester, ChatGPTIngester, GeminiIngester
from src.models.base import Platform
from src.models.concept import Concept
from src.models.entity import Entity
from src.models.message import Conversation
from src.models.relationship import Relationship
from src.output.obsidian import ObsidianWriter
from src.output.graph import GraphExporter
from src.process.classifier import ConversationClassifier
from src.process.enricher import Enricher
from src.process.extractor import EntityConceptExtractor, ExtractionResult
from src.process.linker import ObjectLinker

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load configuration from a YAML settings file."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        vault_cfg = data.get("vault", {})
        ingest_cfg = data.get("ingest", {})
        proc_cfg = data.get("processing", {})
        out_cfg = data.get("output", {}).get("obsidian", {})

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
        )


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""

    conversations: List[Conversation] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    written_files: List[Path] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"Pipeline complete: {len(self.conversations)} conversations, "
            f"{len(self.entities)} entities, {len(self.concepts)} concepts, "
            f"{len(self.relationships)} relationships, "
            f"{len(self.written_files)} notes written"
        )


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
        3. Extract — pull entities and concepts from text
        4. Link    — build relationships between objects
        5. Enrich  — deduplicate and add metadata
        6. Output  — write to Obsidian vault (and optionally graph DB)
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

        # Stage 3: Extract
        logger.info("Stage 3: Extracting entities and concepts...")
        extractor = EntityConceptExtractor(
            confidence_threshold=self.config.confidence_threshold
        )
        extractions = extractor.extract_batch(conversations)

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

        # Stage 6: Output
        logger.info("Stage 6: Writing to Obsidian vault...")
        writer = ObsidianWriter(
            vault_path=self.config.vault_path,
            tag_prefix=self.config.tag_prefix,
            date_format=self.config.date_format,
            dataview_fields=self.config.dataview_fields,
        )
        written = writer.write_all(conversations, all_entities, all_concepts, relationships)

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
        )
        logger.info(result.summary)
        return result

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
