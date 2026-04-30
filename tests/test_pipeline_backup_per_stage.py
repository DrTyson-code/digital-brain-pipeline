"""Tests for vault backups around pipeline write stages."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models.base import Platform
from src.models.message import Conversation
from src.pipeline import CurationConfig, Pipeline, PipelineConfig
from src.process.extractor import ExtractionResult


def _config(vault_path: Path) -> PipelineConfig:
    return PipelineConfig(
        vault_path=vault_path,
        curation=CurationConfig(
            enable_source_scoring=False,
            enable_entity_resolution=False,
            enable_temporal_tracking=False,
            enable_contradiction_detection=False,
        ),
    )


def test_backup_fires_before_each_pipeline_write_stage(tmp_path, monkeypatch):
    import src.pipeline as pipeline_module

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "existing.md").write_text("seed", encoding="utf-8")
    config = _config(vault)
    events: list[str] = []
    conversation = Conversation(
        id="conv-1",
        title="Test",
        platform=Platform.CLAUDE,
        created_at=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(
        Pipeline,
        "_backup_vault",
        lambda self, stage_name: events.append(f"backup:{stage_name}") or tmp_path,
    )
    monkeypatch.setattr(
        Pipeline,
        "_ingest",
        lambda self, sources: [conversation],
    )

    class FakeClassifier:
        def __init__(self, *args, **kwargs):
            pass

        def classify_batch(self, conversations):
            return None

    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract_batch(self, conversations):
            return [ExtractionResult(conversation_id=conversation.id)]

    class FakeLinker:
        def link(self, conversations, extractions):
            return []

    class FakeEnricher:
        def __init__(self, *args, **kwargs):
            pass

        def enrich(self, extractions, relationships):
            return extractions, relationships

    class FakeReviewQueue:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return [], set()

        def load_corrections(self, vault_path):
            return []

    class FakeCrossDomainSynthesizer:
        def find_bridges(self, entities, concepts):
            return []

        def generate_synthesis_notes(self, bridges):
            return []

    class FakeObsidianWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write_all(self, *args, **kwargs):
            events.append("obsidian-write")
            return [Path("obsidian.md")]

    class FakeMOCGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate_all(self, *args, **kwargs):
            events.append("moc-write")
            return [Path("moc.md")]

    monkeypatch.setattr(pipeline_module, "ConversationClassifier", FakeClassifier)
    monkeypatch.setattr(pipeline_module, "EntityConceptExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline_module, "ObjectLinker", FakeLinker)
    monkeypatch.setattr(pipeline_module, "Enricher", FakeEnricher)
    monkeypatch.setattr(pipeline_module, "ReviewQueueGenerator", FakeReviewQueue)
    monkeypatch.setattr(
        pipeline_module, "CrossDomainSynthesizer", FakeCrossDomainSynthesizer
    )
    monkeypatch.setattr(pipeline_module, "ObsidianWriter", FakeObsidianWriter)
    monkeypatch.setattr(pipeline_module, "MOCGenerator", FakeMOCGenerator)

    result = Pipeline(config).run(source_files={"claude": tmp_path})

    assert events == [
        "backup:Stage 6 Obsidian write",
        "obsidian-write",
        "backup:Stage 7 MOC generation",
        "moc-write",
    ]
    assert result.written_files == [Path("obsidian.md"), Path("moc.md")]


def test_backup_absent_vault_logs_and_continues(tmp_path, caplog):
    pipeline = Pipeline(_config(tmp_path / "missing-vault"))

    with caplog.at_level(logging.INFO):
        backup_path = pipeline._backup_vault("Stage 6 Obsidian write")

    assert backup_path is None
    assert "Skipping vault backup before Stage 6 Obsidian write" in caplog.text


def test_backup_copies_existing_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "existing.md").write_text("seed", encoding="utf-8")

    backup_path = Pipeline(_config(vault))._backup_vault("Stage 6 Obsidian write")

    assert backup_path is not None
    assert backup_path.parent == tmp_path / ".backup"
    assert (backup_path / "existing.md").read_text(encoding="utf-8") == "seed"
