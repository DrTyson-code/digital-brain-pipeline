"""Comprehensive tests for the Anki export module.

Covers:
- AnkiCard construction and TSV formatting
- Card generation from Concept objects (basic, cloze, stale, high-confidence)
- Card generation from SynthesisNote objects
- Vault note scanning
- Domain filtering
- Batch size limiting
- AnkiExporter TSV output
- AnkiScheduler: deduplication, incremental export, run history
- Dry-run mode
- Deck assignment by domain
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest
import yaml

from src.export.anki import (
    AnkiCard,
    AnkiCardGenerator,
    AnkiExporter,
    DECK_PREFIX,
    GENANKI_AVAILABLE,
    _detect_domain,
    _domain_to_deck,
    _try_cloze_deletion,
    _strip_cloze_markers,
    _parse_frontmatter,
)
from src.export.anki_scheduler import AnkiScheduler
from src.models.base import Platform
from src.models.concept import Concept, ConceptType
from src.models.message import ChatMessage, Conversation, Role
from src.process.cross_domain import SynthesisNote
from src.process.temporal_tracker import ConceptStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _concept(
    content: str,
    confidence: float = 0.9,
    concept_type: ConceptType = ConceptType.INSIGHT,
    context: str = "",
    tags: list = None,
) -> Concept:
    return Concept(
        concept_type=concept_type,
        content=content,
        confidence=confidence,
        context=context or None,
        tags=tags or [],
    )


def _decision(content: str, confidence: float = 0.9) -> Concept:
    return _concept(content, confidence, ConceptType.DECISION)


def _action(content: str, confidence: float = 0.9) -> Concept:
    return _concept(content, confidence, ConceptType.ACTION_ITEM)


def _synthesis_note(
    title: str = "tech x medicine: monitoring",
    domain_a: str = "technology",
    domain_b: str = "medicine",
    explanation: str = "Monitoring patterns appear in both domains.",
    common_tokens: list = None,
) -> SynthesisNote:
    return SynthesisNote(
        title=title,
        domain_a=domain_a,
        domain_b=domain_b,
        bridge_type="pattern",
        confidence=0.6,
        similarity=0.4,
        source_notes=["[[Note A]]", "[[Note B]]"],
        common_tokens=common_tokens or ["monitoring", "alerts"],
        explanation=explanation,
        discovered="2024-01-15",
    )


def _make_vault_note(
    tmp_path: Path,
    title: str,
    note_type: str = "concept",
    confidence: float = 0.9,
    status: str = "active",
    domain_tags: list = None,
    body: str = "This is the note content for the flashcard back.",
    filename: str = None,
) -> Path:
    """Write a minimal vault note with YAML frontmatter to tmp_path."""
    tags = domain_tags or ["ai-brain"]
    frontmatter = {
        "title": title,
        "type": note_type,
        "confidence": confidence,
        "status": status,
        "tags": tags,
    }
    fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
    content = f"---\n{fm_str}---\n\n{body}\n"

    name = filename or f"{title.replace(' ', '_')}.md"
    note_path = tmp_path / name
    note_path.write_text(content, encoding="utf-8")
    return note_path


def _make_scheduler(tmp_path: Path) -> AnkiScheduler:
    return AnkiScheduler(db_path=tmp_path / "test_anki.db", batch_size=50)


# ---------------------------------------------------------------------------
# AnkiCard tests
# ---------------------------------------------------------------------------


class TestAnkiCard:
    def test_tsv_row_basic(self):
        card = AnkiCard(
            front="What is propofol?",
            back="A sedative agent used for induction.",
            card_type="basic",
            tags=["ai-brain", "domain::medicine"],
        )
        row = card.tsv_row
        parts = row.split("\t")
        assert len(parts) == 3
        assert parts[0] == "What is propofol?"
        assert "sedative" in parts[1]
        assert "ai-brain" in parts[2]

    def test_tsv_row_cloze_strips_markers_from_front(self):
        card = AnkiCard(
            front="{{c1::Propofol}} is a sedative used for induction.",
            back="",
            card_type="cloze",
            tags=["ai-brain"],
        )
        row = card.tsv_row
        front_col = row.split("\t")[0]
        assert "{{c1::" not in front_col
        assert "Propofol" in front_col

    def test_tsv_row_cloze_keeps_markup_in_back(self):
        cloze_text = "{{c1::Propofol}} is a sedative."
        card = AnkiCard(
            front=cloze_text,
            back="",
            card_type="cloze",
            tags=[],
        )
        row = card.tsv_row
        back_col = row.split("\t")[1]
        assert "{{c1::Propofol}}" in back_col

    def test_sanitized_tags_replace_slash(self):
        card = AnkiCard(
            front="Q", back="A", card_type="basic",
            tags=["ai-brain/synthesis", "domain::medicine"]
        )
        sanitized = card.sanitized_tags
        assert "ai-brain::synthesis" in sanitized
        assert "domain::medicine" in sanitized

    def test_content_hash_stable(self):
        card = AnkiCard(front="Q", back="A", card_type="basic")
        assert card.content_hash == card.content_hash
        card2 = AnkiCard(front="Q", back="A", card_type="basic")
        assert card.content_hash == card2.content_hash

    def test_content_hash_differs_on_content_change(self):
        card1 = AnkiCard(front="Q", back="A", card_type="basic")
        card2 = AnkiCard(front="Q", back="B", card_type="basic")
        assert card1.content_hash != card2.content_hash

    def test_default_deck_is_prefix(self):
        card = AnkiCard(front="Q", back="A", card_type="basic")
        assert card.deck == DECK_PREFIX


# ---------------------------------------------------------------------------
# Domain detection and deck assignment
# ---------------------------------------------------------------------------


class TestDomainDetection:
    def test_detects_medicine_from_content(self):
        domain = _detect_domain("The patient received propofol for sedation")
        assert domain == "medicine"

    def test_detects_technology_from_content(self):
        domain = _detect_domain("We decided to use Python and Docker for the project")
        assert domain == "technology"

    def test_detects_domain_from_tag(self):
        domain = _detect_domain("some content", tags=["ai-brain/medicine"])
        assert domain == "medicine"

    def test_falls_back_to_general(self):
        domain = _detect_domain("This is a note with no recognizable domain content xyz")
        assert domain == "general"

    def test_domain_to_deck_medicine(self):
        assert _domain_to_deck("medicine") == "DigitalBrain::Medicine"

    def test_domain_to_deck_technology(self):
        assert _domain_to_deck("technology") == "DigitalBrain::Technology"

    def test_domain_to_deck_general(self):
        assert _domain_to_deck("general") == "DigitalBrain::General"


# ---------------------------------------------------------------------------
# Cloze deletion helpers
# ---------------------------------------------------------------------------


class TestCloze:
    def test_cloze_for_is_a_pattern(self):
        result = _try_cloze_deletion("Propofol is a sedative used for induction of anesthesia.")
        assert result is not None
        assert "{{c1::Propofol}}" in result
        assert "sedative" in result

    def test_cloze_for_decided_to_pattern(self):
        result = _try_cloze_deletion("We decided to use PostgreSQL for the main database.")
        assert result is not None
        assert "{{c1::" in result

    def test_cloze_returns_none_for_short_text(self):
        result = _try_cloze_deletion("Too short")
        assert result is None

    def test_strip_cloze_markers(self):
        text = "{{c1::Propofol}} is a {{c2::sedative}}."
        stripped = _strip_cloze_markers(text)
        assert stripped == "Propofol is a sedative."
        assert "{{" not in stripped


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_parses_valid_frontmatter(self):
        raw = "---\ntitle: Test Note\nconfidence: 0.9\n---\n\nBody text here."
        fm, body = _parse_frontmatter(raw)
        assert fm is not None
        assert fm["title"] == "Test Note"
        assert fm["confidence"] == 0.9
        assert "Body text here" in body

    def test_returns_none_for_no_frontmatter(self):
        raw = "# Heading\n\nBody without frontmatter."
        fm, body = _parse_frontmatter(raw)
        assert fm is None

    def test_returns_none_for_malformed_yaml(self):
        raw = "---\nkey: [unclosed\n---\n\nBody"
        fm, body = _parse_frontmatter(raw)
        assert fm is None


# ---------------------------------------------------------------------------
# AnkiCardGenerator — from concepts
# ---------------------------------------------------------------------------


class TestGenerateFromConcepts:
    def test_generates_card_from_high_confidence_concept(self):
        concepts = [_concept("Anesthesia monitoring is critical for patient safety", confidence=0.9)]
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts(concepts)
        assert len(cards) == 1

    def test_skips_low_confidence_non_stale_concept(self):
        concepts = [_concept("Some insight", confidence=0.5)]
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts(concepts)
        assert len(cards) == 0

    def test_stale_concept_included_despite_low_confidence(self):
        c = _concept("Old insight", confidence=0.5)
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts(concepts=[c], stale_ids={c.id})
        assert len(cards) == 1

    def test_stale_card_has_urgent_tag(self):
        c = _concept("Outdated decision", confidence=0.9)
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c], stale_ids={c.id})
        assert any("review::urgent" in card.tags for card in cards)

    def test_non_stale_card_lacks_urgent_tag(self):
        c = _concept("A fresh high confidence insight", confidence=0.9)
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c], stale_ids=set())
        assert not any("review::urgent" in card.tags for card in cards)

    def test_stale_cards_sorted_first(self):
        c1 = _concept("Fresh insight", confidence=0.9)
        c2 = _concept("Stale insight", confidence=0.9)
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c1, c2], stale_ids={c2.id})
        urgent_positions = [i for i, c in enumerate(cards) if "review::urgent" in c.tags]
        assert urgent_positions[0] == 0  # stale card is first

    def test_decision_concept_has_meaningful_front(self):
        c = _decision("We decided to use PostgreSQL as the primary database")
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c])
        assert len(cards) == 1
        front = cards[0].front
        assert "decided" in front.lower() or "PostgreSQL" in front or "decision" in front.lower()

    def test_question_concept_front_is_the_question(self):
        c = _concept("How do we handle anesthesia for pediatric patients?", concept_type=ConceptType.QUESTION)
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c])
        assert len(cards) == 1
        assert "pediatric" in cards[0].front

    def test_skips_superseded_non_stale_concept(self):
        c = _concept("Old decision about storage", confidence=0.9, concept_type=ConceptType.DECISION)
        status = ConceptStatus(concept_id=c.id, status="superseded")
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c], concept_statuses={c.id: status})
        assert len(cards) == 0

    def test_domain_filter_excludes_non_matching(self):
        c1 = _concept("Python is a programming language used for automation.")
        c2 = _concept("Propofol is used for induction of general anesthesia.")
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c1, c2], domain_filter=["medicine"])
        # Only the medicine concept should pass
        assert all("medicine" in card.tags[1] for card in cards)

    def test_source_id_is_concept_id(self):
        c = _concept("Insight about monitoring", confidence=0.9)
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c])
        assert cards[0].source_id == c.id

    def test_source_type_is_concept(self):
        c = _concept("Some insight", confidence=0.9)
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c])
        assert cards[0].source_type == "concept"

    def test_deck_assigned_from_domain(self):
        c = _concept("Propofol is used for general anesthesia induction.", confidence=0.9)
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_concepts([c])
        assert "Medicine" in cards[0].deck or "DigitalBrain" in cards[0].deck


# ---------------------------------------------------------------------------
# AnkiCardGenerator — from synthesis notes
# ---------------------------------------------------------------------------


class TestGenerateFromSynthesisNotes:
    def test_generates_card_from_synthesis_note(self):
        note = _synthesis_note()
        gen = AnkiCardGenerator()
        cards = gen.generate_from_synthesis_notes([note])
        assert len(cards) == 1

    def test_synthesis_card_front_mentions_domains(self):
        note = _synthesis_note(domain_a="technology", domain_b="medicine")
        gen = AnkiCardGenerator()
        cards = gen.generate_from_synthesis_notes([note])
        front = cards[0].front.lower()
        assert "technology" in front or "medicine" in front

    def test_synthesis_card_back_is_explanation(self):
        note = _synthesis_note(explanation="Both fields use feedback loops for optimization.")
        gen = AnkiCardGenerator()
        cards = gen.generate_from_synthesis_notes([note])
        assert "feedback" in cards[0].back

    def test_synthesis_card_deck_is_crossdomain(self):
        note = _synthesis_note()
        gen = AnkiCardGenerator()
        cards = gen.generate_from_synthesis_notes([note])
        assert "CrossDomain" in cards[0].deck

    def test_synthesis_card_source_type(self):
        note = _synthesis_note()
        gen = AnkiCardGenerator()
        cards = gen.generate_from_synthesis_notes([note])
        assert cards[0].source_type == "synthesis"

    def test_synthesis_domain_filter(self):
        notes = [
            _synthesis_note(domain_a="technology", domain_b="medicine"),
            _synthesis_note(title="business x fitness", domain_a="business", domain_b="fitness"),
        ]
        gen = AnkiCardGenerator()
        cards = gen.generate_from_synthesis_notes(notes, domain_filter=["medicine"])
        assert len(cards) == 1

    def test_synthesis_front_uses_common_tokens(self):
        note = _synthesis_note(common_tokens=["monitoring", "alerts", "thresholds"])
        gen = AnkiCardGenerator()
        cards = gen.generate_from_synthesis_notes([note])
        assert "monitoring" in cards[0].front


# ---------------------------------------------------------------------------
# AnkiCardGenerator — from vault
# ---------------------------------------------------------------------------


class TestGenerateFromVault:
    def test_generates_card_from_vault_note(self, tmp_path):
        _make_vault_note(
            tmp_path, "Propofol Mechanism",
            note_type="concept",
            confidence=0.9,
            body="Propofol acts on GABA receptors to produce sedation and hypnosis.",
        )
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_vault(tmp_path)
        assert len(cards) == 1

    def test_skips_low_confidence_vault_note(self, tmp_path):
        _make_vault_note(
            tmp_path, "Low confidence note",
            confidence=0.4,
            status="active",
        )
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_vault(tmp_path)
        assert len(cards) == 0

    def test_stale_vault_note_included_despite_low_confidence(self, tmp_path):
        _make_vault_note(
            tmp_path, "Stale note",
            confidence=0.3,
            status="stale",
            body="This is an old concept that needs review.",
        )
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_vault(tmp_path)
        assert len(cards) == 1
        assert "review::urgent" in cards[0].tags

    def test_stale_only_excludes_active_notes(self, tmp_path):
        _make_vault_note(tmp_path, "Active note", status="active", confidence=0.9,
                         body="Active content for review.", filename="active.md")
        _make_vault_note(tmp_path, "Stale note", status="stale", confidence=0.9,
                         body="Stale content for review.", filename="stale.md")
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_vault(tmp_path, stale_only=True)
        assert len(cards) == 1
        assert "review::urgent" in cards[0].tags

    def test_domain_filter_for_vault_notes(self, tmp_path):
        _make_vault_note(
            tmp_path, "Python Pattern",
            confidence=0.9,
            domain_tags=["ai-brain/technology"],
            body="Python is a programming language used for data science.",
            filename="python.md",
        )
        _make_vault_note(
            tmp_path, "Propofol Use",
            confidence=0.9,
            domain_tags=["ai-brain/medicine"],
            body="Propofol is used for induction of anesthesia.",
            filename="propofol.md",
        )
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_vault(tmp_path, domain_filter=["medicine"])
        assert len(cards) == 1

    def test_skips_notes_without_frontmatter(self, tmp_path):
        no_fm = tmp_path / "no_frontmatter.md"
        no_fm.write_text("# Just a heading\n\nSome body text.", encoding="utf-8")
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_vault(tmp_path)
        assert len(cards) == 0

    def test_nonexistent_vault_returns_empty(self, tmp_path):
        gen = AnkiCardGenerator()
        cards = gen.generate_from_vault(tmp_path / "nonexistent")
        assert cards == []

    def test_vault_source_type_is_vault_note(self, tmp_path):
        _make_vault_note(tmp_path, "Test Note", confidence=0.9,
                         body="Content for testing source type.")
        gen = AnkiCardGenerator(min_confidence=0.7)
        cards = gen.generate_from_vault(tmp_path)
        assert cards[0].source_type == "vault_note"


# ---------------------------------------------------------------------------
# AnkiExporter — TSV
# ---------------------------------------------------------------------------


class TestAnkiExporter:
    def test_to_tsv_writes_file(self, tmp_path):
        cards = [
            AnkiCard(front="What is Python?", back="A programming language.", card_type="basic", tags=["tech"]),
        ]
        output = tmp_path / "export.txt"
        exporter = AnkiExporter()
        written = exporter.to_tsv(cards, output)
        assert written.exists()

    def test_to_tsv_header_contains_directives(self, tmp_path):
        cards = [AnkiCard(front="Q", back="A", card_type="basic")]
        output = tmp_path / "export.txt"
        AnkiExporter().to_tsv(cards, output)
        content = output.read_text()
        assert "#separator:tab" in content
        assert "#notetype:Basic" in content

    def test_to_tsv_contains_card_content(self, tmp_path):
        cards = [AnkiCard(front="What is propofol?", back="A sedative.", card_type="basic", tags=["medicine"])]
        output = tmp_path / "export.txt"
        AnkiExporter().to_tsv(cards, output)
        content = output.read_text()
        assert "What is propofol?" in content
        assert "sedative" in content

    def test_to_tsv_dry_run_no_file(self, tmp_path):
        cards = [AnkiCard(front="Q", back="A", card_type="basic")]
        output = tmp_path / "dry_export.txt"
        AnkiExporter().to_tsv(cards, output, dry_run=True)
        assert not output.exists()

    def test_export_fallback_to_tsv_without_genanki(self, tmp_path):
        """export() with fmt=apkg falls back to tsv if genanki unavailable."""
        cards = [AnkiCard(front="Q", back="A", card_type="basic")]
        output = tmp_path / "test.apkg"
        exporter = AnkiExporter()
        if not GENANKI_AVAILABLE:
            written = exporter.export(cards, output, fmt="apkg")
            # Should have written a .txt file
            assert written.suffix == ".txt"
        else:
            # If genanki IS available, just verify it works without error
            written = exporter.export(cards, output, fmt="apkg")
            assert written.exists()

    def test_export_creates_parent_dirs(self, tmp_path):
        cards = [AnkiCard(front="Q", back="A", card_type="basic")]
        output = tmp_path / "nested" / "deep" / "export.txt"
        AnkiExporter().to_tsv(cards, output)
        assert output.exists()

    def test_batch_size_limits_cards(self, tmp_path):
        """Scheduler batch_size caps the number of exported cards."""
        concepts = [
            _concept(f"Insight number {i}", confidence=0.9)
            for i in range(100)
        ]
        gen = AnkiCardGenerator(min_confidence=0.7)
        scheduler = _make_scheduler(tmp_path)
        scheduler.batch_size = 10
        all_cards = gen.generate_from_concepts(concepts)
        filtered = scheduler.filter_new(all_cards)
        assert len(filtered) == 10


# ---------------------------------------------------------------------------
# AnkiScheduler — dedup and incremental export
# ---------------------------------------------------------------------------


class TestAnkiScheduler:
    def test_filter_new_returns_all_on_first_run(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        cards = [
            AnkiCard(front="Q1", back="A1", card_type="basic", source_id="id-1", source_type="concept"),
            AnkiCard(front="Q2", back="A2", card_type="basic", source_id="id-2", source_type="concept"),
        ]
        result = scheduler.filter_new(cards)
        assert len(result) == 2

    def test_filter_new_excludes_previously_exported(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        card = AnkiCard(
            front="Q", back="A", card_type="basic",
            source_id="id-already-done", source_type="concept",
        )
        scheduler.mark_exported("id-already-done", "concept", content_hash=card.content_hash)
        result = scheduler.filter_new([card])
        assert len(result) == 0

    def test_filter_new_includes_changed_content(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        old_hash = "oldhash123"
        scheduler.mark_exported("id-changed", "concept", content_hash=old_hash)
        card = AnkiCard(
            front="Q new content", back="A updated", card_type="basic",
            source_id="id-changed", source_type="concept",
        )
        # card.content_hash != old_hash → should appear as new
        result = scheduler.filter_new([card])
        assert len(result) == 1

    def test_mark_exported_records_entry(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        scheduler.mark_exported("src-1", "concept")
        assert scheduler.is_exported("src-1", "concept")

    def test_is_exported_false_before_marking(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        assert not scheduler.is_exported("unknown-id", "concept")

    def test_mark_exported_batch(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        cards = [
            AnkiCard(front="Q1", back="A1", card_type="basic", source_id="a", source_type="concept"),
            AnkiCard(front="Q2", back="A2", card_type="basic", source_id="b", source_type="concept"),
        ]
        scheduler.mark_exported_batch(cards)
        assert scheduler.is_exported("a", "concept")
        assert scheduler.is_exported("b", "concept")

    def test_exported_count(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        scheduler.mark_exported("x1", "concept")
        scheduler.mark_exported("x2", "synthesis")
        assert scheduler.exported_count() == 2

    def test_start_and_complete_run(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        run_id = scheduler.start_run("tsv", Path("data/export.txt"))
        scheduler.complete_run(run_id, card_count=42)
        history = scheduler.export_history()
        assert len(history) == 1
        assert history[0]["card_count"] == 42
        assert history[0]["format"] == "tsv"
        assert history[0]["run_id"] == run_id

    def test_export_history_sorted_by_recency(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        r1 = scheduler.start_run("tsv", Path("out1.txt"))
        scheduler.complete_run(r1, 5)
        r2 = scheduler.start_run("tsv", Path("out2.txt"))
        scheduler.complete_run(r2, 10)
        history = scheduler.export_history()
        # Most recent first
        assert history[0]["run_id"] == r2

    def test_reset_clears_history(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        scheduler.mark_exported("x", "concept")
        scheduler.start_run("tsv", Path("out.txt"))
        scheduler.reset()
        assert scheduler.exported_count() == 0
        assert len(scheduler.export_history()) == 0

    def test_batch_size_respected_in_filter_new(self, tmp_path):
        scheduler = AnkiScheduler(db_path=tmp_path / "test.db", batch_size=3)
        cards = [
            AnkiCard(front=f"Q{i}", back="A", card_type="basic",
                     source_id=f"id-{i}", source_type="concept")
            for i in range(10)
        ]
        result = scheduler.filter_new(cards)
        assert len(result) == 3

    def test_cards_without_source_id_always_included(self, tmp_path):
        scheduler = _make_scheduler(tmp_path)
        card = AnkiCard(front="Q", back="A", card_type="basic", source_id="")
        result = scheduler.filter_new([card])
        assert len(result) == 1

    def test_frequency_gate_suppresses_changed_content(self, tmp_path):
        scheduler = AnkiScheduler(
            db_path=tmp_path / "test.db",
            batch_size=50,
            export_frequency_days=7,
        )
        # Mark as exported just now
        scheduler.mark_exported("freq-id", "concept", content_hash="oldhash")
        # A card with changed content but within 7-day window
        card = AnkiCard(
            front="Q updated", back="A new", card_type="basic",
            source_id="freq-id", source_type="concept",
        )
        result = scheduler.filter_new([card])
        assert len(result) == 0

    def test_incremental_export_workflow(self, tmp_path):
        """Simulate two export runs: second run only exports new cards."""
        scheduler = _make_scheduler(tmp_path)
        gen = AnkiCardGenerator(min_confidence=0.7)
        exporter = AnkiExporter()

        # First run: 3 concepts
        batch_1 = [_concept(f"Insight {i}", confidence=0.9) for i in range(3)]
        cards_1 = gen.generate_from_concepts(batch_1)
        new_1 = scheduler.filter_new(cards_1)
        exporter.to_tsv(new_1, tmp_path / "run1.txt")
        scheduler.mark_exported_batch(new_1)

        assert len(new_1) == 3

        # Second run: same 3 + 2 new concepts
        batch_2 = batch_1 + [_concept(f"New insight {i}", confidence=0.9) for i in range(2)]
        cards_2 = gen.generate_from_concepts(batch_2)
        new_2 = scheduler.filter_new(cards_2)
        # Only the 2 truly new concepts should be exported
        assert len(new_2) == 2


# ---------------------------------------------------------------------------
# Full integration: generate + schedule + export
# ---------------------------------------------------------------------------


class TestFullIntegration:
    def test_full_pipeline_with_vault_notes(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()

        # Stale note (should be included even at low confidence)
        _make_vault_note(
            vault, "Old Anesthesia Protocol",
            status="stale", confidence=0.5,
            body="The old protocol required higher doses of fentanyl.",
        )
        # High-confidence note
        _make_vault_note(
            vault, "Propofol Mechanism",
            status="active", confidence=0.95,
            body="Propofol potentiates GABA-A receptors causing sedation.",
        )
        # Below-threshold note (should be skipped)
        _make_vault_note(
            vault, "Vague observation",
            status="active", confidence=0.4,
            body="Something interesting happened today.",
            filename="vague.md",
        )

        scheduler = AnkiScheduler(db_path=tmp_path / "anki.db", batch_size=50)
        gen = AnkiCardGenerator(min_confidence=0.7)
        exporter = AnkiExporter()
        output = tmp_path / "full_export.txt"

        cards = gen.generate_from_vault(vault)
        assert len(cards) == 2  # stale + high-conf; vague skipped

        new_cards = scheduler.filter_new(cards)
        written = exporter.to_tsv(new_cards, output)
        scheduler.mark_exported_batch(new_cards)

        assert written.exists()
        content = written.read_text()
        assert "#notetype:Basic" in content

        # Re-run: nothing new
        cards_2 = gen.generate_from_vault(vault)
        new_cards_2 = scheduler.filter_new(cards_2)
        assert len(new_cards_2) == 0
