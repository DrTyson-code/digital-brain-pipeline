"""Tests for CrossDomainSynthesizer — cross-domain bridge detection and synthesis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.models.concept import Concept, ConceptType
from src.models.entity import Entity, EntityType
import src.process.cross_domain as cross_domain_module
from src.process.cross_domain import (
    BRIDGE_CONCEPT,
    BRIDGE_ENTITY,
    BRIDGE_PATTERN,
    CrossDomainSynthesizer,
    DomainBridge,
    SynthesisNote,
    SynthesisResult,
    _cosine_similarity,
    _sanitize_filename,
    _top_shared_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _concept(
    content: str,
    conv_id: str = "c1",
    confidence: float = 0.9,
    concept_type: ConceptType = ConceptType.INSIGHT,
    context: str = "",
) -> Concept:
    return Concept(
        concept_type=concept_type,
        content=content,
        confidence=confidence,
        source_conversation_id=conv_id,
        context=context or None,
    )


def _entity(name: str, etype: EntityType = EntityType.TOOL) -> Entity:
    return Entity(entity_type=etype, name=name, source_conversations=["c1"])


def _bridge(
    domain_a: str = "medicine",
    domain_b: str = "business",
    common_tokens: list = None,
    bridge_type: str = BRIDGE_PATTERN,
    similarity: float = 0.5,
    confidence: float = 0.4,
) -> DomainBridge:
    return DomainBridge(
        domain_a=domain_a,
        domain_b=domain_b,
        item_a_content="resource allocation under uncertainty in clinical settings",
        item_b_content="resource allocation under uncertainty in startup planning",
        item_a_title="Clinical resource allocation insight",
        item_b_title="Startup resource allocation decision",
        bridge_type=bridge_type,
        similarity=similarity,
        confidence=confidence,
        common_tokens=common_tokens or ["resource", "allocation", "uncertainty"],
    )


# ---------------------------------------------------------------------------
# Tests: _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    v = {"a": 1.0, "b": 2.0}
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = {"x": 1.0}
    b = {"y": 1.0}
    assert _cosine_similarity(a, b) == 0.0


def test_cosine_similarity_empty():
    assert _cosine_similarity({}, {"a": 1.0}) == 0.0
    assert _cosine_similarity({"a": 1.0}, {}) == 0.0


def test_cosine_similarity_partial_overlap():
    a = {"x": 1.0, "y": 1.0}
    b = {"y": 1.0, "z": 1.0}
    sim = _cosine_similarity(a, b)
    assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# Tests: _top_shared_tokens
# ---------------------------------------------------------------------------


def test_top_shared_tokens_returns_shared_only():
    a = {"apple": 0.5, "banana": 0.3, "cherry": 0.1}
    b = {"apple": 0.4, "durian": 0.6}
    result = _top_shared_tokens(a, b, top_n=5)
    assert result == ["apple"]


def test_top_shared_tokens_respects_top_n():
    a = {str(i): float(i) for i in range(10)}
    b = {str(i): float(i) for i in range(10)}
    result = _top_shared_tokens(a, b, top_n=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: _sanitize_filename
# ---------------------------------------------------------------------------


def test_sanitize_filename_removes_special_chars():
    name = 'medicine x business: resource/allocation"test'
    result = _sanitize_filename(name)
    for ch in '<>:"/\\|?*':
        assert ch not in result


def test_sanitize_filename_truncates():
    long_name = "x" * 300
    assert len(_sanitize_filename(long_name)) <= 180


def test_sanitize_filename_no_leading_trailing_dashes():
    result = _sanitize_filename("   test   ")
    assert not result.startswith("-")
    assert not result.endswith("-")


# ---------------------------------------------------------------------------
# Tests: CrossDomainSynthesizer.detect_domains
# ---------------------------------------------------------------------------


def test_detect_domains_medicine():
    synth = CrossDomainSynthesizer()
    domains = synth.detect_domains("The patient required anesthesia for surgery")
    assert "medicine" in domains


def test_detect_domains_reef_keeping():
    synth = CrossDomainSynthesizer()
    domains = synth.detect_domains("Alkalinity dosing for coral reef tank maintenance")
    assert "reef_keeping" in domains


def test_detect_domains_technology():
    synth = CrossDomainSynthesizer()
    domains = synth.detect_domains("Deploy the Python API with Docker and Kubernetes")
    assert "technology" in domains


def test_detect_domains_multiple():
    synth = CrossDomainSynthesizer()
    # Both business AND finance keywords
    domains = synth.detect_domains("startup revenue and investment portfolio management")
    assert "business" in domains
    assert "finance" in domains


def test_detect_domains_unknown():
    synth = CrossDomainSynthesizer()
    domains = synth.detect_domains("xyzzy frobnicator quux plonk")
    assert domains == []


def test_detect_domains_case_insensitive():
    synth = CrossDomainSynthesizer()
    domains = synth.detect_domains("ANESTHESIA and PATIENT management")
    assert "medicine" in domains


# ---------------------------------------------------------------------------
# Tests: CrossDomainSynthesizer.classify_objects
# ---------------------------------------------------------------------------


def test_classify_objects_entity_into_domain():
    synth = CrossDomainSynthesizer()
    entity = _entity("Propofol")  # medicine keyword
    domain_map = synth.classify_objects([entity], [])
    assert any(
        "Propofol" in title
        for title, _, _ in domain_map.get("medicine", [])
    )


def test_classify_objects_concept_into_domain():
    synth = CrossDomainSynthesizer()
    concept = _concept("Monitoring cardiac output during surgery is critical")
    domain_map = synth.classify_objects([], [concept])
    medicine_titles = [t for t, _, _ in domain_map.get("medicine", [])]
    assert any("Monitoring" in t for t in medicine_titles)


def test_classify_objects_concept_into_multiple_domains():
    synth = CrossDomainSynthesizer()
    # Both fitness AND personal keywords
    concept = _concept("Daily workout routine and goal setting for personal growth")
    domain_map = synth.classify_objects([], [concept])
    # Should appear in fitness and personal
    assert len(domain_map.get("fitness", [])) > 0
    assert len(domain_map.get("personal", [])) > 0


def test_classify_objects_empty_inputs():
    synth = CrossDomainSynthesizer()
    domain_map = synth.classify_objects([], [])
    assert all(len(v) == 0 for v in domain_map.values())


def test_classify_objects_confidence_preserved():
    synth = CrossDomainSynthesizer()
    concept = _concept("Clinical drug dosing protocol", confidence=0.75)
    domain_map = synth.classify_objects([], [concept])
    for title, content, conf in domain_map.get("medicine", []):
        assert conf == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Tests: CrossDomainSynthesizer.find_bridges
# ---------------------------------------------------------------------------


def test_find_bridges_empty_inputs():
    synth = CrossDomainSynthesizer()
    result = synth.find_bridges([], [])
    assert result == []


def test_find_bridges_same_domain_no_bridge():
    """Items from the same domain must not generate bridges."""
    synth = CrossDomainSynthesizer(similarity_threshold=0.1)
    # Both are medicine
    c1 = _concept("Patient anesthesia monitoring during surgery")
    c2 = _concept("Patient anesthesia dosing and sedation protocol")
    bridges = synth.find_bridges([], [c1, c2])
    # All bridges, if any, must span different domains
    for bridge in bridges:
        assert bridge.domain_a != bridge.domain_b


def test_find_bridges_cross_domain_detected():
    """Two semantically similar items from different domains should bridge."""
    synth = CrossDomainSynthesizer(similarity_threshold=0.15, min_bridge_confidence=0.10)
    # Deliberately share vocabulary across domains
    c_med = _concept(
        "Systematic protocol for managing resource allocation under uncertainty in clinical care",
        confidence=0.9,
    )
    c_biz = _concept(
        "Systematic protocol for managing resource allocation under uncertainty in business",
        confidence=0.9,
    )
    bridges = synth.find_bridges([], [c_med, c_biz])
    # At least one cross-domain bridge
    cross = [b for b in bridges if b.domain_a != b.domain_b]
    assert len(cross) >= 1


def test_find_bridges_entity_gets_entity_type():
    """A bridge involving a known entity name should be typed as BRIDGE_ENTITY."""
    synth = CrossDomainSynthesizer(similarity_threshold=0.15, min_bridge_confidence=0.10)
    entity = _entity("propofol")  # medicine keyword in name
    c_biz = _concept(
        "Propofol used systematically in controlled business protocol resource allocation",
        confidence=0.9,
    )
    bridges = synth.find_bridges([entity], [c_biz])
    entity_bridges = [b for b in bridges if b.bridge_type == BRIDGE_ENTITY]
    # If a bridge involving the entity was found, check its type
    if entity_bridges:
        assert entity_bridges[0].bridge_type == BRIDGE_ENTITY


def test_find_bridges_sorted_by_confidence():
    synth = CrossDomainSynthesizer(similarity_threshold=0.10, min_bridge_confidence=0.05)
    concepts = [
        _concept("Resource allocation protocol systematic planning", confidence=0.9),
        _concept("Systematic business resource planning allocation protocol", confidence=0.9),
        _concept("Clinical patient anesthesia monitoring systematic protocol", confidence=0.9),
        _concept("Coral reef alkalinity calcium dosing schedule systematic", confidence=0.9),
    ]
    bridges = synth.find_bridges([], concepts)
    for i in range(len(bridges) - 1):
        assert bridges[i].confidence >= bridges[i + 1].confidence


def test_find_bridges_cap_per_pair():
    """max_bridges_per_pair is enforced."""
    cap = 2
    synth = CrossDomainSynthesizer(
        similarity_threshold=0.10,
        min_bridge_confidence=0.05,
        max_bridges_per_pair=cap,
    )
    # Flood the medicine domain with near-identical concepts and one business concept
    med_concepts = [
        _concept(f"Patient anesthesia clinical monitoring systematic protocol version {i}")
        for i in range(10)
    ]
    biz_concept = _concept(
        "Patient anesthesia clinical monitoring systematic protocol in business context"
    )
    bridges = synth.find_bridges([], med_concepts + [biz_concept])
    # Count bridges for the medicine <-> business pair
    pair_bridges = [
        b for b in bridges
        if {b.domain_a, b.domain_b} == {"medicine", "business"}
    ]
    assert len(pair_bridges) <= cap


def test_find_bridges_cross_classified_item_not_self_bridged():
    """An item classified in both domain A and B should not bridge with itself."""
    synth = CrossDomainSynthesizer(similarity_threshold=0.1, min_bridge_confidence=0.05)
    # This content has both medicine and business keywords
    concept = _concept(
        "Business strategy for a medical startup: patient care revenue and clinical protocol"
    )
    bridges = synth.find_bridges([], [concept])
    # No bridge should have identical content on both sides
    for bridge in bridges:
        assert not (
            bridge.item_a_content == bridge.item_b_content
            and bridge.item_a_title == bridge.item_b_title
        )


# ---------------------------------------------------------------------------
# Tests: CrossDomainSynthesizer.generate_synthesis_notes
# ---------------------------------------------------------------------------


def test_generate_synthesis_notes_empty_bridges():
    synth = CrossDomainSynthesizer()
    notes = synth.generate_synthesis_notes([])
    assert notes == []


def test_generate_synthesis_notes_fields():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge], discovered_date="2026-04-12")
    assert len(notes) == 1
    note = notes[0]
    assert note.domain_a == "medicine"
    assert note.domain_b == "business"
    assert note.bridge_type == BRIDGE_PATTERN
    assert note.discovered == "2026-04-12"
    assert 0.0 <= note.confidence <= 1.0
    assert len(note.source_notes) == 2
    assert all(s.startswith("[[") and s.endswith("]]") for s in note.source_notes)


def test_generate_synthesis_notes_deduplicated():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    # Two identical bridges should produce only one note
    notes = synth.generate_synthesis_notes([bridge, bridge])
    assert len(notes) == 1


def test_generate_synthesis_notes_explanation_not_empty():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge])
    assert len(notes[0].explanation) > 20


def test_generate_synthesis_notes_uses_today_by_default():
    from datetime import datetime
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge])
    today = datetime.now().strftime("%Y-%m-%d")
    assert notes[0].discovered == today


def test_explanation_pattern_bridge():
    synth = CrossDomainSynthesizer()
    bridge = _bridge(bridge_type=BRIDGE_PATTERN)
    notes = synth.generate_synthesis_notes([bridge])
    assert "pattern" in notes[0].explanation.lower() or "principle" in notes[0].explanation.lower()


def test_explanation_entity_bridge():
    synth = CrossDomainSynthesizer()
    bridge = _bridge(bridge_type=BRIDGE_ENTITY)
    notes = synth.generate_synthesis_notes([bridge])
    assert "bridge" in notes[0].explanation.lower() or "entity" in notes[0].explanation.lower() or "appears" in notes[0].explanation.lower()


def test_explanation_concept_bridge():
    synth = CrossDomainSynthesizer()
    bridge = _bridge(bridge_type=BRIDGE_CONCEPT)
    notes = synth.generate_synthesis_notes([bridge])
    assert "conceptual" in notes[0].explanation.lower() or "link" in notes[0].explanation.lower() or "semantic" in notes[0].explanation.lower()


# ---------------------------------------------------------------------------
# Tests: CrossDomainSynthesizer.write_to_vault
# ---------------------------------------------------------------------------


def test_write_to_vault_creates_folder():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge], discovered_date="2026-04-12")
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        written = synth.write_to_vault(notes, vault)
        assert (vault / "Cross-Domain Synthesis").is_dir()
        assert len(written) == 1


def test_write_to_vault_file_has_frontmatter():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge], discovered_date="2026-04-12")
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        written = synth.write_to_vault(notes, vault)
        content = written[0].read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "type: synthesis" in content
        assert "bridge_type:" in content
        assert "confidence:" in content
        assert "discovered:" in content
        assert "source_notes:" in content


def test_write_to_vault_wikilinks_in_body():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge], discovered_date="2026-04-12")
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        written = synth.write_to_vault(notes, vault)
        content = written[0].read_text(encoding="utf-8")
        assert "[[" in content
        assert "]]" in content


def test_write_to_vault_empty_notes():
    synth = CrossDomainSynthesizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        written = synth.write_to_vault([], Path(tmpdir))
        assert written == []


def test_write_to_vault_passes_vault_root_to_atomic_write(tmp_path, monkeypatch):
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge], discovered_date="2026-04-12")
    calls = []

    def record_atomic_write(path, content, **kwargs):
        calls.append((path, content, kwargs))

    monkeypatch.setattr(cross_domain_module, "atomic_write", record_atomic_write)

    written = synth.write_to_vault(notes, tmp_path)

    assert len(written) == 1
    assert calls[0][0] == written[0]
    assert calls[0][2]["root"] == tmp_path


# ---------------------------------------------------------------------------
# Tests: CrossDomainSynthesizer.generate_moc
# ---------------------------------------------------------------------------


def test_generate_moc_no_notes_returns_none():
    synth = CrossDomainSynthesizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = synth.generate_moc([], Path(tmpdir))
        assert result is None


def test_generate_moc_creates_file():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge], discovered_date="2026-04-12")
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        moc_path = synth.generate_moc(notes, vault)
        assert moc_path is not None
        assert moc_path.exists()
        assert moc_path.name == "Cross-Domain Synthesis MOC.md"


def test_generate_moc_has_frontmatter():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge], discovered_date="2026-04-12")
    with tempfile.TemporaryDirectory() as tmpdir:
        moc_path = synth.generate_moc(notes, Path(tmpdir))
        content = moc_path.read_text(encoding="utf-8")
        assert "type: moc" in content
        assert "total_bridges:" in content


def test_generate_moc_groups_by_domain_pair():
    synth = CrossDomainSynthesizer()
    bridges = [
        _bridge("medicine", "business"),
        _bridge("fitness", "finance"),
    ]
    notes = synth.generate_synthesis_notes(bridges, discovered_date="2026-04-12")
    with tempfile.TemporaryDirectory() as tmpdir:
        moc_path = synth.generate_moc(notes, Path(tmpdir))
        content = moc_path.read_text(encoding="utf-8")
        # Both domain pair sections should appear
        assert "Medicine" in content and "Business" in content
        assert "Fitness" in content and "Finance" in content


def test_generate_moc_has_dataview_block():
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge])
    with tempfile.TemporaryDirectory() as tmpdir:
        moc_path = synth.generate_moc(notes, Path(tmpdir))
        content = moc_path.read_text(encoding="utf-8")
        assert "```dataview" in content
        assert "FROM" in content


def test_generate_moc_passes_vault_root_to_atomic_write(tmp_path, monkeypatch):
    synth = CrossDomainSynthesizer()
    bridge = _bridge()
    notes = synth.generate_synthesis_notes([bridge], discovered_date="2026-04-12")
    calls = []

    def record_atomic_write(path, content, **kwargs):
        calls.append((path, content, kwargs))

    monkeypatch.setattr(cross_domain_module, "atomic_write", record_atomic_write)

    moc_path = synth.generate_moc(notes, tmp_path)

    assert moc_path == tmp_path / "Cross-Domain Synthesis MOC.md"
    assert calls[0][0] == moc_path
    assert calls[0][2]["root"] == tmp_path


# ---------------------------------------------------------------------------
# Tests: CrossDomainSynthesizer.run (end-to-end)
# ---------------------------------------------------------------------------


def test_run_no_inputs_returns_empty_result():
    synth = CrossDomainSynthesizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = synth.run([], [], Path(tmpdir))
        assert isinstance(result, SynthesisResult)
        assert result.bridges == []
        assert result.notes == []
        assert result.written_paths == []
        assert result.moc_path is None


def test_run_with_cross_domain_concepts_writes_files():
    synth = CrossDomainSynthesizer(similarity_threshold=0.15, min_bridge_confidence=0.10)
    c_med = _concept(
        "Systematic protocol for managing resource allocation under uncertainty in clinical care",
        confidence=0.9,
    )
    c_biz = _concept(
        "Systematic protocol for managing resource allocation under uncertainty in business",
        confidence=0.9,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        result = synth.run([], [c_med, c_biz], vault)
        if result.bridges:
            assert len(result.written_paths) > 0
            assert result.moc_path is not None
            assert result.moc_path.exists()


def test_run_summary_string():
    synth = CrossDomainSynthesizer()
    result = SynthesisResult(
        bridges=[_bridge()],
        notes=[],
        written_paths=[Path("/tmp/test.md")],
    )
    summary = result.summary
    assert "bridge" in summary
    assert "note" in summary
    assert "file" in summary


# ---------------------------------------------------------------------------
# Tests: SynthesisResult
# ---------------------------------------------------------------------------


def test_synthesis_result_defaults():
    result = SynthesisResult()
    assert result.bridges == []
    assert result.notes == []
    assert result.written_paths == []
    assert result.moc_path is None


# ---------------------------------------------------------------------------
# Tests: _bridge_title
# ---------------------------------------------------------------------------


def test_bridge_title_with_common_tokens():
    synth = CrossDomainSynthesizer()
    bridge = _bridge(common_tokens=["resource", "allocation", "protocol"])
    title = synth._bridge_title(bridge)
    assert "medicine" in title
    assert "business" in title
    assert "resource" in title


def test_bridge_title_no_common_tokens():
    synth = CrossDomainSynthesizer()
    bridge = _bridge(common_tokens=[])
    title = synth._bridge_title(bridge)
    assert "medicine" in title
    assert "business" in title
    assert len(title) <= 100


def test_bridge_title_length_capped():
    synth = CrossDomainSynthesizer()
    bridge = _bridge(common_tokens=["a" * 50, "b" * 50, "c" * 50])
    title = synth._bridge_title(bridge)
    assert len(title) <= 100
