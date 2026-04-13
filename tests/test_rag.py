"""Tests for the RAG layer: ContextBuilder, HybridSearchEngine, and the FastAPI server."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from src.rag.context_builder import ContextBuilder
from src.search.hybrid_engine import (
    HybridResult,
    HybridSearchEngine,
    _bm25_score,
    _detect_note_type,
    _passes_filters,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_result(
    path: str = "/vault/concepts/test.md",
    title: str = "Test Note",
    score: float = 0.8,
    snippet: str = "This is a test note about clinical workflows.",
    note_type: str = "concept",
    domain: str = "medicine",
    tags: str = "clinical workflow",
) -> HybridResult:
    return HybridResult(
        note_path=path,
        title=title,
        score=score,
        bm25_score=score,
        vector_score=0.0,
        snippet=snippet,
        note_type=note_type,
        domain=domain,
        tags=tags,
        metadata={"title": title, "type": note_type},
    )


def _make_vault(tmp_path: Path, files: Dict[str, str]) -> Path:
    """Create a temp vault from {relative_path: content}."""
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in files.items():
        p = vault / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return vault


def _build_index(tmp_path: Path, vault: Path) -> Path:
    """Index a vault and return the DB path."""
    from src.search.embedder import NoteEmbedder

    db_path = tmp_path / "embeddings.db"
    indexer = NoteEmbedder(db_path=db_path)
    indexer.embed_vault(vault)
    indexer.close()
    return db_path


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class TestContextBuilder:
    def test_empty_results_returns_empty_string(self):
        assert ContextBuilder().build([]) == ""

    def test_includes_title(self):
        ctx = ContextBuilder().build([_make_result(title="Ketamine NMDA")])
        assert "Ketamine NMDA" in ctx

    def test_includes_query_header(self):
        ctx = ContextBuilder().build(
            [_make_result()], query="anesthesia workflow"
        )
        assert "anesthesia workflow" in ctx

    def test_includes_snippet(self):
        ctx = ContextBuilder().build(
            [_make_result(snippet="TIVA protocols for induction.")]
        )
        assert "TIVA protocols" in ctx

    def test_metadata_block_present_when_enabled(self):
        ctx = ContextBuilder(include_metadata=True).build(
            [_make_result(note_type="concept", domain="medicine")]
        )
        assert "concept" in ctx
        assert "medicine" in ctx
        assert "```yaml" in ctx

    def test_metadata_block_absent_when_disabled(self):
        ctx = ContextBuilder(include_metadata=False).build(
            [_make_result(note_type="concept", domain="medicine")]
        )
        assert "```yaml" not in ctx

    def test_respects_token_budget(self):
        builder = ContextBuilder(max_tokens=50)  # very tight budget
        results = [
            _make_result(
                path=f"/vault/{i}.md",
                title=f"Note {i}",
                snippet="word " * 200,
            )
            for i in range(10)
        ]
        ctx = builder.build(results)
        # Should be much shorter than all 10 full snippets
        assert len(ctx) < 50 * 4 * 5  # generous allowance

    def test_higher_scoring_results_appear_first(self):
        r_low = _make_result(title="Low", score=0.2, path="/vault/low.md")
        r_high = _make_result(title="High", score=0.9, path="/vault/high.md")
        # Pass in descending score order (engine guarantee)
        ctx = ContextBuilder().build([r_high, r_low])
        assert ctx.index("High") < ctx.index("Low")

    def test_source_filename_in_context(self):
        ctx = ContextBuilder(include_metadata=True).build(
            [_make_result(path="/vault/concepts/ketamine.md")]
        )
        assert "ketamine.md" in ctx

    def test_estimate_tokens_rough_approximation(self):
        # 4 chars ≈ 1 token
        assert ContextBuilder.estimate_tokens("abcd") == 1
        assert ContextBuilder.estimate_tokens("a" * 400) == 100

    def test_multiple_results_all_included_within_budget(self):
        results = [
            _make_result(
                path=f"/vault/{i}.md",
                title=f"Note {i}",
                snippet="Short snippet.",
                score=1.0 - i * 0.1,
            )
            for i in range(3)
        ]
        ctx = ContextBuilder(max_tokens=4000).build(results)
        assert "Note 0" in ctx
        assert "Note 1" in ctx
        assert "Note 2" in ctx


# ---------------------------------------------------------------------------
# HybridSearchEngine
# ---------------------------------------------------------------------------


class TestHybridSearchEngine:
    def test_load_raises_if_db_missing(self, tmp_path: Path):
        engine = HybridSearchEngine(db_path=tmp_path / "nonexistent.db")
        with pytest.raises(FileNotFoundError):
            engine.load()

    def test_load_returns_doc_count(self, tmp_path: Path):
        vault = _make_vault(
            tmp_path,
            {
                "a.md": "---\ntitle: Alpha\n---\nAlpha content about workflows.",
                "b.md": "---\ntitle: Beta\n---\nBeta content about protocols.",
            },
        )
        db = _build_index(tmp_path, vault)
        engine = HybridSearchEngine(db_path=db)
        assert engine.load() == 2

    def test_search_returns_relevant_result_first(self, tmp_path: Path):
        vault = _make_vault(
            tmp_path,
            {
                "workflow.md": (
                    "---\ntitle: Workflow Note\n---\n"
                    "Clinical workflow automation for anesthesia procedures."
                ),
                "other.md": (
                    "---\ntitle: Unrelated\n---\n"
                    "Weather patterns and seasonal forecasts."
                ),
            },
        )
        db = _build_index(tmp_path, vault)
        engine = HybridSearchEngine(db_path=db)
        engine.load()
        results = engine.search("clinical workflow anesthesia", top_k=5)
        assert results
        assert results[0].title == "Workflow Note"

    def test_search_before_load_raises_runtime_error(self, tmp_path: Path):
        vault = _make_vault(tmp_path, {"a.md": "content"})
        db = _build_index(tmp_path, vault)
        engine = HybridSearchEngine(db_path=db)
        with pytest.raises(RuntimeError, match="load()"):
            engine.search("query")

    def test_filter_by_note_type(self, tmp_path: Path):
        vault = _make_vault(
            tmp_path,
            {
                "concepts/concept_a.md": (
                    "---\ntitle: Concept A\ntype: concept\n---\n"
                    "A concept about medicine and pharmacology."
                ),
                "actions/action_b.md": (
                    "---\ntitle: Action B\ntype: action\n---\n"
                    "An action about medicine and protocols."
                ),
            },
        )
        db = _build_index(tmp_path, vault)
        engine = HybridSearchEngine(db_path=db)
        engine.load()
        results = engine.search("medicine", top_k=10, note_type="concept")
        for r in results:
            assert "concept" in r.note_type.lower()

    def test_stop_word_only_query_returns_empty(self, tmp_path: Path):
        vault = _make_vault(tmp_path, {"a.md": "---\ntitle: A\n---\nsome content"})
        db = _build_index(tmp_path, vault)
        engine = HybridSearchEngine(db_path=db)
        engine.load()
        assert engine.search("the and or is") == []

    def test_doc_count_property(self, tmp_path: Path):
        vault = _make_vault(
            tmp_path,
            {"a.md": "content a", "b.md": "content b", "c.md": "content c"},
        )
        db = _build_index(tmp_path, vault)
        engine = HybridSearchEngine(db_path=db)
        engine.load()
        assert engine.doc_count == 3

    def test_results_ordered_descending_by_score(self, tmp_path: Path):
        vault = _make_vault(
            tmp_path,
            {
                "high.md": (
                    "---\ntitle: High Relevance\n---\n"
                    "clinical workflow clinical workflow clinical workflow automation."
                ),
                "low.md": (
                    "---\ntitle: Low Relevance\n---\n"
                    "Barely related to workflows in a distant way."
                ),
            },
        )
        db = _build_index(tmp_path, vault)
        engine = HybridSearchEngine(db_path=db)
        engine.load()
        results = engine.search("clinical workflow automation", top_k=5)
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_result_includes_snippet_and_metadata(self, tmp_path: Path):
        vault = _make_vault(
            tmp_path,
            {
                "note.md": (
                    "---\ntitle: My Note\ntype: concept\ndomain: medicine\n"
                    "tags: anesthesia\n---\n"
                    "Propofol is used for total intravenous anesthesia."
                )
            },
        )
        db = _build_index(tmp_path, vault)
        engine = HybridSearchEngine(db_path=db)
        engine.load()
        results = engine.search("propofol anesthesia", top_k=1)
        assert results
        r = results[0]
        assert r.title == "My Note"
        assert r.note_type == "concept"
        assert r.domain == "medicine"
        assert "Propofol" in r.snippet


# ---------------------------------------------------------------------------
# BM25 scoring unit tests
# ---------------------------------------------------------------------------


class TestBm25Score:
    def test_zero_for_missing_term(self):
        assert (
            _bm25_score(["xyz"], {"hello": 0.5}, {"hello": 2.0}, dl=10, avg_dl=10)
            == 0.0
        )

    def test_positive_for_matching_term(self):
        score = _bm25_score(
            ["hello"], {"hello": 0.5, "world": 0.5}, {"hello": 2.0}, dl=10, avg_dl=10
        )
        assert score > 0.0

    def test_higher_tf_gives_higher_score(self):
        idf = {"clinical": 3.0}
        s_low = _bm25_score(["clinical"], {"clinical": 0.1}, idf, dl=10, avg_dl=10)
        s_high = _bm25_score(["clinical"], {"clinical": 0.5}, idf, dl=10, avg_dl=10)
        assert s_high > s_low

    def test_longer_doc_gets_lower_score(self):
        # Same raw count (1 occurrence) but different doc lengths.
        # Normalised TF = 1/dl, so shorter doc has higher tf_norm.
        # BM25 length normalisation should penalise the longer doc.
        idf = {"term": 2.0}
        # short doc: 1 occurrence in 5-word doc → tf_norm = 1/5 = 0.2
        s_short = _bm25_score(["term"], {"term": 0.2}, idf, dl=5, avg_dl=10)
        # long doc: 1 occurrence in 50-word doc → tf_norm = 1/50 = 0.02
        s_long = _bm25_score(["term"], {"term": 0.02}, idf, dl=50, avg_dl=10)
        assert s_short > s_long

    def test_multiple_query_terms_add_up(self):
        idf = {"clinical": 2.0, "workflow": 1.5}
        s_one = _bm25_score(
            ["clinical"], {"clinical": 0.3, "workflow": 0.3}, idf, dl=10, avg_dl=10
        )
        s_two = _bm25_score(
            ["clinical", "workflow"],
            {"clinical": 0.3, "workflow": 0.3},
            idf,
            dl=10,
            avg_dl=10,
        )
        assert s_two > s_one


# ---------------------------------------------------------------------------
# Filter helper tests
# ---------------------------------------------------------------------------


class TestPassesFilters:
    def test_no_filters_always_passes(self):
        assert _passes_filters({}, "/vault/a.md", None, None, None)

    def test_note_type_match(self):
        assert _passes_filters({"type": "concept"}, "/a.md", "concept", None, None)

    def test_note_type_mismatch(self):
        assert not _passes_filters({"type": "concept"}, "/a.md", "action", None, None)

    def test_domain_match(self):
        assert _passes_filters({"domain": "medicine"}, "/a.md", None, "medicine", None)

    def test_domain_mismatch(self):
        assert not _passes_filters(
            {"domain": "medicine"}, "/a.md", None, "finance", None
        )

    def test_tags_substring_match(self):
        assert _passes_filters(
            {"tags": "clinical workflow automation"}, "/a.md", None, None, "clinical"
        )

    def test_tags_mismatch(self):
        assert not _passes_filters(
            {"tags": "clinical workflow"}, "/a.md", None, None, "ketamine"
        )

    def test_all_filters_must_pass(self):
        meta = {"type": "concept", "domain": "medicine", "tags": "anesthesia"}
        # All three match
        assert _passes_filters(meta, "/a.md", "concept", "medicine", "anesthesia")
        # Domain fails
        assert not _passes_filters(meta, "/a.md", "concept", "finance", "anesthesia")


# ---------------------------------------------------------------------------
# FastAPI server integration tests
# ---------------------------------------------------------------------------


class TestRagServer:
    @pytest.fixture()
    def client(self, tmp_path: Path):
        """Spin up a TestClient backed by a real (small) vault + index."""
        import os

        from fastapi.testclient import TestClient

        from src.rag.server import app
        from src.search.embedder import NoteEmbedder

        vault = _make_vault(
            tmp_path,
            {
                "concepts/anesthesia.md": (
                    "---\ntitle: Anesthesia Workflow\ntype: concept\n"
                    "domain: medicine\ntags: anesthesia clinical\n---\n"
                    "Anesthesia workflows involve pre-operative assessment and "
                    "TIVA protocols for total intravenous anesthesia."
                ),
                "concepts/ketamine.md": (
                    "---\ntitle: Ketamine NMDA\ntype: concept\n"
                    "domain: medicine\ntags: ketamine pharmacology\n---\n"
                    "Ketamine is an NMDA receptor antagonist used for induction "
                    "and procedural sedation."
                ),
                "actions/morning_rounds.md": (
                    "---\ntitle: Morning Rounds\ntype: action\n---\n"
                    "Complete morning rounds and chart review before cases begin."
                ),
            },
        )
        db_path = tmp_path / "embeddings.db"
        indexer = NoteEmbedder(db_path=db_path)
        indexer.embed_vault(vault)
        indexer.close()

        os.environ["RAG_VAULT_PATH"] = str(vault)
        os.environ["RAG_DB_PATH"] = str(db_path)
        os.environ["RAG_CHROMA_PATH"] = ""  # disable vector search in tests

        with TestClient(app) as c:
            yield c

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["doc_count"] == 3

    def test_query_returns_results(self, client):
        resp = client.post(
            "/query", json={"query": "anesthesia workflow", "top_k": 3}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        titles = [r["title"] for r in data["results"]]
        assert "Anesthesia Workflow" in titles

    def test_query_empty_string_returns_400(self, client):
        resp = client.post("/query", json={"query": "   ", "top_k": 5})
        assert resp.status_code == 400

    def test_query_includes_context_string(self, client):
        resp = client.post("/query", json={"query": "ketamine NMDA", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data
        assert len(data["context"]) > 0

    def test_query_filter_note_type(self, client):
        resp = client.post(
            "/query",
            json={
                "query": "medicine",
                "top_k": 10,
                "filters": {"note_type": "action"},
            },
        )
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert "action" in r["note_type"].lower()

    def test_context_get_endpoint(self, client):
        resp = client.get("/context?q=anesthesia&top_k=2")
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_result_fields_present(self, client):
        resp = client.post(
            "/query", json={"query": "ketamine", "top_k": 1}
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert results
        r = results[0]
        for field in ("note_path", "title", "score", "snippet", "note_type", "metadata"):
            assert field in r, f"Missing field: {field}"

    def test_top_k_respected(self, client):
        resp = client.post("/query", json={"query": "medicine", "top_k": 2})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) <= 2
