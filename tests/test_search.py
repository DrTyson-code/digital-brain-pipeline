"""Tests for the semantic search layer (src/search/).

Uses only stdlib — no fixtures requiring a real vault or network access.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.search.embedder import (
    NoteEmbedder,
    content_hash,
    strip_frontmatter,
    tokenize,
    STOPWORDS,
)
from src.search.engine import (
    VaultSearchEngine,
    SearchResult,
    _cosine_similarity,
    _tfidf_vector,
    _make_snippet,
    _detect_note_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vault(tmp_path: Path, notes: dict[str, str]) -> Path:
    """Write a dict of {filename: content} into a temp directory."""
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in notes.items():
        p = vault / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return vault


# ---------------------------------------------------------------------------
# tokenize / strip_frontmatter / content_hash
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self):
        assert "python" in tokenize("Python")

    def test_removes_punctuation(self):
        tokens = tokenize("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_stopwords(self):
        tokens = tokenize("the quick brown fox")
        assert "the" not in tokens

    def test_removes_single_char(self):
        tokens = tokenize("a b c python")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "python" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_kept(self):
        assert "42" in tokenize("version 42")


class TestStripFrontmatter:
    def test_no_frontmatter(self):
        text = "Just some content."
        meta, body = strip_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_with_frontmatter(self):
        text = "---\ntitle: My Note\ntags: python, coding\n---\n\nBody here."
        meta, body = strip_frontmatter(text)
        assert meta["title"] == "My Note"
        assert "Body here." in body
        assert "---" not in body

    def test_frontmatter_no_close(self):
        text = "---\ntitle: Broken\n"
        meta, body = strip_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_frontmatter(self):
        text = "---\n---\nContent."
        meta, body = strip_frontmatter(text)
        assert body.strip() == "Content."


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_input(self):
        assert content_hash("hello") != content_hash("world")

    def test_returns_hex(self):
        h = content_hash("test")
        assert len(h) == 64  # SHA-256 hex
        int(h, 16)  # should not raise


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = {"a": 1.0, "b": 2.0}
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine_similarity(a, b) == 0.0

    def test_empty_vectors(self):
        assert _cosine_similarity({}, {"a": 1.0}) == 0.0
        assert _cosine_similarity({"a": 1.0}, {}) == 0.0

    def test_partial_overlap(self):
        a = {"python": 1.0, "coding": 1.0}
        b = {"python": 1.0, "medicine": 1.0}
        sim = _cosine_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_symmetry(self):
        a = {"x": 1.0, "y": 0.5}
        b = {"x": 0.5, "z": 1.0}
        assert abs(_cosine_similarity(a, b) - _cosine_similarity(b, a)) < 1e-9


class TestTfidfVector:
    def test_multiplies_tf_by_idf(self):
        tf = {"python": 0.5}
        idf = {"python": 2.0}
        vec = _tfidf_vector(tf, idf)
        assert abs(vec["python"] - 1.0) < 1e-9

    def test_missing_term_uses_default_idf_1(self):
        tf = {"rare": 0.5}
        idf = {}
        vec = _tfidf_vector(tf, idf)
        assert abs(vec["rare"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# NoteEmbedder
# ---------------------------------------------------------------------------


class TestNoteEmbedder:
    def test_embed_vault_indexes_all_notes(self, tmp_path):
        notes = {
            "concept_python.md": "---\ntitle: Python Basics\ntype: concept\n---\nPython is a programming language.",
            "entity_numpy.md": "---\ntitle: NumPy\ntype: entity\n---\nNumPy is a numerical computing library.",
        }
        vault = _make_vault(tmp_path, notes)
        db_path = tmp_path / "embeddings.db"

        embedder = NoteEmbedder(db_path=db_path)
        changed = embedder.embed_vault(vault)
        embedder.close()

        assert changed == 2

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM note_embeddings").fetchone()[0]
        idf_count = conn.execute("SELECT COUNT(*) FROM corpus_idf").fetchone()[0]
        conn.close()

        assert count == 2
        assert idf_count > 0

    def test_incremental_no_change(self, tmp_path):
        notes = {"note.md": "Some content about Python and medicine."}
        vault = _make_vault(tmp_path, notes)
        db_path = tmp_path / "embeddings.db"

        embedder = NoteEmbedder(db_path=db_path)
        first = embedder.embed_vault(vault)
        second = embedder.embed_vault(vault)  # nothing changed
        embedder.close()

        assert first == 1
        assert second == 0  # no re-embedding needed

    def test_incremental_detects_change(self, tmp_path):
        md = tmp_path / "vault" / "note.md"
        md.parent.mkdir()
        md.write_text("Original content about surgery.", encoding="utf-8")

        db_path = tmp_path / "embeddings.db"
        embedder = NoteEmbedder(db_path=db_path)
        embedder.embed_vault(tmp_path / "vault")

        md.write_text("Updated content about anesthesia.", encoding="utf-8")
        changed = embedder.embed_vault(tmp_path / "vault")
        embedder.close()

        assert changed == 1

    def test_force_rebuild(self, tmp_path):
        notes = {"a.md": "Content A.", "b.md": "Content B."}
        vault = _make_vault(tmp_path, notes)
        db_path = tmp_path / "embeddings.db"

        embedder = NoteEmbedder(db_path=db_path)
        embedder.embed_vault(vault)
        # Second call with force_rebuild should re-embed both notes
        changed = embedder.embed_vault(vault, force_rebuild=True)
        embedder.close()

        assert changed == 2

    def test_idf_populated(self, tmp_path):
        notes = {
            "a.md": "Python machine learning model training.",
            "b.md": "Python clinical workflow automation.",
        }
        vault = _make_vault(tmp_path, notes)
        db_path = tmp_path / "embeddings.db"

        with NoteEmbedder(db_path=db_path) as embedder:
            embedder.embed_vault(vault)
            idf = embedder.get_idf()

        # "python" appears in both docs — should have lower IDF than rare terms
        assert "python" in idf
        assert idf["python"] < idf.get("training", idf["python"] + 1)

    def test_doc_count(self, tmp_path):
        notes = {"a.md": "Note A.", "b.md": "Note B.", "c.md": "Note C."}
        vault = _make_vault(tmp_path, notes)
        db_path = tmp_path / "embeddings.db"

        with NoteEmbedder(db_path=db_path) as embedder:
            embedder.embed_vault(vault)
            assert embedder.get_doc_count() == 3

    def test_missing_vault_returns_zero(self, tmp_path):
        db_path = tmp_path / "embeddings.db"
        with NoteEmbedder(db_path=db_path) as embedder:
            changed = embedder.embed_vault(tmp_path / "nonexistent")
        assert changed == 0

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "embeddings.db"
        with NoteEmbedder(db_path=db_path) as embedder:
            assert embedder.get_doc_count() == 0


# ---------------------------------------------------------------------------
# VaultSearchEngine
# ---------------------------------------------------------------------------


class TestVaultSearchEngine:
    def _build_index(self, tmp_path: Path, notes: dict[str, str]) -> Path:
        vault = _make_vault(tmp_path, notes)
        db_path = tmp_path / "embeddings.db"
        with NoteEmbedder(db_path=db_path) as embedder:
            embedder.embed_vault(vault)
        return db_path

    def test_load_returns_doc_count(self, tmp_path):
        notes = {"a.md": "Machine learning models.", "b.md": "Clinical workflows."}
        db_path = self._build_index(tmp_path, notes)

        engine = VaultSearchEngine(db_path=db_path)
        count = engine.load()
        assert count == 2

    def test_search_returns_ranked_results(self, tmp_path):
        notes = {
            "python_concept.md": "---\ntitle: Python Programming\ntype: concept\n---\nPython is used for scripting and automation.",
            "surgery_note.md": "---\ntitle: Surgical Technique\ntype: concept\n---\nSurgical technique for laparoscopic procedures.",
            "anesthesia.md": "---\ntitle: Anesthesia\ntype: concept\n---\nAnesthesia management during surgical operations.",
        }
        db_path = self._build_index(tmp_path, notes)

        engine = VaultSearchEngine(db_path=db_path)
        engine.load()
        results = engine.search("Python scripting automation", top_n=3)

        assert len(results) >= 1
        # Python note should be at the top
        assert "python" in results[0].title.lower() or "python" in results[0].note_path.lower()

    def test_search_respects_top_n(self, tmp_path):
        notes = {f"note_{i}.md": f"Python coding example number {i}." for i in range(10)}
        db_path = self._build_index(tmp_path, notes)

        engine = VaultSearchEngine(db_path=db_path)
        engine.load()
        results = engine.search("Python coding", top_n=3)
        assert len(results) <= 3

    def test_search_similarity_ordering(self, tmp_path):
        notes = {
            "very_relevant.md": "Python programming language scripting automation coding.",
            "somewhat_relevant.md": "Python is used in science.",
            "unrelated.md": "The weather today is sunny and warm.",
        }
        db_path = self._build_index(tmp_path, notes)

        engine = VaultSearchEngine(db_path=db_path)
        engine.load()
        results = engine.search("Python programming scripting")

        assert len(results) >= 1
        # Similarities must be non-increasing
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

    def test_filter_by_note_type(self, tmp_path):
        notes = {
            "Concepts/concept_python.md": "---\ntitle: Python\ntype: concept\n---\nPython programming.",
            "Entities/entity_numpy.md": "---\ntitle: NumPy\ntype: entity\n---\nNumPy Python library.",
        }
        db_path = self._build_index(tmp_path, notes)

        engine = VaultSearchEngine(db_path=db_path)
        engine.load()

        concept_results = engine.search("Python", note_type="concept")
        for r in concept_results:
            assert "concept" in r.note_type.lower() or "concept" in r.note_path.lower()

    def test_filter_by_tags(self, tmp_path):
        notes = {
            "note_a.md": "---\ntitle: A\ntags: python, coding\n---\nPython content.",
            "note_b.md": "---\ntitle: B\ntags: medicine, clinical\n---\nClinical content.",
        }
        db_path = self._build_index(tmp_path, notes)

        engine = VaultSearchEngine(db_path=db_path)
        engine.load()

        results = engine.search("content", tags="python")
        assert all("python" in r.tags.lower() for r in results)

    def test_search_empty_query_tokens(self, tmp_path):
        notes = {"a.md": "Some content."}
        db_path = self._build_index(tmp_path, notes)

        engine = VaultSearchEngine(db_path=db_path)
        engine.load()
        # A query consisting only of stop-words
        results = engine.search("the a an")
        assert results == []

    def test_load_missing_db_raises(self, tmp_path):
        engine = VaultSearchEngine(db_path=tmp_path / "missing.db")
        with pytest.raises(FileNotFoundError):
            engine.load()

    def test_search_before_load_raises(self, tmp_path):
        db_path = tmp_path / "embeddings.db"
        # Create DB but don't load
        with NoteEmbedder(db_path=db_path) as _:
            pass
        engine = VaultSearchEngine(db_path=db_path)
        with pytest.raises(RuntimeError):
            engine.search("query")

    def test_doc_count_property(self, tmp_path):
        notes = {"x.md": "Content X.", "y.md": "Content Y."}
        db_path = self._build_index(tmp_path, notes)

        engine = VaultSearchEngine(db_path=db_path)
        engine.load()
        assert engine.doc_count == 2


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestMakeSnippet:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert _make_snippet(text, max_chars=500) == "Short text."

    def test_truncates_at_word_boundary(self):
        text = "word1 word2 word3 word4 word5"
        snippet = _make_snippet(text, max_chars=10)
        assert snippet.endswith("…")
        # Should not cut mid-word
        assert all(c != " " for c in snippet[:-1].split("…")[0][-5:].lstrip())

    def test_skips_blank_lines(self):
        text = "\n\nActual content here."
        snippet = _make_snippet(text)
        assert snippet.startswith("Actual")


class TestDetectNoteType:
    def test_uses_meta_type(self):
        assert _detect_note_type({"type": "decision"}, "/some/path") == "decision"

    def test_infers_from_path(self):
        assert _detect_note_type({}, "/Vault/Concepts/note.md") == "concept"
        assert _detect_note_type({}, "/Vault/Entities/person.md") == "entity"

    def test_fallback(self):
        assert _detect_note_type({}, "/Vault/misc/note.md") == "note"
