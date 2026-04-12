"""Semantic search engine using TF-IDF cosine similarity.

Pure-Python implementation — no numpy or scikit-learn required.

At query time the full TF-IDF index is loaded from SQLite into memory
as a list of sparse dicts.  Cosine similarity against all 11K notes
takes well under 1 second in pure Python.

Usage::

    engine = VaultSearchEngine(db_path=Path(".dbp_cache/embeddings.db"))
    engine.load()
    results = engine.search("clinical workflow automation", top_n=10)
    for r in results:
        print(r.title, r.similarity, r.snippet)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.search.embedder import NoteEmbedder, tokenize, strip_frontmatter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single ranked search result."""

    note_path: str
    title: str
    similarity: float
    snippet: str
    note_type: str = ""
    tags: str = ""


# ---------------------------------------------------------------------------
# Vector math (sparse dicts, no numpy)
# ---------------------------------------------------------------------------


def _tfidf_vector(
    tf: Dict[str, float], idf: Dict[str, float]
) -> Dict[str, float]:
    """Multiply raw TF by IDF to produce a sparse TF-IDF vector."""
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}


def _cosine_similarity(
    a: Dict[str, float], b: Dict[str, float]
) -> float:
    """Cosine similarity between two sparse vectors.

    Only iterates over shared terms (fast for sparse vectors).
    """
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    if dot == 0.0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snippet(body: str, max_chars: int = 220) -> str:
    """Return a short content preview from *body* text."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    text = " ".join(lines)
    if len(text) <= max_chars:
        return text
    # Break on a word boundary
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _detect_note_type(meta: Dict[str, str], path_str: str) -> str:
    """Infer note type from frontmatter ``type`` field or directory name."""
    if meta.get("type"):
        return meta["type"]
    path_lower = path_str.lower()
    _type_to_dirs = {
        "entity": ("entity", "entities"),
        "concept": ("concept", "concepts"),
        "decision": ("decision", "decisions"),
        "action": ("action", "actions"),
        "conversation": ("conversation", "conversations"),
        "contact": ("contact", "contacts"),
        "project": ("project", "projects"),
    }
    parts = set(Path(path_str).parts)
    parts_lower = {p.lower() for p in parts}
    for note_type, dir_names in _type_to_dirs.items():
        if parts_lower & set(dir_names):
            return note_type
    return "note"


# ---------------------------------------------------------------------------
# VaultSearchEngine
# ---------------------------------------------------------------------------


# Internal row type stored in memory after load()
_DocRow = Tuple[str, Dict[str, float], Dict[str, str]]  # (path, tfidf_vec, meta)


class VaultSearchEngine:
    """Load TF-IDF index from SQLite and answer natural-language queries.

    The index is loaded once into memory; queries run in < 1 s for 11 K notes.

    Lifecycle::

        engine = VaultSearchEngine()
        engine.load()           # reads SQLite once
        results = engine.search("...")
    """

    def __init__(
        self, db_path: Path = Path(".dbp_cache/embeddings.db")
    ) -> None:
        self._db_path = Path(db_path)
        self._idf: Dict[str, float] = {}
        self._docs: List[_DocRow] = []

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def load(self) -> int:
        """Load embeddings from SQLite into memory.

        Returns:
            Number of documents loaded.

        Raises:
            FileNotFoundError: if the embeddings DB does not exist.
        """
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Embeddings DB not found at {self._db_path}. "
                "Run `python3 scripts/search_vault.py --index-only` first, "
                "or pass --rebuild to regenerate."
            )

        conn = sqlite3.connect(str(self._db_path))
        try:
            idf_rows = conn.execute(
                "SELECT term, idf FROM corpus_idf"
            ).fetchall()
            self._idf = {term: idf for term, idf in idf_rows}

            doc_rows = conn.execute(
                "SELECT note_path, tf_json, meta_json FROM note_embeddings"
            ).fetchall()
        finally:
            conn.close()

        self._docs = []
        for note_path, tf_json, meta_json in doc_rows:
            tf = json.loads(tf_json)
            meta = json.loads(meta_json)
            vec = _tfidf_vector(tf, self._idf)
            self._docs.append((note_path, vec, meta))

        logger.debug(
            "Search index loaded: %d docs, %d IDF terms",
            len(self._docs),
            len(self._idf),
        )
        return len(self._docs)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_n: int = 10,
        note_type: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> List[SearchResult]:
        """Return the top-N notes most similar to *query*.

        Args:
            query: Natural-language search string.
            top_n: Maximum number of results to return.
            note_type: If set, only return notes whose type contains this
                string (case-insensitive).  E.g. ``"concept"`` or
                ``"decision"``.
            tags: If set, only return notes whose tag string contains
                this substring (case-insensitive).
        """
        if not self._docs:
            raise RuntimeError(
                "Index is empty — call load() before search()."
            )

        # Embed the query the same way we embedded documents
        query_tf = NoteEmbedder._compute_tf(query)
        query_vec = _tfidf_vector(query_tf, self._idf)

        if not query_vec:
            logger.warning(
                "Query %r produced no tokens after stop-word removal", query
            )
            return []

        # Score every document
        scored: List[Tuple[float, str, Dict[str, str]]] = []
        for note_path, doc_vec, meta in self._docs:
            # --- filters ---
            if note_type is not None:
                detected = _detect_note_type(meta, note_path)
                if note_type.lower() not in detected.lower():
                    continue
            if tags is not None:
                note_tags = meta.get("tags", "")
                if tags.lower() not in note_tags.lower():
                    continue

            sim = _cosine_similarity(query_vec, doc_vec)
            if sim > 0.0:
                scored.append((sim, note_path, meta))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        for sim, note_path, meta in scored[:top_n]:
            title = meta.get("title", Path(note_path).stem)
            note_type_detected = _detect_note_type(meta, note_path)
            snippet = _read_snippet(note_path)
            results.append(
                SearchResult(
                    note_path=note_path,
                    title=title,
                    similarity=sim,
                    snippet=snippet,
                    note_type=note_type_detected,
                    tags=meta.get("tags", ""),
                )
            )
        return results

    @property
    def doc_count(self) -> int:
        """Number of documents currently loaded in memory."""
        return len(self._docs)


# ---------------------------------------------------------------------------
# File I/O helper (kept outside the class to be unit-testable)
# ---------------------------------------------------------------------------


def _read_snippet(note_path: str, max_chars: int = 220) -> str:
    """Read *note_path* and return a body snippet (frontmatter stripped)."""
    try:
        text = Path(note_path).read_text(encoding="utf-8", errors="replace")
        _, body = strip_frontmatter(text)
        return _make_snippet(body, max_chars)
    except OSError:
        return ""
