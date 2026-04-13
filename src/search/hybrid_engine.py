"""Hybrid BM25 + optional dense-vector search engine.

Runs BM25 against the existing TF-IDF SQLite index — no new indexing step
required.  If sentence-transformers and ChromaDB are installed, the vector
store is also queried and scores are fused via weighted combination.

Usage::

    engine = HybridSearchEngine(db_path=Path(".dbp_cache/embeddings.db"))
    engine.load()
    results = engine.search("anesthesia workflow", top_k=5)
    for r in results:
        print(r.title, r.score, r.snippet)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.search.embedder import tokenize, strip_frontmatter

logger = logging.getLogger(__name__)

# Okapi BM25 hyperparameters
_K1 = 1.5   # term-frequency saturation
_B = 0.75   # document-length normalisation


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HybridResult:
    """A single ranked search result from the hybrid engine."""

    note_path: str
    title: str
    score: float            # fused score, normalised to [0, 1]
    bm25_score: float       # normalised BM25 component
    vector_score: float     # normalised vector component (0 if unavailable)
    snippet: str
    note_type: str = ""
    domain: str = ""
    tags: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HybridSearchEngine
# ---------------------------------------------------------------------------

# Internal tuple stored per document after load()
_DocRow = Tuple[str, Dict[str, float], Dict[str, str], int]
# (note_path, normalised_tf, meta_dict, approx_doc_len)


class HybridSearchEngine:
    """BM25 + optional vector search over the vault SQLite index.

    Lifecycle::

        engine = HybridSearchEngine(db_path=..., chroma_path=...)
        engine.load()          # loads SQLite; opens vector store if available
        results = engine.search("...", top_k=5)
    """

    def __init__(
        self,
        db_path: Path = Path(".dbp_cache/embeddings.db"),
        chroma_path: Optional[Path] = None,
        vault_path: Optional[Path] = None,
        vector_weight: float = 0.3,
    ) -> None:
        self._db_path = Path(db_path)
        # Treat empty string as "disabled"
        self._chroma_path = (
            Path(chroma_path) if chroma_path and str(chroma_path) else None
        )
        self._vault_path = (
            Path(vault_path).expanduser() if vault_path else None
        )
        self._vector_weight = max(0.0, min(1.0, vector_weight))
        self._bm25_weight = 1.0 - self._vector_weight

        self._docs: List[_DocRow] = []
        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 1.0
        self._n_docs: int = 0
        self._vector_store: Any = None  # ChromaDB collection or None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> int:
        """Load the SQLite index into memory.

        Returns the number of documents loaded.

        Raises:
            FileNotFoundError: if the embeddings DB does not exist.
        """
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Embeddings DB not found at {self._db_path}. "
                "Run the indexer first: python3 scripts/start_rag_server.py --rebuild"
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

        total_len = 0
        self._docs = []
        for note_path, tf_json, meta_json in doc_rows:
            raw_tf = json.loads(tf_json)   # normalised TF: count / doc_len
            meta = json.loads(meta_json)
            # Approximate raw token count from normalised TF.
            # The smallest tf value is ≈ 1/doc_len (for the rarest term).
            approx_dl = round(1.0 / min(raw_tf.values())) if raw_tf else 1
            total_len += approx_dl
            self._docs.append((note_path, raw_tf, meta, approx_dl))

        self._n_docs = len(self._docs)
        self._avg_dl = total_len / self._n_docs if self._n_docs else 1.0

        logger.info(
            "HybridSearchEngine loaded %d docs (avg_dl≈%.1f words)",
            self._n_docs,
            self._avg_dl,
        )

        if self._chroma_path:
            self._vector_store = _try_load_vector_store(self._chroma_path)

        return self._n_docs

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        note_type: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> List[HybridResult]:
        """Return top-k notes most relevant to *query*.

        Args:
            query: Natural-language search string.
            top_k: Maximum results to return.
            note_type: Case-insensitive substring match on note type.
            domain: Case-insensitive substring match on note domain field.
            tags: Case-insensitive substring match on note tags field.
        """
        if not self._docs:
            raise RuntimeError("Index is empty — call load() before search().")

        query_terms = tokenize(query)
        if not query_terms:
            logger.warning("Query %r produced no tokens after stop-word removal", query)
            return []

        # --- BM25 scoring ---
        bm25_scores: Dict[str, float] = {}
        for path, raw_tf, meta, dl in self._docs:
            if not _passes_filters(meta, path, note_type, domain, tags):
                continue
            score = _bm25_score(
                query_terms, raw_tf, self._idf, dl, self._avg_dl
            )
            if score > 0.0:
                bm25_scores[path] = score

        # --- Vector scoring (optional) ---
        vector_scores: Dict[str, float] = {}
        if self._vector_store is not None and self._vector_weight > 0:
            vector_scores = _vector_search(self._vector_store, query, top_k * 3)

        # --- Fuse scores (normalise each component to [0, 1]) ---
        max_bm25 = max(bm25_scores.values(), default=1.0) or 1.0
        max_vector = max(vector_scores.values(), default=1.0) or 1.0

        all_paths = set(bm25_scores) | set(vector_scores)
        fused: Dict[str, float] = {
            path: (
                (bm25_scores.get(path, 0.0) / max_bm25) * self._bm25_weight
                + (vector_scores.get(path, 0.0) / max_vector) * self._vector_weight
            )
            for path in all_paths
        }

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build a fast lookup for metadata
        meta_lookup: Dict[str, Tuple[Dict[str, str], int]] = {
            p: (m, dl) for p, _, m, dl in self._docs
        }

        results: List[HybridResult] = []
        for path, fused_score in ranked:
            meta, _ = meta_lookup.get(path, ({}, 0))
            title = meta.get("title", Path(path).stem)
            results.append(
                HybridResult(
                    note_path=path,
                    title=title,
                    score=round(fused_score, 6),
                    bm25_score=round(bm25_scores.get(path, 0.0) / max_bm25, 6),
                    vector_score=round(
                        vector_scores.get(path, 0.0) / max_vector, 6
                    ),
                    snippet=_read_snippet(path),
                    note_type=_detect_note_type(meta, path),
                    domain=meta.get("domain", meta.get("category", "")),
                    tags=meta.get("tags", ""),
                    metadata=meta,
                )
            )
        return results

    @property
    def doc_count(self) -> int:
        """Number of documents loaded in memory."""
        return self._n_docs

    def is_loaded(self) -> bool:
        return self._n_docs > 0


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------


def _bm25_score(
    query_terms: List[str],
    raw_tf: Dict[str, float],   # normalised TF (count / doc_len)
    idf: Dict[str, float],
    dl: int,
    avg_dl: float,
) -> float:
    """Okapi BM25 score for a single document against a query."""
    score = 0.0
    for term in query_terms:
        if term not in raw_tf:
            continue
        # Recover approximate raw count from normalised TF
        tf_raw = raw_tf[term] * dl
        term_idf = idf.get(term, 1.0)
        numerator = tf_raw * (_K1 + 1)
        denominator = tf_raw + _K1 * (1 - _B + _B * dl / avg_dl)
        score += term_idf * (numerator / denominator)
    return score


def _passes_filters(
    meta: Dict[str, str],
    path: str,
    note_type: Optional[str],
    domain: Optional[str],
    tags: Optional[str],
) -> bool:
    if note_type is not None:
        detected = _detect_note_type(meta, path)
        if note_type.lower() not in detected.lower():
            return False
    if domain is not None:
        note_domain = meta.get("domain", meta.get("category", "")).lower()
        if domain.lower() not in note_domain:
            return False
    if tags is not None:
        if tags.lower() not in meta.get("tags", "").lower():
            return False
    return True


def _detect_note_type(meta: Dict[str, str], path_str: str) -> str:
    if meta.get("type"):
        return meta["type"]
    _type_dirs = {
        "entity": ("entity", "entities"),
        "concept": ("concept", "concepts"),
        "decision": ("decision", "decisions"),
        "action": ("action", "actions"),
        "conversation": ("conversation", "conversations"),
        "contact": ("contact", "contacts"),
        "project": ("project", "projects"),
    }
    parts_lower = {p.lower() for p in Path(path_str).parts}
    for note_type, dir_names in _type_dirs.items():
        if parts_lower & set(dir_names):
            return note_type
    return "note"


def _read_snippet(note_path: str, max_chars: int = 400) -> str:
    try:
        text = Path(note_path).read_text(encoding="utf-8", errors="replace")
        _, body = strip_frontmatter(text)
        lines = [ln for ln in body.splitlines() if ln.strip()]
        flat = " ".join(lines)
        if len(flat) <= max_chars:
            return flat
        return flat[:max_chars].rsplit(" ", 1)[0] + "…"
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Optional vector store integration
# ---------------------------------------------------------------------------


def _try_load_vector_store(chroma_path: Path) -> Any:
    """Return a ChromaDB collection, or None if chromadb is unavailable."""
    try:
        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection("vault")
        logger.info("ChromaDB loaded from %s (%d docs)", chroma_path, collection.count())
        return collection
    except ImportError:
        logger.info("chromadb not installed; vector search disabled")
        return None
    except Exception as exc:
        logger.warning("Failed to open ChromaDB at %s: %s", chroma_path, exc)
        return None


def _vector_search(collection: Any, query: str, top_k: int) -> Dict[str, float]:
    """Query ChromaDB; return {note_path: similarity_score} dict."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        # Cache the model on the function object (poor-man's singleton)
        if not hasattr(_vector_search, "_model"):
            _vector_search._model = SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore

        model = _vector_search._model  # type: ignore
        embedding = model.encode(query).tolist()
        n = min(top_k, collection.count() or 1)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["distances"],
        )
        scores: Dict[str, float] = {}
        if results and results.get("ids"):
            for doc_id, dist in zip(results["ids"][0], results["distances"][0]):
                scores[doc_id] = 1.0 / (1.0 + dist)   # L2 → similarity
        return scores
    except Exception as exc:
        logger.warning("Vector search failed: %s", exc)
        return {}
