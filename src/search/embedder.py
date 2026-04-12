"""TF-IDF note embedder with SQLite storage.

Pure-Python implementation — no numpy or scikit-learn required.
Embeddings are stored as sparse TF-IDF term-frequency maps (JSON) keyed
by note path.  IDF is computed over the full corpus and stored separately.

Only notes whose content hash has changed are re-processed; IDF is
recomputed from scratch whenever any document changes (because adding a
document shifts global term frequencies).

Usage::

    embedder = NoteEmbedder(db_path=Path(".dbp_cache/embeddings.db"))
    changed = embedder.embed_vault(Path("~/Desktop/claude-vault-output"))
    embedder.close()
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words
# ---------------------------------------------------------------------------

STOPWORDS: frozenset = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "need",
        "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
        "it", "its", "not", "no", "if", "then", "than", "so", "yet", "both",
        "also", "just", "about", "up", "out", "into", "through", "during",
        "before", "after", "between", "each", "more", "other", "some", "such",
        "only", "own", "same", "too", "very", "s", "t", "all", "any", "am",
        "what", "which", "who", "whom", "how", "when", "where", "why",
        "their", "there", "here", "my", "your", "our", "his", "her",
        "obsidian", "note", "created", "updated", "tags", "type",
    }
)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, remove stop-words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


def strip_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Strip YAML frontmatter and return ``(meta_dict, body_text)``."""
    meta: Dict[str, str] = {}
    if not text.startswith("---"):
        return meta, text
    end = text.find("\n---", 3)
    if end == -1:
        return meta, text
    frontmatter = text[3:end].strip()
    body = text[end + 4 :].strip()
    for line in frontmatter.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta, body


def content_hash(text: str) -> str:
    """SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# NoteEmbedder
# ---------------------------------------------------------------------------


class NoteEmbedder:
    """Read vault notes, build TF-IDF vectors, persist in SQLite.

    Schema
    ------
    ``note_embeddings``
        note_path TEXT PRIMARY KEY,
        content_hash TEXT,
        tf_json TEXT        — JSON object mapping term → raw TF (float),
        meta_json TEXT      — JSON object with frontmatter key/values,
        updated_at TEXT

    ``corpus_idf``
        term TEXT PRIMARY KEY,
        idf REAL

    ``corpus_meta``
        key TEXT PRIMARY KEY,
        value TEXT
    """

    def __init__(
        self, db_path: Path = Path(".dbp_cache/embeddings.db")
    ) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS note_embeddings (
                note_path    TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                tf_json      TEXT NOT NULL,
                meta_json    TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS corpus_idf (
                term  TEXT PRIMARY KEY,
                idf   REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS corpus_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_vault(self, vault_path: Path, force_rebuild: bool = False) -> int:
        """Scan *vault_path* recursively, embed new/changed notes.

        Returns the number of notes that were added or updated.
        IDF is recomputed when any note changes.
        """
        vault_path = Path(vault_path).expanduser()
        if not vault_path.exists():
            logger.warning("Vault path does not exist: %s", vault_path)
            return 0

        md_files = list(vault_path.rglob("*.md"))
        logger.info("Found %d markdown files in %s", len(md_files), vault_path)

        changed_count = 0

        for md_file in md_files:
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Could not read %s: %s", md_file, exc)
                continue

            h = content_hash(text)
            path_str = str(md_file)

            if not force_rebuild:
                row = self._conn.execute(
                    "SELECT content_hash FROM note_embeddings WHERE note_path = ?",
                    (path_str,),
                ).fetchone()
                if row and row[0] == h:
                    continue  # unchanged — skip

            meta, body = strip_frontmatter(text)
            # Boost title and tags by repeating them in the searchable text
            title = meta.get("title", md_file.stem)
            tag_str = meta.get("tags", "")
            searchable = f"{title} {title} {tag_str} {body}"
            tf = self._compute_tf(searchable)

            self._conn.execute(
                """
                INSERT OR REPLACE INTO note_embeddings
                    (note_path, content_hash, tf_json, meta_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    path_str,
                    h,
                    json.dumps(tf),
                    json.dumps(meta),
                    datetime.now().isoformat(),
                ),
            )
            changed_count += 1

        self._conn.commit()

        if changed_count > 0 or force_rebuild:
            logger.info("%d notes changed — recomputing IDF...", changed_count)
            self._recompute_idf()

        doc_count = self._conn.execute(
            "SELECT COUNT(*) FROM note_embeddings"
        ).fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO corpus_meta (key, value) VALUES ('doc_count', ?)",
            (str(doc_count),),
        )
        self._conn.commit()

        logger.info(
            "Embedding index: %d total notes, %d updated", doc_count, changed_count
        )
        return changed_count

    def get_doc_count(self) -> int:
        """Return the number of indexed notes."""
        row = self._conn.execute(
            "SELECT value FROM corpus_meta WHERE key = 'doc_count'"
        ).fetchone()
        return int(row[0]) if row else 0

    def get_idf(self) -> Dict[str, float]:
        """Return the full IDF table as a dict."""
        rows = self._conn.execute("SELECT term, idf FROM corpus_idf").fetchall()
        return {term: idf for term, idf in rows}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "NoteEmbedder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tf(text: str) -> Dict[str, float]:
        """Raw term frequency normalised by document length."""
        tokens = tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in counts.items()}

    def _recompute_idf(self) -> None:
        """Recompute IDF for all terms across the full indexed corpus.

        Uses sklearn-style smoothed IDF:
            idf(t) = log((1 + N) / (1 + df(t))) + 1
        """
        rows = self._conn.execute(
            "SELECT tf_json FROM note_embeddings"
        ).fetchall()
        n_docs = len(rows)
        if n_docs == 0:
            return

        df: Counter = Counter()
        for (tf_json,) in rows:
            terms = json.loads(tf_json).keys()
            df.update(terms)

        idf = {
            term: math.log((1 + n_docs) / (1 + doc_freq)) + 1.0
            for term, doc_freq in df.items()
        }

        self._conn.execute("DELETE FROM corpus_idf")
        self._conn.executemany(
            "INSERT INTO corpus_idf (term, idf) VALUES (?, ?)",
            idf.items(),
        )
        self._conn.commit()
        logger.debug("IDF recomputed: %d unique terms", len(idf))
