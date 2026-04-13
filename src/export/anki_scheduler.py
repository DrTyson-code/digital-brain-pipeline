"""Anki export scheduler — tracks export history and supports incremental exports.

Uses a SQLite database at data/anki_exports.db to record:
- Which source notes/concepts have already been exported (deduplication)
- Export run history (timestamps, card counts, output paths)

This enables incremental exports: only new or changed notes are exported on
subsequent runs, keeping decks fresh without duplicate cards.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.export.anki import AnkiCard

logger = logging.getLogger(__name__)

# SQLite schema
_SCHEMA_EXPORTED_NOTES = """
CREATE TABLE IF NOT EXISTS exported_notes (
    source_id   TEXT    NOT NULL,
    source_type TEXT    NOT NULL,
    exported_at TEXT    NOT NULL,
    card_count  INTEGER DEFAULT 1,
    content_hash TEXT,
    PRIMARY KEY (source_id, source_type)
)
"""

_SCHEMA_EXPORT_RUNS = """
CREATE TABLE IF NOT EXISTS export_runs (
    run_id      TEXT    PRIMARY KEY,
    started_at  TEXT    NOT NULL,
    completed_at TEXT,
    card_count  INTEGER DEFAULT 0,
    format      TEXT,
    output_path TEXT
)
"""


@dataclass
class ExportRecord:
    """Metadata for a single exported note/concept."""

    source_id: str
    source_type: str
    exported_at: str
    card_count: int = 1
    content_hash: Optional[str] = None


@dataclass
class ExportRun:
    """Summary of a single export run."""

    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    card_count: int = 0
    format: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class AnkiScheduler:
    """SQLite-backed scheduler for incremental Anki exports.

    Tracks which source notes/concepts have already been exported to prevent
    duplicate cards in the Anki deck across multiple pipeline runs.

    Args:
        db_path: Path to the SQLite database file.  Created on first use.
        batch_size: Maximum number of new cards to include in one export.
        export_frequency_days: Minimum days between full re-exports of the
            same source (0 = no minimum, always allow).
    """

    db_path: Path = field(default_factory=lambda: Path("data/anki_exports.db"))
    batch_size: int = 50
    export_frequency_days: int = 0

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_new(self, cards: List[AnkiCard]) -> List[AnkiCard]:
        """Return only cards whose source has not been exported before.

        If ``export_frequency_days`` > 0, cards whose last export was within
        that window are also excluded (even if content changed).

        Args:
            cards: Full candidate card list.

        Returns:
            Subset of cards eligible for export, capped at ``batch_size``.
        """
        new_cards: List[AnkiCard] = []
        with self._connect() as conn:
            for card in cards:
                if not card.source_id:
                    new_cards.append(card)
                    continue

                row = conn.execute(
                    "SELECT exported_at, content_hash FROM exported_notes "
                    "WHERE source_id = ? AND source_type = ?",
                    (card.source_id, card.source_type),
                ).fetchone()

                if row is None:
                    # Never exported
                    new_cards.append(card)
                    continue

                last_exported_str, last_hash = row

                # Check if content changed (re-export if so, unless freq gate blocks it)
                if last_hash and card.content_hash != last_hash:
                    if self._within_frequency_gate(last_exported_str):
                        logger.debug(
                            "Card %s changed but within frequency gate — skipping",
                            card.source_id,
                        )
                        continue
                    new_cards.append(card)
                # else: same content, same source_id — skip (already exported)

        logger.info(
            "filter_new: %d new card(s) from %d candidate(s) (batch_size=%d)",
            min(len(new_cards), self.batch_size),
            len(cards),
            self.batch_size,
        )
        return new_cards[: self.batch_size]

    def mark_exported(
        self,
        source_id: str,
        source_type: str,
        card_count: int = 1,
        content_hash: Optional[str] = None,
    ) -> None:
        """Record that a source note/concept has been exported.

        Upserts the exported_notes table so re-exports update the timestamp
        and hash rather than inserting duplicate rows.
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO exported_notes (source_id, source_type, exported_at, card_count, content_hash)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_id, source_type) DO UPDATE SET
                    exported_at  = excluded.exported_at,
                    card_count   = excluded.card_count,
                    content_hash = excluded.content_hash
                """,
                (source_id, source_type, now, card_count, content_hash),
            )

    def mark_exported_batch(self, cards: List[AnkiCard]) -> None:
        """Mark all cards in a batch as exported."""
        for card in cards:
            if card.source_id:
                self.mark_exported(
                    card.source_id, card.source_type, content_hash=card.content_hash
                )

    def is_exported(self, source_id: str, source_type: str) -> bool:
        """Return True if this source has been exported before."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM exported_notes WHERE source_id = ? AND source_type = ?",
                (source_id, source_type),
            ).fetchone()
        return row is not None

    def start_run(self, fmt: str, output_path: Path) -> str:
        """Record the start of an export run; return the run_id."""
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO export_runs (run_id, started_at, format, output_path) VALUES (?, ?, ?, ?)",
                (run_id, now, fmt, str(output_path)),
            )
        return run_id

    def complete_run(self, run_id: str, card_count: int) -> None:
        """Record the completion of an export run."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE export_runs SET completed_at = ?, card_count = ? WHERE run_id = ?",
                (now, card_count, run_id),
            )

    def export_history(self, limit: int = 20) -> List[Dict]:
        """Return the most recent export runs as a list of dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, started_at, completed_at, card_count, format, output_path "
                "FROM export_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ["run_id", "started_at", "completed_at", "card_count", "format", "output_path"]
        return [dict(zip(keys, row)) for row in rows]

    def exported_count(self) -> int:
        """Return the total number of distinct source notes recorded."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM exported_notes").fetchone()
        return row[0] if row else 0

    def reset(self) -> None:
        """Clear all export history (useful for testing or full re-export)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM exported_notes")
            conn.execute("DELETE FROM export_runs")
        logger.info("Anki export history cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA_EXPORTED_NOTES)
            conn.execute(_SCHEMA_EXPORT_RUNS)

    def _within_frequency_gate(self, last_exported_str: str) -> bool:
        """Return True if last export was within the frequency gate window."""
        if self.export_frequency_days <= 0:
            return False
        try:
            last = datetime.fromisoformat(last_exported_str)
            age_days = (datetime.now() - last).days
            return age_days < self.export_frequency_days
        except ValueError:
            return False
