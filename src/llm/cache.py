"""SQLite-backed extraction cache.

Responses are keyed on (conversation_hash, stage, prompt_version, model) so
that re-running the pipeline on unchanged conversations is free — and upgrading
a prompt automatically invalidates the stale cache entries for that stage.

Cache location: ``.dbp_cache/extractions.db`` (gitignored).

Usage::

    cache = ExtractionCache()

    hit = cache.get(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="claude-sonnet-4-20250514",
        response_model=LLMEntityExtraction,
    )
    if hit is not None:
        return hit.result, hit.response   # fast path

    # ... call the LLM ...
    cache.put(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="claude-sonnet-4-20250514",
        result=extraction,
        response=llm_response,
    )
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.llm.provider import LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class CachedExtraction(Generic[T]):
    """A single cache hit, returned by ``ExtractionCache.get()``."""

    result: T
    response: "LLMResponse"
    cached_at: datetime


class ExtractionCache:
    """SQLite-backed cache for LLM extraction results.

    Thread-safety: SQLite in WAL mode allows one writer at a time.
    For the typical single-process pipeline this is fine.
    """

    def __init__(self, cache_path: Path = Path(".dbp_cache/extractions.db")) -> None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = cache_path
        self._conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS extractions (
                conversation_hash TEXT NOT NULL,
                stage             TEXT NOT NULL,
                prompt_version    TEXT NOT NULL,
                model             TEXT NOT NULL,
                result_json       TEXT NOT NULL,
                response_json     TEXT NOT NULL,
                cached_at         TEXT NOT NULL,
                PRIMARY KEY (conversation_hash, stage, prompt_version, model)
            );

            CREATE INDEX IF NOT EXISTS idx_stage ON extractions (stage);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def hash_conversation(conversation_text: str) -> str:
        """Return a stable SHA-256 hex digest of the conversation text."""
        return hashlib.sha256(conversation_text.encode("utf-8")).hexdigest()

    def get(
        self,
        conversation_hash: str,
        stage: str,
        prompt_version: str,
        model: str,
        response_model: Type[T],
    ) -> Optional[CachedExtraction[T]]:
        """Return a cached extraction, or None on a miss.

        Args:
            conversation_hash: SHA-256 hex digest of the conversation text
                               (use ``hash_conversation()`` to compute it).
            stage: Extraction stage, e.g. ``"entity"``, ``"relationship"``.
            prompt_version: Prompt version string, e.g. ``"v1"``.
            model: Model name, e.g. ``"claude-sonnet-4-20250514"``.
            response_model: The Pydantic model class to deserialize into.
        """
        from src.llm.provider import LLMResponse  # noqa: PLC0415

        row = self._conn.execute(
            """
            SELECT result_json, response_json, cached_at
              FROM extractions
             WHERE conversation_hash = ?
               AND stage             = ?
               AND prompt_version    = ?
               AND model             = ?
            """,
            (conversation_hash, stage, prompt_version, model),
        ).fetchone()

        if row is None:
            return None

        try:
            result = response_model.model_validate_json(row[0])
            response = LLMResponse.model_validate_json(row[1])
            cached_at = datetime.fromisoformat(row[2])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cache entry corrupt for %s/%s — evicting: %s",
                stage,
                conversation_hash[:12],
                exc,
            )
            self.invalidate_conversation(conversation_hash)
            return None

        # Mark as cached so cost tracking counts it correctly
        response = response.model_copy(update={"cached": True})
        return CachedExtraction(result=result, response=response, cached_at=cached_at)

    def put(
        self,
        conversation_hash: str,
        stage: str,
        prompt_version: str,
        model: str,
        result: BaseModel,
        response: "LLMResponse",
    ) -> None:
        """Store an extraction result in the cache.

        Replaces any existing entry for the same key.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO extractions
                (conversation_hash, stage, prompt_version, model,
                 result_json, response_json, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_hash,
                stage,
                prompt_version,
                model,
                result.model_dump_json(),
                response.model_dump_json(),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()

    def invalidate_conversation(self, conversation_hash: str) -> int:
        """Delete all cached entries for a conversation.

        Returns the number of rows deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM extractions WHERE conversation_hash = ?",
            (conversation_hash,),
        )
        self._conn.commit()
        return cursor.rowcount

    def invalidate_stage(self, stage: str) -> int:
        """Delete all cached entries for a given extraction stage.

        Use this after updating a prompt to force re-extraction.
        Returns the number of rows deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM extractions WHERE stage = ?",
            (stage,),
        )
        self._conn.commit()
        return cursor.rowcount

    def invalidate_prompt_version(self, stage: str, prompt_version: str) -> int:
        """Delete entries for a specific stage that don't match the given version."""
        cursor = self._conn.execute(
            "DELETE FROM extractions WHERE stage = ? AND prompt_version != ?",
            (stage, prompt_version),
        )
        self._conn.commit()
        return cursor.rowcount

    def stats(self) -> Dict[str, int]:
        """Return row counts grouped by stage."""
        rows = self._conn.execute(
            "SELECT stage, COUNT(*) FROM extractions GROUP BY stage"
        ).fetchall()
        return {stage: count for stage, count in rows}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ExtractionCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
