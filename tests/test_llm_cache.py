"""Tests for the SQLite-backed ExtractionCache.

Uses a temporary in-memory SQLite database so no files are created on disk.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import BaseModel

from src.llm.cache import ExtractionCache
from src.llm.provider import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleResult(BaseModel):
    entities: list[str] = []
    count: int = 0


def _make_response(tokens_in: int = 100, tokens_out: int = 50) -> LLMResponse:
    return LLMResponse(
        content='{"entities": [], "count": 0}',
        model="claude-sonnet-4-20250514",
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost_usd=0.002,
        latency_ms=42.0,
    )


@pytest.fixture()
def cache(tmp_path: Path) -> ExtractionCache:
    """ExtractionCache backed by a temp file database."""
    return ExtractionCache(cache_path=tmp_path / "test_cache.db")


# ---------------------------------------------------------------------------
# hash_conversation
# ---------------------------------------------------------------------------


def test_hash_conversation_deterministic():
    h1 = ExtractionCache.hash_conversation("hello world")
    h2 = ExtractionCache.hash_conversation("hello world")
    assert h1 == h2


def test_hash_conversation_sensitive_to_content():
    h1 = ExtractionCache.hash_conversation("hello world")
    h2 = ExtractionCache.hash_conversation("hello World")
    assert h1 != h2


def test_hash_conversation_returns_hex_string():
    h = ExtractionCache.hash_conversation("test")
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex = 64 chars


# ---------------------------------------------------------------------------
# Cache miss
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none(cache: ExtractionCache):
    result = cache.get(
        conversation_hash="nonexistent",
        stage="entity",
        prompt_version="v1",
        model="claude-sonnet-4-20250514",
        response_model=SimpleResult,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Cache put → get (hit)
# ---------------------------------------------------------------------------


def test_cache_hit_after_put(cache: ExtractionCache):
    result = SimpleResult(entities=["Alice", "Bob"], count=2)
    response = _make_response()

    cache.put(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="claude-sonnet-4-20250514",
        result=result,
        response=response,
    )

    hit = cache.get(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="claude-sonnet-4-20250514",
        response_model=SimpleResult,
    )

    assert hit is not None
    assert hit.result.entities == ["Alice", "Bob"]
    assert hit.result.count == 2


def test_cache_hit_marks_response_as_cached(cache: ExtractionCache):
    cache.put(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="gpt-4o-mini",
        result=SimpleResult(),
        response=_make_response(),
    )
    hit = cache.get(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="gpt-4o-mini",
        response_model=SimpleResult,
    )
    assert hit is not None
    assert hit.response.cached is True


def test_cache_miss_on_different_stage(cache: ExtractionCache):
    cache.put(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="gpt-4o-mini",
        result=SimpleResult(),
        response=_make_response(),
    )
    miss = cache.get(
        conversation_hash="abc123",
        stage="relationship",  # different stage
        prompt_version="v1",
        model="gpt-4o-mini",
        response_model=SimpleResult,
    )
    assert miss is None


def test_cache_miss_on_different_prompt_version(cache: ExtractionCache):
    cache.put(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="gpt-4o-mini",
        result=SimpleResult(),
        response=_make_response(),
    )
    miss = cache.get(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v2",  # different version
        model="gpt-4o-mini",
        response_model=SimpleResult,
    )
    assert miss is None


def test_cache_miss_on_different_model(cache: ExtractionCache):
    cache.put(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="gpt-4o-mini",
        result=SimpleResult(),
        response=_make_response(),
    )
    miss = cache.get(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="claude-sonnet-4-20250514",  # different model
        response_model=SimpleResult,
    )
    assert miss is None


# ---------------------------------------------------------------------------
# put replaces existing entry
# ---------------------------------------------------------------------------


def test_put_replaces_existing(cache: ExtractionCache):
    cache.put(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="gpt-4o-mini",
        result=SimpleResult(count=1),
        response=_make_response(),
    )
    cache.put(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="gpt-4o-mini",
        result=SimpleResult(count=99),
        response=_make_response(),
    )
    hit = cache.get(
        conversation_hash="abc123",
        stage="entity",
        prompt_version="v1",
        model="gpt-4o-mini",
        response_model=SimpleResult,
    )
    assert hit is not None
    assert hit.result.count == 99


# ---------------------------------------------------------------------------
# invalidate_conversation
# ---------------------------------------------------------------------------


def test_invalidate_conversation_removes_all_stages(cache: ExtractionCache):
    for stage in ("entity", "relationship", "classification"):
        cache.put(
            conversation_hash="abc123",
            stage=stage,
            prompt_version="v1",
            model="gpt-4o-mini",
            result=SimpleResult(),
            response=_make_response(),
        )

    deleted = cache.invalidate_conversation("abc123")
    assert deleted == 3

    for stage in ("entity", "relationship", "classification"):
        assert (
            cache.get("abc123", stage, "v1", "gpt-4o-mini", SimpleResult) is None
        )


def test_invalidate_conversation_leaves_others_intact(cache: ExtractionCache):
    for conv_hash in ("conv-A", "conv-B"):
        cache.put(
            conversation_hash=conv_hash,
            stage="entity",
            prompt_version="v1",
            model="gpt-4o-mini",
            result=SimpleResult(),
            response=_make_response(),
        )

    cache.invalidate_conversation("conv-A")

    assert cache.get("conv-A", "entity", "v1", "gpt-4o-mini", SimpleResult) is None
    assert cache.get("conv-B", "entity", "v1", "gpt-4o-mini", SimpleResult) is not None


# ---------------------------------------------------------------------------
# invalidate_stage
# ---------------------------------------------------------------------------


def test_invalidate_stage(cache: ExtractionCache):
    cache.put("conv-A", "entity", "v1", "gpt-4o-mini", SimpleResult(), _make_response())
    cache.put("conv-A", "relationship", "v1", "gpt-4o-mini", SimpleResult(), _make_response())
    cache.put("conv-B", "entity", "v1", "gpt-4o-mini", SimpleResult(), _make_response())

    deleted = cache.invalidate_stage("entity")
    assert deleted == 2

    assert cache.get("conv-A", "entity", "v1", "gpt-4o-mini", SimpleResult) is None
    assert cache.get("conv-B", "entity", "v1", "gpt-4o-mini", SimpleResult) is None
    # relationship entry should survive
    assert cache.get("conv-A", "relationship", "v1", "gpt-4o-mini", SimpleResult) is not None


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_counts_by_stage(cache: ExtractionCache):
    for i in range(3):
        cache.put(f"conv-{i}", "entity", "v1", "gpt-4o-mini", SimpleResult(), _make_response())
    for i in range(2):
        cache.put(f"conv-{i}", "relationship", "v1", "gpt-4o-mini", SimpleResult(), _make_response())

    stats = cache.stats()
    assert stats["entity"] == 3
    assert stats["relationship"] == 2


def test_stats_empty_cache(cache: ExtractionCache):
    assert cache.stats() == {}


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager(tmp_path: Path):
    db_path = tmp_path / "ctx_test.db"
    with ExtractionCache(cache_path=db_path) as cache:
        cache.put("h", "entity", "v1", "m", SimpleResult(), _make_response())
        assert cache.get("h", "entity", "v1", "m", SimpleResult) is not None
    # Connection should be closed after __exit__ — accessing it should fail
    with pytest.raises(Exception):
        cache._conn.execute("SELECT 1")
