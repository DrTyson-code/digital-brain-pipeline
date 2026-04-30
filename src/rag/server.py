"""Lightweight FastAPI RAG server for the digital brain vault.

Exposes the vault's hybrid search index over a JSON API so any LLM can
query it at inference time.

Endpoints
---------
POST /query
    Body: {"query": "...", "top_k": 5, "filters": {"note_type": "...", "domain": "..."}}
    Returns ranked results with snippets, metadata, scores, and a pre-built
    context string ready to inject into an LLM prompt.

GET /health
    Liveness check — returns doc count and vector search status.

GET /context?q=...&top_k=5
    Shortcut that returns only the assembled context string (handy for curl).

Configuration (environment variables)
--------------------------------------
RAG_VAULT_PATH   Path to vault (default ~/Vault/Claude-Brain)
RAG_DB_PATH      SQLite embeddings DB (default .dbp_cache/embeddings.db)
RAG_CHROMA_PATH  ChromaDB directory (default data/chroma; empty = disabled)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.rag.context_builder import ContextBuilder
from src.search.hybrid_engine import HybridSearchEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class QueryFilters(BaseModel):
    note_type: Optional[str] = None
    domain: Optional[str] = None
    tags: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: QueryFilters = Field(default_factory=QueryFilters)


class ResultItem(BaseModel):
    note_path: str
    title: str
    score: float
    bm25_score: float
    vector_score: float
    snippet: str
    note_type: str
    domain: str
    tags: str
    metadata: Dict[str, str]


class QueryResponse(BaseModel):
    query: str
    total: int
    results: List[ResultItem]
    context: str   # pre-built context string for LLM injection


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    vault_path = Path(
        os.environ.get("RAG_VAULT_PATH", "~/Vault/Claude-Brain")
    ).expanduser()
    db_path = Path(os.environ.get("RAG_DB_PATH", ".dbp_cache/embeddings.db"))

    chroma_raw = os.environ.get("RAG_CHROMA_PATH", "data/chroma")
    chroma_path: Optional[Path] = Path(chroma_raw) if chroma_raw else None

    # Auto-index if the DB is missing
    if not db_path.exists():
        logger.warning(
            "Embeddings DB not found at %s — indexing vault now…", db_path
        )
        from src.search.embedder import NoteEmbedder

        db_path.parent.mkdir(parents=True, exist_ok=True)
        indexer = NoteEmbedder(db_path=db_path)
        n = indexer.embed_vault(vault_path)
        indexer.close()
        logger.info("Indexed %d notes", n)

    engine = HybridSearchEngine(
        db_path=db_path,
        chroma_path=chroma_path,
        vault_path=vault_path,
    )
    engine.load()

    app.state.engine = engine
    app.state.context_builder = ContextBuilder()

    logger.info(
        "RAG server ready — %d docs indexed, vault=%s, vector=%s",
        engine.doc_count,
        vault_path,
        engine._vector_store is not None,
    )
    yield
    # Nothing to clean up on shutdown


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Digital Brain RAG API",
    description=(
        "Query the Claude/ChatGPT vault via hybrid BM25 + vector search. "
        "POST /query to retrieve relevant notes and a ready-to-use LLM context."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    engine: HybridSearchEngine = request.app.state.engine
    return {
        "status": "ok",
        "doc_count": engine.doc_count,
        "vector_search_enabled": engine._vector_store is not None,
    }


@app.post("/query", response_model=QueryResponse)
async def query_vault(req: QueryRequest, request: Request) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    engine: HybridSearchEngine = request.app.state.engine
    builder: ContextBuilder = request.app.state.context_builder

    try:
        results = engine.search(
            query=req.query,
            top_k=req.top_k,
            note_type=req.filters.note_type,
            domain=req.filters.domain,
            tags=req.filters.tags,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    context = builder.build(results, query=req.query)

    return QueryResponse(
        query=req.query,
        total=len(results),
        results=[
            ResultItem(
                note_path=r.note_path,
                title=r.title,
                score=round(r.score, 4),
                bm25_score=round(r.bm25_score, 4),
                vector_score=round(r.vector_score, 4),
                snippet=r.snippet,
                note_type=r.note_type,
                domain=r.domain,
                tags=r.tags,
                metadata=r.metadata,
            )
            for r in results
        ],
        context=context,
    )


@app.get("/context")
async def get_context(
    q: str,
    top_k: int = 5,
    note_type: Optional[str] = None,
    domain: Optional[str] = None,
    request: Request = None,  # type: ignore[assignment]
) -> Dict[str, Any]:
    """GET shortcut — returns only the context string and source list."""
    req = QueryRequest(
        query=q,
        top_k=top_k,
        filters=QueryFilters(note_type=note_type, domain=domain),
    )
    resp = await query_vault(req, request)
    return {
        "query": q,
        "context": resp.context,
        "sources": [r.note_path for r in resp.results],
    }
