#!/usr/bin/env python3
"""Start the Digital Brain RAG server.

Optionally rebuilds the TF-IDF (and vector) index before starting uvicorn.

Usage::

    # Quick start (uses existing index if present)
    python3 scripts/start_rag_server.py

    # Custom vault / port
    python3 scripts/start_rag_server.py --port 8742 --vault-path ~/Desktop/claude-vault-output

    # Force full re-index before starting
    python3 scripts/start_rag_server.py --rebuild
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is importable when running the script directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("rag.start")


def _index_vault(vault_path: Path, db_path: Path, force: bool) -> None:
    from src.search.embedder import NoteEmbedder

    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Indexing vault at %s …", vault_path)
    indexer = NoteEmbedder(db_path=db_path)
    n = indexer.embed_vault(vault_path, force_rebuild=force)
    indexer.close()
    logger.info("TF-IDF index: %d notes", n)


def _index_vectors(vault_path: Path, chroma_path: Path, force: bool) -> None:
    try:
        from src.search.vector_embedder import VectorEmbedder

        vec = VectorEmbedder(chroma_path=chroma_path)
        if vec.is_available():
            logger.info("Building vector index …")
            n = vec.embed_vault(vault_path, force_rebuild=force)
            logger.info("Vector index: %d notes", n)
        else:
            logger.info(
                "sentence-transformers / chromadb not installed; "
                "skipping vector index"
            )
    except Exception as exc:
        logger.warning("Vector indexing skipped: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the Digital Brain RAG server"
    )
    parser.add_argument(
        "--port", type=int, default=8742, help="HTTP port (default: 8742)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--vault-path",
        default="~/Desktop/claude-vault-output",
        help="Path to the Obsidian vault",
    )
    parser.add_argument(
        "--db-path",
        default=".dbp_cache/embeddings.db",
        help="SQLite embeddings database path",
    )
    parser.add_argument(
        "--chroma-path",
        default="data/chroma",
        help="ChromaDB persistence directory (empty string to disable)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the search index before starting",
    )
    args = parser.parse_args()

    vault_path = Path(args.vault_path).expanduser()
    db_path = Path(args.db_path)

    if args.rebuild or not db_path.exists():
        _index_vault(vault_path, db_path, force=args.rebuild)
        if args.chroma_path:
            _index_vectors(vault_path, Path(args.chroma_path), force=args.rebuild)

    # Pass config to the server via environment variables
    os.environ["RAG_VAULT_PATH"] = str(vault_path)
    os.environ["RAG_DB_PATH"] = str(db_path)
    os.environ["RAG_CHROMA_PATH"] = args.chroma_path

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip3 install fastapi uvicorn")
        sys.exit(1)

    logger.info(
        "Starting RAG server on http://%s:%d  (vault: %s, docs: %s)",
        args.host,
        args.port,
        vault_path,
        db_path,
    )
    uvicorn.run(
        "src.rag.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
