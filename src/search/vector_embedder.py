"""Dense vector embedder using sentence-transformers + ChromaDB (optional).

Falls back gracefully if either dependency is missing.  Documents are
identified by their absolute file path, used as the ChromaDB document ID.

Usage::

    embedder = VectorEmbedder(chroma_path=Path("data/chroma/"))
    if embedder.is_available():
        n = embedder.embed_vault(vault_path=Path("~/Desktop/claude-vault-output"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.search.embedder import strip_frontmatter

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64
_MODEL_NAME = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "vault"


class VectorEmbedder:
    """Optional dense-vector layer over the vault.

    If ``sentence-transformers`` or ``chromadb`` are not installed, all
    methods are no-ops and :meth:`is_available` returns ``False``.
    """

    def __init__(self, chroma_path: Path = Path("data/chroma/")) -> None:
        self._chroma_path = Path(chroma_path)
        self._model = None
        self._client = None
        self._collection = None
        self._available = False

        try:
            import chromadb  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._chroma_path))
            self._collection = self._client.get_or_create_collection(
                _COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._model = SentenceTransformer(_MODEL_NAME)
            self._available = True
            logger.info("VectorEmbedder initialised (model=%s)", _MODEL_NAME)
        except ImportError:
            logger.info(
                "sentence-transformers or chromadb not installed; "
                "vector embeddings disabled"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("VectorEmbedder failed to initialise: %s", exc)

    def is_available(self) -> bool:
        """Return True if vector search dependencies are installed."""
        return self._available

    def embed_vault(
        self,
        vault_path: Path,
        force_rebuild: bool = False,
    ) -> int:
        """Embed all markdown notes in *vault_path* into ChromaDB.

        Returns the number of documents added or updated.
        """
        if not self._available:
            return 0

        vault_path = Path(vault_path).expanduser()
        if not vault_path.exists():
            logger.warning("Vault path not found: %s", vault_path)
            return 0

        md_files = list(vault_path.rglob("*.md"))
        logger.info("VectorEmbedder: found %d markdown files", len(md_files))

        assert self._client is not None
        assert self._collection is not None

        if force_rebuild:
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                _COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Find which paths are already indexed
        existing_ids: set = set()
        if not force_rebuild:
            try:
                result = self._collection.get(include=[])
                existing_ids = set(result.get("ids", []))
            except Exception:
                pass

        to_embed: List[Path] = [f for f in md_files if str(f) not in existing_ids]
        if not to_embed:
            logger.info(
                "VectorEmbedder: all %d notes already indexed", len(md_files)
            )
            return 0

        logger.info("VectorEmbedder: embedding %d new notes", len(to_embed))

        assert self._model is not None
        added = 0

        for i in range(0, len(to_embed), _BATCH_SIZE):
            batch = to_embed[i : i + _BATCH_SIZE]
            texts, ids, metas = [], [], []

            for path in batch:
                try:
                    raw = path.read_text(encoding="utf-8", errors="replace")
                    meta_dict, body = strip_frontmatter(raw)
                    title = meta_dict.get("title", path.stem)
                    # Truncate for embedding model context window
                    text = f"{title}\n{body}"[:2000]
                    texts.append(text)
                    ids.append(str(path))
                    metas.append(
                        {
                            "title": title,
                            "note_type": meta_dict.get("type", "note"),
                            "domain": meta_dict.get("domain", ""),
                            "tags": meta_dict.get("tags", ""),
                        }
                    )
                except OSError as exc:
                    logger.warning("Could not read %s: %s", path, exc)

            if not texts:
                continue

            embeddings = self._model.encode(
                texts, show_progress_bar=False
            ).tolist()
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metas,
            )
            added += len(texts)
            logger.debug("Embedded batch %d–%d (%d docs)", i, i + len(batch), added)

        logger.info("VectorEmbedder: added %d documents total", added)
        return added

    @property
    def doc_count(self) -> int:
        """Number of documents in the vector store."""
        if not self._available or self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
