"""Semantic search layer for the Digital Brain Pipeline.

Uses TF-IDF with cosine similarity (pure Python, no extra dependencies).
"""
from src.search.embedder import NoteEmbedder
from src.search.engine import VaultSearchEngine, SearchResult

__all__ = ["NoteEmbedder", "VaultSearchEngine", "SearchResult"]
