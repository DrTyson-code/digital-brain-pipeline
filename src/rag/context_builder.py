"""Build a context window from HybridSearchEngine results.

Takes ranked search results and assembles a formatted string ready to inject
into an LLM prompt.  Higher-scoring results are included first; the builder
stops adding notes once the token budget is exhausted.

Usage::

    builder = ContextBuilder(max_tokens=4000)
    context = builder.build(results, query="What do I know about ketamine?")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.search.hybrid_engine import HybridResult

# Rough approximation used throughout: 4 characters ≈ 1 token.
_CHARS_PER_TOKEN = 4


class ContextBuilder:
    """Assemble a RAG context string from a ranked list of search results.

    Parameters
    ----------
    max_tokens:
        Approximate upper bound on context length (default 4000 tokens).
    snippet_tokens:
        Maximum tokens to use per note snippet (default 300 tokens).
    include_metadata:
        Whether to prepend a YAML-style metadata block for each note
        (default True).  Disable to reduce context overhead.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        snippet_tokens: int = 300,
        include_metadata: bool = True,
    ) -> None:
        self._max_chars = max_tokens * _CHARS_PER_TOKEN
        self._snippet_chars = snippet_tokens * _CHARS_PER_TOKEN
        self._include_metadata = include_metadata

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        results: List[HybridResult],
        query: Optional[str] = None,
    ) -> str:
        """Return a formatted context string ready for LLM injection.

        Results must be pre-sorted by score descending (the HybridSearchEngine
        already returns them in this order).  The builder fills the token
        budget greedily, highest-scoring first.

        Args:
            results: Ranked search results from HybridSearchEngine.
            query: Optional query string; used to add a header line.

        Returns:
            Multi-section markdown string, or empty string if no results.
        """
        if not results:
            return ""

        header = f"# Vault context for: {query}\n\n" if query else ""
        budget = self._max_chars - len(header)
        sections: List[str] = []

        for i, result in enumerate(results):
            section = self._format_result(i + 1, result)
            if len(section) > budget:
                # Try a compressed version with a shorter snippet
                truncated_chars = max(200, budget - 300)
                section = self._format_result(
                    i + 1, result, max_snippet_chars=truncated_chars
                )
            if len(section) > budget:
                break  # budget exhausted
            sections.append(section)
            budget -= len(section)

        return (header + "\n".join(sections)).strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_result(
        self,
        rank: int,
        result: HybridResult,
        max_snippet_chars: Optional[int] = None,
    ) -> str:
        max_chars = max_snippet_chars if max_snippet_chars is not None else self._snippet_chars
        note_name = Path(result.note_path).name

        lines: List[str] = [
            f"## [{rank}] {result.title}  (score: {result.score:.3f})",
        ]

        if self._include_metadata:
            meta_parts: List[str] = []
            if result.note_type:
                meta_parts.append(f"type: {result.note_type}")
            if result.domain:
                meta_parts.append(f"domain: {result.domain}")
            if result.tags:
                meta_parts.append(f"tags: {result.tags}")
            meta_parts.append(f"source: {note_name}")
            lines.append("```yaml")
            lines.extend(meta_parts)
            lines.append("```")

        snippet = result.snippet
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "…"
        if snippet:
            lines.append(snippet)
        lines.append("")  # blank line between sections

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return max(1, len(text) // _CHARS_PER_TOKEN)
