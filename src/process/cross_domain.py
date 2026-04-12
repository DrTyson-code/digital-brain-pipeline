"""Cross-domain synthesis — surface hidden connections across knowledge domains.

Approach:
1. Classify each concept and entity into one or more domains using keyword
   matching (fuzzy multi-domain assignment allowed).
2. Find "bridges" — pairs of items from *different* domains that are
   semantically similar (TF cosine similarity above threshold).
3. Generate synthesis notes for each bridge with templated explanations
   and wikilinks to source notes.
4. Write notes to a ``Cross-Domain Synthesis/`` folder in the vault with
   YAML frontmatter compatible with Obsidian and Dataview.
5. Generate a ``Cross-Domain Synthesis MOC.md`` indexing all bridges by
   domain pair.
6. Optional: ``synthesize_with_llm()`` enriches explanations using the
   configured LLM provider.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from src.models.concept import Concept
from src.models.entity import Entity
from src.search.embedder import NoteEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

#: Core domain keyword lists.  Each item is a lowercase substring that is
#: checked against the full text of a concept/entity via ``in`` membership.
CORE_DOMAINS: Dict[str, List[str]] = {
    "medicine": [
        "anesthesia", "anesthesiology", "anesthesiologist", "medical", "patient",
        "surgery", "drug", "pharmacology", "physiology", "clinical", "hospital",
        "dose", "medication", "intubation", "sedation", "pain", "opioid", "airway",
        "monitoring", "ventilation", "cardiac", "respiratory", "perioperative",
        "intraoperative", "postoperative", "crna", "attending", "resident",
        "icu", "operating room", "fentanyl", "propofol", "ketamine", "midazolam",
    ],
    "reef_keeping": [
        "reef", "coral", "aquarium", "tank", "saltwater", "marine", "fish",
        "invertebrate", "alkalinity", "calcium", "magnesium", "nitrate", "phosphate",
        "salinity", "dosing pump", "skimmer", "sump", "refugium", "lighting",
        "acropora", "euphyllia", "zoanthid", "clownfish", "wrasse", "rodi",
        "water change", "beneficial bacteria", "par meter",
    ],
    "business": [
        "revenue", "startup", "business", "entrepreneurship", "marketing", "strategy",
        "customer", "sales", "product", "pricing", "brand", "growth",
        "boundary press", "publishing", "author", "manuscript", "isbn",
        "sba", "loan", "profit", "cash flow", "invoice", "vendor", "contract",
    ],
    "fitness": [
        "workout", "exercise", "training", "gym", "strength", "cardio", "running",
        "lifting", "muscle", "protein", "macros", "calories", "recovery", "sleep",
        "athletic", "conditioning", "mobility", "flexibility", "rep", "set",
        "deadlift", "squat", "bench press",
    ],
    "finance": [
        "investment", "portfolio", "stock", "bond", "etf", "retirement", "401k",
        "ira", "roth", "savings", "debt", "mortgage", "dividend",
        "net worth", "financial planning", "tax", "income", "expense",
        "emergency fund", "compound interest", "asset allocation",
    ],
    "technology": [
        "python", "javascript", "typescript", "software", "code", "programming",
        "api", "database", "git", "docker", "kubernetes", "cloud", "aws", "gcp",
        "machine learning", "artificial intelligence", "llm", "automation",
        "algorithm", "deployment", "server", "framework", "library",
    ],
    "personal": [
        "journal", "goal", "habit", "routine", "productivity", "mindset",
        "reflection", "personal growth", "family", "relationship", "travel",
        "learning", "meditation", "gratitude", "identity", "values",
    ],
}

# Bridge type labels
BRIDGE_PATTERN = "pattern"
BRIDGE_ENTITY = "entity"
BRIDGE_CONCEPT = "concept"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DomainBridge:
    """A discovered connection between two different domains."""

    domain_a: str
    domain_b: str
    item_a_content: str      # Full text content from domain_a
    item_b_content: str      # Full text content from domain_b
    item_a_title: str        # Short label used as wikilink target
    item_b_title: str
    bridge_type: str         # BRIDGE_PATTERN | BRIDGE_ENTITY | BRIDGE_CONCEPT
    similarity: float        # Cosine similarity [0, 1]
    confidence: float        # Derived from similarity + concept confidence
    common_tokens: List[str] # Shared tokens driving the similarity


@dataclass
class SynthesisNote:
    """A single cross-domain synthesis note ready to write to the vault."""

    title: str
    domain_a: str
    domain_b: str
    bridge_type: str
    confidence: float
    similarity: float
    source_notes: List[str]  # Wikilink strings: "[[Note Title]]"
    common_tokens: List[str]
    explanation: str         # Human-readable synthesis text
    discovered: str          # ISO date string (YYYY-MM-DD)


@dataclass
class SynthesisResult:
    """Output bag from the cross-domain synthesis stage."""

    bridges: List[DomainBridge] = field(default_factory=list)
    notes: List[SynthesisNote] = field(default_factory=list)
    written_paths: List[Path] = field(default_factory=list)
    moc_path: Optional[Path] = None

    @property
    def summary(self) -> str:
        return (
            f"Cross-domain synthesis: {len(self.bridges)} bridge(s) found, "
            f"{len(self.notes)} synthesis note(s), "
            f"{len(self.written_paths)} file(s) written"
        )


# ---------------------------------------------------------------------------
# Vector math helpers (pure Python, no numpy)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    if dot == 0.0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _top_shared_tokens(
    a: Dict[str, float], b: Dict[str, float], top_n: int = 8
) -> List[str]:
    """Return the top-N tokens shared between two TF vectors, ranked by combined weight."""
    shared = {t: a[t] + b[t] for t in a if t in b}
    return sorted(shared, key=lambda t: shared[t], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------


@dataclass
class CrossDomainSynthesizer:
    """Find and surface hidden connections across the user's knowledge domains.

    Workflow:
    1. Assign domains to each concept/entity via keyword matching.
    2. For each ordered pair of domains (A, B), compare every item in A
       against every item in B using TF cosine similarity.
    3. Pairs above ``similarity_threshold`` become DomainBridge objects.
    4. Bridges are rendered as SynthesisNote objects and written to the vault.
    5. A MOC note indexes all synthesis notes by domain pair.
    """

    similarity_threshold: float = 0.25
    min_bridge_confidence: float = 0.20
    max_bridges_per_pair: int = 5
    max_common_tokens: int = 8
    domains: Dict[str, List[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in CORE_DOMAINS.items()}
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_domains(self, content: str) -> List[str]:
        """Return all domain names whose keywords appear in *content*.

        A domain matches when any of its keywords is found as a case-insensitive
        substring.  Multiple domains may match for the same content.
        """
        content_lower = content.lower()
        return [
            domain
            for domain, keywords in self.domains.items()
            if any(kw in content_lower for kw in keywords)
        ]

    def classify_objects(
        self,
        entities: List[Entity],
        concepts: List[Concept],
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """Assign each object to one or more domains.

        Returns:
            Mapping of domain → list of ``(title, full_text, confidence)``
            tuples.  An object may appear under multiple domains.
        """
        domain_map: Dict[str, List[Tuple[str, str, float]]] = {
            d: [] for d in self.domains
        }

        for entity in entities:
            text = entity.name
            if entity.properties:
                text += " " + " ".join(str(v) for v in entity.properties.values())
            for domain in self.detect_domains(text):
                domain_map[domain].append((entity.name, text, 1.0))

        for concept in concepts:
            text = concept.content
            if concept.context:
                text += " " + concept.context
            for domain in self.detect_domains(text):
                domain_map[domain].append(
                    (concept.content[:80], text, concept.confidence)
                )

        counts = {d: len(v) for d, v in domain_map.items() if v}
        logger.info(
            "Domain classification: %s",
            ", ".join(f"{d}={n}" for d, n in sorted(counts.items())),
        )
        return domain_map

    def find_bridges(
        self,
        entities: List[Entity],
        concepts: List[Concept],
    ) -> List[DomainBridge]:
        """Discover cross-domain connections using TF cosine similarity.

        For every ordered pair of domains (A, B), each item in A is compared
        against each item in B.  Pairs meeting both the similarity and
        confidence thresholds are returned as DomainBridge objects, capped at
        ``max_bridges_per_pair`` per domain pair (highest confidence first).

        Returns:
            List of DomainBridge objects sorted by confidence descending.
        """
        domain_map = self.classify_objects(entities, concepts)

        # Pre-compute TF vectors keyed by "domain::title"
        vectors: Dict[str, Tuple[str, Dict[str, float], float]] = {}
        for domain, items in domain_map.items():
            for title, content, conf in items:
                key = f"{domain}::{title}"
                if key not in vectors:
                    vectors[key] = (content, NoteEmbedder._compute_tf(content), conf)

        entity_names: Set[str] = {e.name for e in entities}
        domain_names = sorted(domain_map.keys())
        bridges: List[DomainBridge] = []

        for i, domain_a in enumerate(domain_names):
            for domain_b in domain_names[i + 1:]:
                items_a = domain_map[domain_a]
                items_b = domain_map[domain_b]
                if not items_a or not items_b:
                    continue

                pair_bridges: List[DomainBridge] = []

                for title_a, content_a, conf_a in items_a:
                    key_a = f"{domain_a}::{title_a}"
                    _, vec_a, _ = vectors[key_a]
                    if not vec_a:
                        continue

                    for title_b, content_b, conf_b in items_b:
                        # Skip items that cross-classified from the same source
                        if title_a == title_b and content_a == content_b:
                            continue

                        key_b = f"{domain_b}::{title_b}"
                        _, vec_b, _ = vectors[key_b]
                        if not vec_b:
                            continue

                        sim = _cosine_similarity(vec_a, vec_b)
                        if sim < self.similarity_threshold:
                            continue

                        # Confidence = geometric mean of item confidences,
                        # boosted by similarity (capped at 1.0)
                        conf = round(
                            math.sqrt(conf_a * conf_b) * min(sim * 1.5, 1.0), 3
                        )
                        if conf < self.min_bridge_confidence:
                            continue

                        common = _top_shared_tokens(vec_a, vec_b, self.max_common_tokens)

                        if title_a in entity_names or title_b in entity_names:
                            btype = BRIDGE_ENTITY
                        elif len(common) >= 3:
                            btype = BRIDGE_PATTERN
                        else:
                            btype = BRIDGE_CONCEPT

                        pair_bridges.append(
                            DomainBridge(
                                domain_a=domain_a,
                                domain_b=domain_b,
                                item_a_content=content_a,
                                item_b_content=content_b,
                                item_a_title=title_a,
                                item_b_title=title_b,
                                bridge_type=btype,
                                similarity=round(sim, 3),
                                confidence=conf,
                                common_tokens=common,
                            )
                        )

                pair_bridges.sort(key=lambda b: b.confidence, reverse=True)
                kept = pair_bridges[: self.max_bridges_per_pair]
                bridges.extend(kept)

                if kept:
                    logger.debug(
                        "Domain pair %s <-> %s: %d bridge(s) (top sim=%.2f)",
                        domain_a,
                        domain_b,
                        len(kept),
                        kept[0].similarity,
                    )

        bridges.sort(key=lambda b: b.confidence, reverse=True)
        logger.info(
            "Cross-domain bridge detection: %d bridge(s) across %d domain pair(s)",
            len(bridges),
            len({(b.domain_a, b.domain_b) for b in bridges}),
        )
        return bridges

    def generate_synthesis_notes(
        self,
        bridges: List[DomainBridge],
        discovered_date: Optional[str] = None,
    ) -> List[SynthesisNote]:
        """Convert DomainBridge objects into SynthesisNote objects.

        Uses rule-based template explanations.  For LLM-enriched text see
        ``synthesize_with_llm()``.
        """
        if discovered_date is None:
            discovered_date = datetime.now().strftime("%Y-%m-%d")

        notes: List[SynthesisNote] = []
        seen_titles: Set[str] = set()

        for bridge in bridges:
            title = self._bridge_title(bridge)
            if title in seen_titles:
                continue
            seen_titles.add(title)

            notes.append(
                SynthesisNote(
                    title=title,
                    domain_a=bridge.domain_a,
                    domain_b=bridge.domain_b,
                    bridge_type=bridge.bridge_type,
                    confidence=bridge.confidence,
                    similarity=bridge.similarity,
                    source_notes=[
                        f"[[{bridge.item_a_title}]]",
                        f"[[{bridge.item_b_title}]]",
                    ],
                    common_tokens=bridge.common_tokens,
                    explanation=self._explain_bridge(bridge),
                    discovered=discovered_date,
                )
            )

        logger.info(
            "Generated %d synthesis note(s) from %d bridge(s)",
            len(notes),
            len(bridges),
        )
        return notes

    async def synthesize_with_llm(
        self,
        bridges: List[DomainBridge],
        provider: object,  # LLMProvider — typed loosely to avoid circular import
        discovered_date: Optional[str] = None,
    ) -> List[SynthesisNote]:
        """LLM-enhanced synthesis notes with richer explanations.

        Calls the provider for each bridge.  On any individual failure, falls
        back to the rule-based explanation for that note and continues.

        Args:
            bridges: Bridges from ``find_bridges()``.
            provider: An ``LLMProvider`` instance (Claude, OpenAI, Ollama).
            discovered_date: ISO date string; defaults to today.

        Returns:
            List of SynthesisNote objects, LLM-enriched where possible.
        """
        if not bridges:
            return []

        baseline = self.generate_synthesis_notes(bridges, discovered_date)
        enhanced: List[SynthesisNote] = []

        for note, bridge in zip(baseline, bridges):
            try:
                llm_text = await self._llm_explain_bridge(bridge, provider)
                enhanced.append(
                    SynthesisNote(
                        title=note.title,
                        domain_a=note.domain_a,
                        domain_b=note.domain_b,
                        bridge_type=note.bridge_type,
                        confidence=note.confidence,
                        similarity=note.similarity,
                        source_notes=note.source_notes,
                        common_tokens=note.common_tokens,
                        explanation=llm_text,
                        discovered=note.discovered,
                    )
                )
                logger.debug("LLM enriched synthesis note: %r", note.title)
            except Exception as exc:
                logger.warning(
                    "LLM enhancement failed for %r; using rule-based: %s",
                    note.title,
                    exc,
                )
                enhanced.append(note)

        return enhanced

    def write_to_vault(
        self,
        notes: List[SynthesisNote],
        vault_path: Path,
        tag_prefix: str = "ai-brain",
    ) -> List[Path]:
        """Write synthesis notes to ``<vault>/Cross-Domain Synthesis/``.

        Returns:
            List of paths that were written successfully.
        """
        vault_path = Path(vault_path).expanduser()
        folder = vault_path / "Cross-Domain Synthesis"
        folder.mkdir(parents=True, exist_ok=True)

        written: List[Path] = []
        for note in notes:
            path = self._write_note(note, folder, tag_prefix)
            if path:
                written.append(path)

        logger.info("Wrote %d synthesis note(s) to %s", len(written), folder)
        return written

    def generate_moc(
        self,
        notes: List[SynthesisNote],
        vault_path: Path,
        tag_prefix: str = "ai-brain",
        date_format: str = "%Y-%m-%d",
    ) -> Optional[Path]:
        """Generate ``Cross-Domain Synthesis MOC.md`` in the vault root.

        Notes are grouped by domain pair and sorted by confidence.
        Returns None if there are no notes to index.
        """
        if not notes:
            logger.info("No synthesis notes — skipping MOC generation")
            return None

        vault_path = Path(vault_path).expanduser()
        moc_path = vault_path / "Cross-Domain Synthesis MOC.md"
        today = datetime.now().strftime(date_format)

        frontmatter: Dict[str, object] = {
            "title": "Cross-Domain Synthesis MOC",
            "type": "moc",
            "tags": [f"{tag_prefix}/moc", f"{tag_prefix}/synthesis"],
            "created": today,
            "total_bridges": len(notes),
        }
        fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)

        # Group by domain pair
        by_pair: Dict[Tuple[str, str], List[SynthesisNote]] = {}
        for note in notes:
            by_pair.setdefault((note.domain_a, note.domain_b), []).append(note)

        all_domains: Set[str] = set()
        for note in notes:
            all_domains.add(note.domain_a)
            all_domains.add(note.domain_b)

        lines: List[str] = [
            f"---\n{fm_str}---\n",
            "# Cross-Domain Synthesis MOC\n",
            "> Connections that cross knowledge domain boundaries — unexpected bridges "
            "between medicine, reef keeping, business, fitness, finance, and technology.\n",
            f"generated:: {today}",
            f"total_bridges:: {len(notes)}\n",
            "## Domain Coverage\n",
            f"Active domains: {', '.join(sorted(all_domains))}\n",
            "## Connections by Domain Pair\n",
        ]

        for (domain_a, domain_b), pair_notes in sorted(by_pair.items()):
            label = (
                f"{domain_a.replace('_', ' ').title()} "
                f"<-> "
                f"{domain_b.replace('_', ' ').title()}"
            )
            lines.append(f"### {label}\n")
            for note in sorted(pair_notes, key=lambda n: n.confidence, reverse=True):
                conf_pct = int(note.confidence * 100)
                type_label = f"[{note.bridge_type[0].upper()}]"
                lines.append(
                    f"- {type_label} [[{note.title}]] "
                    f"(confidence: {conf_pct}%, type: {note.bridge_type})"
                )
            lines.append("")

        # Dataview block for dynamic updates
        lines += [
            "## All Synthesis Notes (Dataview)\n",
            "```dataview",
            "TABLE domains, bridge_type, confidence, discovered",
            'FROM "Cross-Domain Synthesis"',
            "SORT confidence DESC",
            "```\n",
        ]

        moc_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Cross-domain MOC written to %s", moc_path)
        return moc_path

    def run(
        self,
        entities: List[Entity],
        concepts: List[Concept],
        vault_path: Path,
        tag_prefix: str = "ai-brain",
        date_format: str = "%Y-%m-%d",
    ) -> SynthesisResult:
        """Full synthesis pipeline: detect bridges -> generate notes -> write vault.

        Returns:
            SynthesisResult with bridges, notes, and written file paths.
        """
        bridges = self.find_bridges(entities, concepts)

        if not bridges:
            logger.info("No cross-domain bridges found — skipping synthesis output")
            return SynthesisResult()

        notes = self.generate_synthesis_notes(bridges)
        written = self.write_to_vault(notes, vault_path, tag_prefix)
        moc_path = self.generate_moc(notes, vault_path, tag_prefix, date_format)

        result = SynthesisResult(
            bridges=bridges,
            notes=notes,
            written_paths=written,
            moc_path=moc_path,
        )
        logger.info(result.summary)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bridge_title(self, bridge: DomainBridge) -> str:
        """Produce a short descriptive title for a bridge."""
        a = bridge.domain_a.replace("_", "-")
        b = bridge.domain_b.replace("_", "-")
        if bridge.common_tokens:
            token_str = " + ".join(bridge.common_tokens[:3])
            return f"{a} x {b}: {token_str}"[:100]
        return f"{a} x {b}: {bridge.item_a_title[:30]}"[:100]

    def _explain_bridge(self, bridge: DomainBridge) -> str:
        """Generate a rule-based explanation for a cross-domain bridge."""
        a_label = bridge.domain_a.replace("_", " ")
        b_label = bridge.domain_b.replace("_", " ")
        token_note = (
            f" (shared concepts: *{', '.join(bridge.common_tokens[:5])}*)"
            if bridge.common_tokens
            else ""
        )

        if bridge.bridge_type == BRIDGE_ENTITY:
            return (
                f"**{bridge.item_a_title}** appears in both the *{a_label}* and "
                f"*{b_label}* domains{token_note}.\n\n"
                f"This entity forms a structural bridge — knowledge from one domain "
                f"may transfer to the other through it.\n\n"
                f"- In **{a_label}**: {bridge.item_a_content[:200]}\n"
                f"- In **{b_label}**: {bridge.item_b_content[:200]}"
            )
        elif bridge.bridge_type == BRIDGE_PATTERN:
            return (
                f"A shared *pattern* links **{a_label}** and **{b_label}**"
                f"{token_note}.\n\n"
                f"The same underlying principle appears to operate in both domains, "
                f"suggesting that strategies from one may apply in the other.\n\n"
                f"- **{a_label}**: {bridge.item_a_content[:200]}\n"
                f"- **{b_label}**: {bridge.item_b_content[:200]}"
            )
        else:  # BRIDGE_CONCEPT
            return (
                f"A conceptual link connects **{a_label}** and **{b_label}**"
                f"{token_note}.\n\n"
                f"These ideas share significant semantic overlap, suggesting a "
                f"transferable insight or common principle.\n\n"
                f"- **{a_label}**: {bridge.item_a_content[:200]}\n"
                f"- **{b_label}**: {bridge.item_b_content[:200]}"
            )

    async def _llm_explain_bridge(
        self, bridge: DomainBridge, provider: object
    ) -> str:
        """Call the LLM provider to generate a richer synthesis explanation."""
        system_prompt = (
            "You are a synthesis assistant helping a user discover unexpected "
            "connections across their knowledge domains. Write a concise, insightful "
            "explanation (2-4 sentences) of how two pieces of knowledge from different "
            "domains are related. Be specific, practical, and highlight what can be "
            "learned from the connection."
        )
        a_label = bridge.domain_a.replace("_", " ")
        b_label = bridge.domain_b.replace("_", " ")
        user_prompt = (
            f"Domain A ({a_label}): {bridge.item_a_content[:300]}\n\n"
            f"Domain B ({b_label}): {bridge.item_b_content[:300]}\n\n"
            f"Shared concepts: {', '.join(bridge.common_tokens[:5])}\n\n"
            "Write a synthesis insight explaining the connection."
        )
        response = await provider.complete(  # type: ignore[attr-defined]
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256,
        )
        return response.content.strip()

    def _write_note(
        self,
        note: SynthesisNote,
        folder: Path,
        tag_prefix: str,
    ) -> Optional[Path]:
        """Write a single synthesis note to *folder*."""
        safe_title = _sanitize_filename(note.title)
        file_path = folder / f"{safe_title}.md"

        a_label = note.domain_a.replace("_", " ").title()
        b_label = note.domain_b.replace("_", " ").title()

        frontmatter: Dict[str, object] = {
            "title": note.title,
            "type": "synthesis",
            "domains": [note.domain_a, note.domain_b],
            "bridge_type": note.bridge_type,
            "confidence": note.confidence,
            "similarity": note.similarity,
            "discovered": note.discovered,
            "source_notes": note.source_notes,
            "tags": [
                f"{tag_prefix}/synthesis",
                f"{tag_prefix}/cross-domain",
                f"{tag_prefix}/{note.domain_a}",
                f"{tag_prefix}/{note.domain_b}",
            ],
        }
        fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)

        lines: List[str] = [
            f"# {note.title}\n",
            f"type:: synthesis",
            f"domains:: [[{note.domain_a}]], [[{note.domain_b}]]",
            f"bridge_type:: {note.bridge_type}",
            f"confidence:: {note.confidence}",
            f"discovered:: {note.discovered}",
            "",
            f"## Connection: {a_label} <-> {b_label}\n",
            note.explanation,
            "",
            "## Source Notes\n",
        ]
        for src in note.source_notes:
            lines.append(f"- {src}")
        lines.append("")

        if note.common_tokens:
            lines.append("## Shared Concepts\n")
            lines.append(", ".join(f"`{t}`" for t in note.common_tokens))
            lines.append("")

        content = f"---\n{fm_str}---\n\n" + "\n".join(lines)

        try:
            file_path.write_text(content, encoding="utf-8")
            return file_path
        except OSError as exc:
            logger.error("Failed to write synthesis note %s: %s", file_path, exc)
            return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "-")
    # Replace any remaining non-ASCII or special characters
    name = re.sub(r"[^\w\s\-.]", "-", name)
    name = re.sub(r"[-\s]+", "-", name).strip("-")
    return name[:180]
