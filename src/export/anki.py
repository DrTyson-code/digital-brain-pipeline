"""Anki flashcard generator for the Digital Brain Pipeline.

Generates Anki-compatible flashcards from:
- Stale concepts (flagged by TemporalTracker.detect_stale_knowledge) → review::urgent
- High-confidence concepts (confidence >= threshold)
- Cross-domain synthesis notes (bridge concepts)
- Vault note files via YAML frontmatter scan (standalone / CLI usage)

Output formats:
- TSV (tab-separated, Basic note type): always available, Anki-importable
- .apkg (Anki package): requires genanki (graceful fallback to TSV if not installed)

Card types:
- Basic: front/back question-answer pair
- Cloze: {{c1::term}} deletion — used in .apkg; rendered as basic in TSV
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

try:
    import genanki
    GENANKI_AVAILABLE = True
except ImportError:
    GENANKI_AVAILABLE = False

from src.models.concept import Concept, ConceptType
from src.process.cross_domain import SynthesisNote
from src.process.temporal_tracker import ConceptStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECK_PREFIX = "DigitalBrain"

#: Map domain names (from CrossDomainSynthesizer) to Anki sub-deck names.
DOMAIN_DECK_MAP: Dict[str, str] = {
    "medicine": "Medicine",
    "reef_keeping": "ReefKeeping",
    "business": "Business",
    "fitness": "Fitness",
    "finance": "Finance",
    "technology": "Technology",
    "personal": "Personal",
}

#: Domain keywords for simple content-based domain detection.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "medicine": [
        "anesthesia", "anesthesiology", "medical", "patient", "surgery", "drug",
        "pharmacology", "clinical", "hospital", "dose", "medication", "sedation",
        "pain", "opioid", "airway", "cardiac", "respiratory", "perioperative",
        "fentanyl", "propofol", "ketamine",
    ],
    "reef_keeping": [
        "reef", "coral", "aquarium", "tank", "saltwater", "marine", "fish",
        "alkalinity", "calcium", "magnesium", "nitrate", "salinity", "skimmer",
    ],
    "business": [
        "revenue", "startup", "business", "entrepreneurship", "marketing", "strategy",
        "customer", "sales", "product", "pricing", "brand", "publishing",
    ],
    "fitness": [
        "workout", "exercise", "training", "gym", "strength", "cardio", "running",
        "lifting", "muscle", "protein", "macros", "calories",
    ],
    "finance": [
        "investment", "portfolio", "stock", "bond", "retirement", "401k", "ira",
        "savings", "debt", "mortgage", "dividend", "financial planning", "tax",
    ],
    "technology": [
        "python", "javascript", "typescript", "software", "code", "programming",
        "api", "database", "git", "docker", "cloud", "machine learning", "llm",
    ],
    "personal": [
        "journal", "goal", "habit", "routine", "productivity", "mindset",
        "reflection", "personal growth", "family", "relationship",
    ],
}

#: Stable Anki model IDs (arbitrary fixed integers required by genanki).
_BASIC_MODEL_ID = 1607392319
_CLOZE_MODEL_ID = 1607392320


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnkiCard:
    """A single Anki flashcard ready for export."""

    front: str
    back: str
    card_type: str  # "basic" or "cloze"
    tags: List[str] = field(default_factory=list)
    deck: str = DECK_PREFIX
    source_id: str = ""  # original concept/note ID for dedup tracking
    source_type: str = "concept"  # "concept" | "synthesis" | "vault_note"

    @property
    def sanitized_tags(self) -> List[str]:
        """Tags with spaces → underscores and / → :: (Anki conventions)."""
        return [t.replace(" ", "_").replace("/", "::") for t in self.tags]

    @property
    def content_hash(self) -> str:
        """SHA-256 of front+back for change detection."""
        digest = hashlib.sha256(f"{self.front}\t{self.back}".encode()).hexdigest()
        return digest[:16]

    @property
    def tsv_row(self) -> str:
        """Format as a single TSV row for Anki Basic import.

        Columns: Front, Back, Tags
        Cloze cards are normalised to basic format in TSV output; the cloze
        markup is stripped from the front and retained in the back field so
        the full original text is visible on the answer side.
        """
        front = _strip_cloze_markers(self.front) if self.card_type == "cloze" else self.front
        back = self.back if self.card_type != "cloze" else f"{self.front}\n\n{self.back}"
        tags_str = " ".join(self.sanitized_tags)
        return f"{front}\t{back}\t{tags_str}"


# ---------------------------------------------------------------------------
# Card generator
# ---------------------------------------------------------------------------


@dataclass
class AnkiCardGenerator:
    """Generate AnkiCard objects from pipeline data or vault notes.

    Priority sources (in order of importance):
    1. Stale concepts → ``review::urgent`` tag
    2. High-confidence concepts (>= min_confidence)
    3. Cross-domain synthesis notes
    4. Vault note scan (for standalone CLI usage)
    """

    min_confidence: float = 0.7
    tag_prefix: str = "ai-brain"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_concepts(
        self,
        concepts: List[Concept],
        stale_ids: Optional[Set[str]] = None,
        concept_statuses: Optional[Dict[str, ConceptStatus]] = None,
        domain_filter: Optional[List[str]] = None,
    ) -> List[AnkiCard]:
        """Generate flashcards from extracted Concept objects.

        Args:
            concepts: All extracted concepts from the pipeline.
            stale_ids: Set of concept IDs flagged as stale by TemporalTracker.
            concept_statuses: Full status map (concept_id → ConceptStatus).
            domain_filter: If set, only include cards matching these domains.

        Returns:
            List of AnkiCard objects sorted with stale cards first.
        """
        _stale = stale_ids or set()
        cards: List[AnkiCard] = []

        for concept in concepts:
            is_stale = concept.id in _stale
            is_high_conf = concept.confidence >= self.min_confidence

            if not is_stale and not is_high_conf:
                continue

            # Skip superseded concepts unless they're stale
            if concept_statuses:
                status = concept_statuses.get(concept.id)
                if status and status.status == "superseded" and not is_stale:
                    continue

            domain = _detect_domain(concept.content, concept.tags)
            if domain_filter and domain not in domain_filter and domain != "general":
                continue

            card = self._concept_to_card(concept, is_stale, domain)
            cards.append(card)

        # Stale cards first (highest review priority)
        cards.sort(key=lambda c: ("review::urgent" not in c.tags, c.source_id))
        logger.info(
            "Generated %d flashcard(s) from %d concept(s) (%d stale)",
            len(cards),
            len(concepts),
            len(_stale),
        )
        return cards

    def generate_from_synthesis_notes(
        self,
        notes: List[SynthesisNote],
        domain_filter: Optional[List[str]] = None,
    ) -> List[AnkiCard]:
        """Generate flashcards from cross-domain synthesis notes.

        Synthesis cards are always treated as high value — no confidence filter.
        """
        cards: List[AnkiCard] = []
        for note in notes:
            if domain_filter and (
                note.domain_a not in domain_filter and note.domain_b not in domain_filter
            ):
                continue
            card = self._synthesis_note_to_card(note)
            cards.append(card)

        logger.info("Generated %d flashcard(s) from synthesis notes", len(cards))
        return cards

    def generate_from_vault(
        self,
        vault_path: Path,
        domain_filter: Optional[List[str]] = None,
        stale_only: bool = False,
    ) -> List[AnkiCard]:
        """Generate flashcards by scanning Obsidian vault note files.

        Reads YAML frontmatter from all Markdown files in the vault.
        Skips files with missing or malformed frontmatter.

        Args:
            vault_path: Root path of the Obsidian vault.
            domain_filter: If set, only include notes matching these domains.
            stale_only: If True, only include notes with ``status: stale``.

        Returns:
            List of AnkiCard objects.
        """
        vault_path = Path(vault_path).expanduser()
        if not vault_path.exists():
            logger.warning("Vault path does not exist: %s", vault_path)
            return []

        cards: List[AnkiCard] = []
        md_files = list(vault_path.rglob("*.md"))
        logger.debug("Scanning %d vault files in %s", len(md_files), vault_path)

        for md_file in md_files:
            card = self._vault_note_to_card(
                md_file, domain_filter=domain_filter, stale_only=stale_only
            )
            if card is not None:
                cards.append(card)

        stale_count = sum(1 for c in cards if "review::urgent" in c.tags)
        logger.info(
            "Generated %d flashcard(s) from vault (%d stale)",
            len(cards),
            stale_count,
        )
        return cards

    # ------------------------------------------------------------------
    # Concept → card
    # ------------------------------------------------------------------

    def _concept_to_card(
        self,
        concept: Concept,
        is_stale: bool,
        domain: str,
    ) -> AnkiCard:
        """Convert a single Concept to an AnkiCard."""
        deck = _domain_to_deck(domain)
        tags = self._base_tags(domain, is_stale)
        tags.append(f"{self.tag_prefix}/concept/{concept.concept_type.value}")

        # Try cloze first for definition-style content
        cloze_front = _try_cloze_deletion(concept.content)
        if cloze_front:
            return AnkiCard(
                front=cloze_front,
                back=concept.context or "",
                card_type="cloze",
                tags=tags,
                deck=deck,
                source_id=concept.id,
                source_type="concept",
            )

        front, back = _concept_front_back(concept)
        return AnkiCard(
            front=front,
            back=back,
            card_type="basic",
            tags=tags,
            deck=deck,
            source_id=concept.id,
            source_type="concept",
        )

    # ------------------------------------------------------------------
    # Synthesis note → card
    # ------------------------------------------------------------------

    def _synthesis_note_to_card(self, note: SynthesisNote) -> AnkiCard:
        """Convert a SynthesisNote to a cross-domain AnkiCard."""
        domain_a = note.domain_a.replace("_", " ").title()
        domain_b = note.domain_b.replace("_", " ").title()

        if note.common_tokens:
            topic = " + ".join(note.common_tokens[:3])
            front = f"How do {domain_a} and {domain_b} connect through '{topic}'?"
        else:
            front = f"What is the cross-domain connection between {domain_a} and {domain_b}?"

        back = note.explanation or note.title

        tags = [
            f"{self.tag_prefix}/synthesis",
            f"{self.tag_prefix}/cross-domain",
            f"domain::{note.domain_a}",
            f"domain::{note.domain_b}",
        ]
        # Synthesis notes are always high-value — treat as urgent if very recent
        deck = f"{DECK_PREFIX}::CrossDomain"

        # Stable source_id from title hash
        source_id = hashlib.sha256(note.title.encode()).hexdigest()[:12]

        return AnkiCard(
            front=front,
            back=back,
            card_type="basic",
            tags=tags,
            deck=deck,
            source_id=source_id,
            source_type="synthesis",
        )

    # ------------------------------------------------------------------
    # Vault note → card
    # ------------------------------------------------------------------

    def _vault_note_to_card(
        self,
        md_file: Path,
        domain_filter: Optional[List[str]],
        stale_only: bool,
    ) -> Optional[AnkiCard]:
        """Parse a vault Markdown file and return an AnkiCard or None."""
        try:
            raw = md_file.read_text(encoding="utf-8")
        except OSError:
            return None

        frontmatter, body = _parse_frontmatter(raw)
        if not frontmatter:
            return None

        note_type = frontmatter.get("type", "")
        confidence = float(frontmatter.get("confidence", 0.0))
        status = frontmatter.get("status", "active")
        tags_raw: List[str] = frontmatter.get("tags") or []
        title: str = frontmatter.get("title") or md_file.stem

        is_stale = status == "stale"
        is_synthesis = note_type == "synthesis"

        # Apply stale filter
        if stale_only and not is_stale:
            return None

        # Confidence gate (synthesis notes bypass it)
        if not is_synthesis and confidence < self.min_confidence and not is_stale:
            return None

        # Domain detection
        domain = _detect_domain(title + " " + body[:200], tags_raw)
        if domain_filter and domain not in domain_filter and not is_synthesis:
            return None

        deck = _domain_to_deck(domain) if not is_synthesis else f"{DECK_PREFIX}::CrossDomain"
        tags = list(tags_raw)
        if is_stale:
            tags.append("review::urgent")
        if is_synthesis:
            tags.append(f"{self.tag_prefix}/synthesis")

        # Build front/back from note type and content
        front, back = _vault_front_back(title, note_type, body)
        if not front or not back:
            return None

        source_id = hashlib.sha256(str(md_file).encode()).hexdigest()[:12]

        return AnkiCard(
            front=front,
            back=back,
            card_type="basic",
            tags=tags,
            deck=deck,
            source_id=source_id,
            source_type="vault_note",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _base_tags(self, domain: str, is_stale: bool) -> List[str]:
        tags = [f"{self.tag_prefix}", f"domain::{domain}"]
        if is_stale:
            tags.append("review::urgent")
        return tags


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


@dataclass
class AnkiExporter:
    """Write AnkiCard lists to TSV or .apkg format."""

    def export(
        self,
        cards: List[AnkiCard],
        output_path: Path,
        fmt: str = "tsv",
        dry_run: bool = False,
    ) -> Path:
        """Export cards to *output_path* in the specified format.

        Args:
            cards: Cards to export.
            output_path: Destination file path.
            fmt: "tsv" or "apkg".
            dry_run: If True, skip writing and return the target path.

        Returns:
            Path that was (or would have been) written.
        """
        output_path = Path(output_path)
        if fmt == "apkg":
            if GENANKI_AVAILABLE:
                return self.to_apkg(cards, output_path, dry_run=dry_run)
            else:
                logger.warning(
                    "genanki not installed — falling back to TSV. "
                    "Install with: pip install genanki"
                )
                tsv_path = output_path.with_suffix(".txt")
                return self.to_tsv(cards, tsv_path, dry_run=dry_run)
        return self.to_tsv(cards, output_path, dry_run=dry_run)

    def to_tsv(
        self,
        cards: List[AnkiCard],
        output_path: Path,
        dry_run: bool = False,
    ) -> Path:
        """Write cards as an Anki-importable TSV file (Basic note type).

        Header directives tell Anki the separator, HTML setting, and note type
        so the import dialog pre-populates correctly.

        Format:
            #separator:tab
            #html:false
            #notetype:Basic
            Front<TAB>Back<TAB>Tags
        """
        lines = [
            "#separator:tab",
            "#html:false",
            "#notetype:Basic",
            f"#generated:{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"#cards:{len(cards)}",
        ]
        for card in cards:
            lines.append(card.tsv_row)

        content = "\n".join(lines) + "\n"

        if dry_run:
            logger.info("[dry-run] Would write %d card(s) to %s", len(cards), output_path)
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info("Wrote %d card(s) to %s", len(cards), output_path)
        return output_path

    def to_apkg(
        self,
        cards: List[AnkiCard],
        output_path: Path,
        dry_run: bool = False,
    ) -> Path:
        """Write cards as an Anki .apkg package using genanki.

        Creates separate Basic and Cloze note types. One genanki Deck is
        created per unique ``card.deck`` value encountered in the card list.

        Raises:
            RuntimeError: If genanki is not installed.
        """
        if not GENANKI_AVAILABLE:
            raise RuntimeError(
                "genanki is not installed. Install with: pip install genanki"
            )

        if dry_run:
            logger.info("[dry-run] Would write %d card(s) to %s (apkg)", len(cards), output_path)
            return output_path

        basic_model = genanki.Model(
            _BASIC_MODEL_ID,
            "DigitalBrain Basic",
            fields=[{"name": "Front"}, {"name": "Back"}],
            templates=[{
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}<hr id=answer>{{Back}}",
            }],
        )
        cloze_model = genanki.Model(
            _CLOZE_MODEL_ID,
            "DigitalBrain Cloze",
            fields=[{"name": "Text"}, {"name": "Extra"}],
            templates=[{
                "name": "Cloze",
                "qfmt": "{{cloze:Text}}",
                "afmt": "{{cloze:Text}}<br><br>{{Extra}}",
            }],
            model_type=genanki.Model.CLOZE,
        )

        # Build decks keyed by deck name
        deck_map: Dict[str, genanki.Deck] = {}

        def _get_deck(name: str) -> genanki.Deck:
            if name not in deck_map:
                deck_id = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**31)
                deck_map[name] = genanki.Deck(deck_id, name)
            return deck_map[name]

        for card in cards:
            tag_list = card.sanitized_tags
            deck = _get_deck(card.deck)

            if card.card_type == "cloze":
                note = genanki.Note(
                    model=cloze_model,
                    fields=[card.front, card.back],
                    tags=tag_list,
                )
            else:
                note = genanki.Note(
                    model=basic_model,
                    fields=[card.front, card.back],
                    tags=tag_list,
                )
            deck.add_note(note)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        package = genanki.Package(list(deck_map.values()))
        package.write_to_file(str(output_path))
        logger.info("Wrote %d card(s) to %s (apkg)", len(cards), output_path)
        return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_domain(text: str, tags: Optional[List[str]] = None) -> str:
    """Detect the primary knowledge domain for a piece of text.

    Checks tags first (looking for known domain names), then falls back to
    keyword scan of the text content.  Returns "general" if no match.
    """
    text_lower = text.lower()

    # Tags may contain strings like "ai-brain/medicine" or "domain::medicine"
    if tags:
        for tag in tags:
            for domain in _DOMAIN_KEYWORDS:
                if domain in tag.lower():
                    return domain

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return domain

    return "general"


def _domain_to_deck(domain: str) -> str:
    """Map a domain name to an Anki sub-deck string."""
    suffix = DOMAIN_DECK_MAP.get(domain, "General")
    return f"{DECK_PREFIX}::{suffix}"


def _try_cloze_deletion(text: str) -> Optional[str]:
    """Attempt to generate a cloze deletion from text.

    Targets definition-style patterns:
    - "X is Y"  → "{{c1::X}} is Y"
    - "decided to X" → "decided to {{c1::X}}"
    - "use X for Y" → "use {{c1::X}} for Y"

    Returns the cloze string, or None if no suitable pattern is found.
    """
    # Pattern: "X is a/an ..." — blank the subject
    m = re.match(
        r'^([A-Z][a-zA-Z0-9\s\-]{2,40}?)\s+(is|are)\s+(.{10,})',
        text.strip(),
    )
    if m:
        subject, verb, rest = m.group(1), m.group(2), m.group(3)
        return f"{{{{c1::{subject}}}}} {verb} {rest}"

    # Pattern: "decided to [action phrase]"
    m = re.search(r'\b(decided to|going with|switched to|use|using)\s+([A-Za-z][a-zA-Z0-9\s\-]{1,40})', text)
    if m:
        keyword = m.group(1)
        target = m.group(2).strip().rstrip(".,;")
        if len(target) > 2:
            return text.replace(f"{keyword} {target}", f"{keyword} {{{{c1::{target}}}}}", 1)

    return None


def _strip_cloze_markers(text: str) -> str:
    """Remove cloze markers, leaving only the revealed text."""
    return re.sub(r'\{\{c\d+::(.+?)\}\}', r'\1', text)


def _concept_front_back(concept: Concept) -> tuple[str, str]:
    """Generate (front, back) question-answer pair for a concept."""
    content = concept.content.strip()
    context = (concept.context or "").strip()

    if concept.concept_type == ConceptType.DECISION:
        # Truncate content for the question
        topic = content[:60].rstrip(".,;")
        front = f"What was decided: {topic}?"
        back = content if len(content) > 60 else (context or content)

    elif concept.concept_type == ConceptType.ACTION_ITEM:
        topic = content[:60].rstrip(".,;")
        front = f"What action was required: {topic}?"
        back = content if len(content) > 60 else (context or content)

    elif concept.concept_type == ConceptType.QUESTION:
        # The concept IS the question
        front = content
        back = context or "(Review this question — no answer recorded)"

    elif concept.concept_type == ConceptType.INSIGHT:
        # First sentence as question prompt, full content as answer
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) > 1:
            front = sentences[0]
            back = content
        else:
            front = f"What is the insight about: {content[:60]}?"
            back = context or content

    else:  # TOPIC and anything else
        front = f"Explain: {content[:80]}"
        back = context or content

    return front, back


def _vault_front_back(
    title: str, note_type: str, body: str
) -> tuple[Optional[str], Optional[str]]:
    """Derive front/back card content from a vault note.

    Returns (None, None) if the note doesn't have enough content.
    """
    # Strip Obsidian Dataview inline fields and wikilinks for cleaner card text
    clean_body = re.sub(r'\[\[(.+?)\]\]', r'\1', body)
    clean_body = re.sub(r'\w+::\s*.+', '', clean_body)
    clean_body = clean_body.strip()

    # First non-empty, non-header paragraph
    paragraphs = [
        p.strip() for p in re.split(r'\n{2,}', clean_body)
        if p.strip() and not p.strip().startswith('#')
    ]
    answer = paragraphs[0] if paragraphs else ""

    if not answer:
        return None, None

    if note_type == "synthesis":
        # Synthesis notes have structured cross-domain explanations
        domains = re.findall(r'\*([^*]+)\*', answer)
        if len(domains) >= 2:
            front = f"What is the connection between {domains[0]} and {domains[1]}?"
        else:
            front = f"Explain the cross-domain insight: {title}"
        back = answer

    elif note_type in ("decision", "concept"):
        front = f"What was the decision regarding: {title}?"
        back = answer

    elif note_type == "entity":
        front = f"Who or what is: {title}?"
        back = answer

    elif note_type == "conversation":
        # Conversation notes are too broad for single cards
        return None, None

    else:
        front = f"Explain: {title}"
        back = answer

    return front, back


def _parse_frontmatter(raw: str) -> tuple[Optional[dict], str]:
    """Split a Markdown file into (frontmatter_dict, body_text).

    Returns (None, raw) if no frontmatter block is found.
    """
    if not raw.startswith("---"):
        return None, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return None, raw
    fm_text = raw[3:end].strip()
    body = raw[end + 4:].strip()
    try:
        fm = yaml.safe_load(fm_text)
        return (fm if isinstance(fm, dict) else None), body
    except yaml.YAMLError:
        return None, raw
