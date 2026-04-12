# Digital Brain Pipeline — Phase 2 Technical Spec
## LLM-Powered Extraction Layer

**Version:** 0.1 draft  
**Date:** 2026-04-11  
**Status:** Implemented  
**Depends on:** Phase 1 (rule-based pipeline, 26 passing tests, Pydantic models stable)

---

## 0. Design Principles

1. **Augment, don't replace.** Rule-based extraction stays as the fast/free baseline. LLM extraction runs on top and merges results. Either layer can run alone.
2. **Pydantic is the contract.** Every LLM output must validate through the existing models (`Entity`, `Concept`, `Relationship`, `Conversation`). If the LLM hallucinates a field, Pydantic catches it.
3. **Cost is a first-class constraint.** A side project shouldn't cost $50/month in API calls. Token budgets, caching, and incremental processing are not optional.
4. **Provider-agnostic from day one.** Claude, GPT-4, local ollama — same interface, same prompts (with minor model-specific tuning).

---

## 1. Architecture Overview

### 1.1 Where LLM Calls Fit

The existing pipeline is: **ingest → classify → extract → link → enrich → output**.

Phase 2 adds an `LLMExtractor` that sits *beside* the rule-based `Extractor`, not *instead of* it. The orchestrator runs both and merges:

```
                    ┌─────────────────┐
                    │   Conversation   │
                    │   (from ingest)  │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                ▼                         ▼
    ┌───────────────────┐    ┌───────────────────┐
    │  RuleBasedExtractor│    │   LLMExtractor    │
    │  (existing Phase 1)│    │   (Phase 2 new)   │
    └─────────┬─────────┘    └─────────┬─────────┘
              │                        │
              └──────────┬─────────────┘
                         ▼
              ┌─────────────────────┐
              │   ExtractionMerger  │
              │  (dedup + reconcile)│
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │  Linker / Enricher  │
              │  (existing, enhanced│
              │   with LLM calls)   │
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │   Obsidian Output   │
              └─────────────────────┘
```

### 1.2 Extraction Modes (User-Configurable)

| Mode | Behavior | Cost |
|------|----------|------|
| `rules_only` | Phase 1 behavior, no LLM calls | Free |
| `llm_augmented` | Run both, merge results (default) | Low-medium |
| `llm_primary` | LLM runs first, rules fill gaps for failed/skipped convos | Medium |
| `llm_only` | Skip rule-based entirely | Medium-high |

Default is `llm_augmented` — safest, gives you the union of both extraction methods.

### 1.3 New Modules

```
src/
├── llm/
│   ├── __init__.py
│   ├── provider.py          # Abstract provider interface
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── claude.py         # Anthropic API
│   │   ├── openai.py         # OpenAI API
│   │   └── ollama.py         # Local models
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── entity_extraction.py
│   │   ├── relationship_mapping.py
│   │   ├── concept_classification.py
│   │   └── cross_linking.py
│   ├── extractor.py          # LLMExtractor class
│   ├── merger.py             # ExtractionMerger
│   ├── cache.py              # Response caching
│   └── cost.py               # Token tracking + budget enforcement
├── process/
│   ├── ... (existing)
│   └── orchestrator.py       # New: coordinates rule-based + LLM extraction
```

---

## 2. LLM Provider Abstraction

### 2.1 Provider Interface

```python
# src/llm/provider.py
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import TypeVar, Type

T = TypeVar("T", bound=BaseModel)

class LLMResponse(BaseModel):
    """Wrapper for any LLM response with metadata."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool = False

class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.0,
    ) -> tuple[T, LLMResponse]:
        """Return parsed Pydantic model + raw response metadata.
        
        Uses tool_use/function_calling for Claude/OpenAI,
        JSON mode + manual parse for ollama.
        """
        ...

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost before making a call."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        ...
```

### 2.2 Provider Implementations

**Claude** (`src/llm/providers/claude.py`):
- Use `anthropic` SDK with `tool_use` for structured extraction
- Define tools whose input schemas match the Pydantic models
- Model: `claude-sonnet-4-20250514` default (good price/quality), `claude-opus-4-20250514` for high-value passes
- Supports prompt caching — enable for system prompts

**OpenAI** (`src/llm/providers/openai.py`):
- Use `openai` SDK with `response_format={"type": "json_schema", ...}` (structured outputs)
- Model: `gpt-4o-mini` default, `gpt-4o` for high-value passes
- Supports batch API for non-urgent processing (50% cost reduction)

**Ollama** (`src/llm/providers/ollama.py`):
- HTTP calls to local `http://localhost:11434`
- JSON mode via `format: "json"` parameter
- Manual Pydantic validation (no native structured output)
- Models: `llama3.1:8b`, `mistral`, `qwen2.5` — user-configurable
- Fallback: if JSON parse fails, retry once with stricter prompt, then fall back to rule-based

### 2.3 Provider Factory

```python
# src/llm/provider.py (continued)

class ProviderConfig(BaseModel):
    provider: str  # "claude" | "openai" | "ollama"
    model: str | None = None  # Override default model
    api_key: str | None = None  # None = read from env
    base_url: str | None = None  # For ollama or proxies
    temperature: float = 0.0
    max_retries: int = 2
    timeout_seconds: int = 30

def create_provider(config: ProviderConfig) -> LLMProvider:
    match config.provider:
        case "claude":
            return ClaudeProvider(config)
        case "openai":
            return OpenAIProvider(config)
        case "ollama":
            return OllamaProvider(config)
        case _:
            raise ValueError(f"Unknown provider: {config.provider}")
```

---

## 3. Prompt Engineering

Each extraction stage gets a dedicated prompt module. Prompts are designed to return structured JSON that maps to existing Pydantic models.

### 3.1 Entity Extraction

**Goal:** Extract people, tools, libraries, organizations, concepts mentioned in a conversation.

```python
# src/llm/prompts/entity_extraction.py

SYSTEM_PROMPT = """You are an entity extraction system for a personal knowledge base.

Given an AI chat conversation, extract all meaningful entities mentioned.

Entity types (from ontology):
- PERSON: Named individuals (authors, researchers, colleagues)
- TOOL: Software tools, libraries, frameworks, APIs
- ORGANIZATION: Companies, institutions, teams
- CONCEPT: Technical concepts, methodologies, patterns
- RESOURCE: Books, papers, URLs, datasets
- PROJECT: Named projects or repos

Rules:
1. Extract the canonical name (e.g., "PyTorch" not "pytorch" or "torch")
2. Include aliases when the conversation uses multiple names for the same thing
3. Only extract entities actually discussed, not passing mentions in boilerplate
4. For each entity, provide a one-sentence description grounded in how it appears in THIS conversation
5. Assign a confidence score (0.0-1.0) based on how clearly the entity is identified

Return JSON matching the provided schema exactly."""

USER_PROMPT_TEMPLATE = """Extract entities from this {source} conversation.

Topic: {topic}
Messages: {message_count}

--- CONVERSATION ---
{conversation_text}
--- END ---

Extract all entities as a JSON array."""
```

**Response schema** (maps to existing `Entity` model):

```python
class LLMEntityExtraction(BaseModel):
    """What the LLM returns — validated, then mapped to Entity."""
    entities: list[ExtractedEntity]

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str  # Must match ontology types
    description: str
    aliases: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    source_quotes: list[str] = []  # Grounding evidence
```

### 3.2 Relationship Mapping

**Goal:** Identify how entities relate to each other within and across conversations.

```python
SYSTEM_PROMPT = """You are a relationship extraction system for a personal knowledge base.

Given a conversation and a list of entities already extracted from it, identify
relationships between entities.

Relationship types (from ontology):
- USES: entity uses/depends on another (e.g., "project USES pytorch")
- CREATED_BY: entity was created by another
- PART_OF: entity is part of another
- RELATED_TO: general topical relationship
- COMPARED_WITH: entities explicitly compared
- IMPLEMENTS: entity implements a concept/pattern
- EXTENDS: entity builds on another

Rules:
1. Only extract relationships with evidence in the conversation
2. Each relationship needs a direction (subject → predicate → object)
3. Assign confidence based on how explicitly the relationship is stated
4. Include a brief evidence quote from the conversation

Return JSON matching the provided schema exactly."""

USER_PROMPT_TEMPLATE = """Given these entities extracted from a {source} conversation:

{entities_json}

And this conversation:
--- CONVERSATION ---
{conversation_text}
--- END ---

Extract all relationships between these entities."""
```

**Response schema:**

```python
class ExtractedRelationship(BaseModel):
    subject: str      # Entity name (must match an extracted entity)
    predicate: str    # Relationship type from ontology
    object: str       # Entity name
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str     # Quote from conversation
    bidirectional: bool = False

class LLMRelationshipExtraction(BaseModel):
    relationships: list[ExtractedRelationship]
```

### 3.3 Concept Classification

**Goal:** Classify conversations into the ontology's concept hierarchy and tag with themes.

```python
SYSTEM_PROMPT = """You are a concept classification system for a personal knowledge base.

Given a conversation, classify it into the knowledge ontology and extract
high-level concepts discussed.

Top-level domains (from schema.yaml):
- software_engineering
- machine_learning
- data_science
- medicine (anesthesiology, pharmacology, physiology)
- productivity
- writing
- personal

Rules:
1. Assign primary and secondary domains
2. Extract specific concepts within those domains (e.g., "transformer architecture" within machine_learning)
3. Identify the conversation's PURPOSE: learning, problem_solving, brainstorming, debugging, research, planning, creative
4. Rate conversation DEPTH: surface, moderate, deep
5. Concepts should be named at the level of granularity useful for linking — not too broad ("programming"), not too narrow ("line 42 bug")

Return JSON matching the provided schema exactly."""
```

**Response schema:**

```python
class ConversationClassification(BaseModel):
    primary_domain: str
    secondary_domains: list[str] = []
    purpose: str
    depth: str  # surface | moderate | deep
    concepts: list[ExtractedConcept]
    summary: str  # 2-3 sentence summary for the Obsidian note

class ExtractedConcept(BaseModel):
    name: str
    domain: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
```

### 3.4 Cross-Linking and Deduplication

**Goal:** Given entities/concepts from *multiple* conversations, identify duplicates and cross-conversation links.

```python
SYSTEM_PROMPT = """You are a knowledge graph deduplication and linking system.

Given two sets of entities/concepts from different conversations, identify:
1. DUPLICATES: Same real-world entity with different names or slight variations
2. CROSS-LINKS: Entities from different conversations that are meaningfully related
3. MERGE CANDIDATES: Entities that should be consolidated into one note

Rules:
1. Be conservative with merges — only merge when clearly the same thing
2. Cross-links should be substantive, not just "both mention Python"
3. For each proposed link/merge, explain why
4. Consider aliases, abbreviations, and common variations

Return JSON matching the provided schema exactly."""

USER_PROMPT_TEMPLATE = """Conversation A entities:
{entities_a_json}

Conversation B entities:
{entities_b_json}

Identify duplicates, cross-links, and merge candidates."""
```

**Response schema:**

```python
class DeduplicationResult(BaseModel):
    duplicates: list[DuplicatePair]
    cross_links: list[CrossLink]
    merge_candidates: list[MergeCandidate]

class DuplicatePair(BaseModel):
    entity_a: str
    entity_b: str
    confidence: float = Field(ge=0.0, le=1.0)
    canonical_name: str  # Preferred name after merge
    reason: str

class CrossLink(BaseModel):
    entity_a: str
    entity_b: str
    relationship: str
    reason: str

class MergeCandidate(BaseModel):
    entities: list[str]
    canonical_name: str
    reason: str
```

---

## 4. Cost Management

### 4.1 Token Budget System

```python
# src/llm/cost.py

class TokenBudget(BaseModel):
    """Per-run cost constraints."""
    max_cost_usd: float = 1.00        # Hard ceiling per pipeline run
    max_cost_per_conversation: float = 0.05  # Per-conversation limit
    warn_at_pct: float = 0.75          # Warn at 75% budget consumed
    
class CostTracker:
    """Tracks spending across a pipeline run."""
    
    def __init__(self, budget: TokenBudget):
        self.budget = budget
        self.total_cost = 0.0
        self.calls: list[CostEntry] = []
    
    def can_afford(self, estimated_cost: float) -> bool:
        return (self.total_cost + estimated_cost) <= self.budget.max_cost_usd
    
    def record(self, response: LLMResponse, stage: str, conversation_id: str):
        entry = CostEntry(
            stage=stage,
            conversation_id=conversation_id,
            cost_usd=response.cost_usd,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            cached=response.cached,
        )
        self.calls.append(entry)
        self.total_cost += response.cost_usd
    
    def report(self) -> CostReport:
        """Generate a cost breakdown by stage and conversation."""
        ...
```

**Pricing table** (baked in, overridable in config):

```python
PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},  # per 1M tokens
    "claude-opus-4-20250514":   {"input": 15.00, "output": 75.00},
    "gpt-4o-mini":              {"input": 0.15, "output": 0.60},
    "gpt-4o":                   {"input": 2.50, "output": 10.00},
    "ollama/*":                 {"input": 0.00, "output": 0.00},
}
```

### 4.2 Cost Reduction Strategies

**Conversation truncation:** Most AI chat exports are long. Before sending to the LLM:
1. Strip system prompts and repeated boilerplate
2. For conversations over N tokens (configurable, default 8000), use a two-pass approach:
   - Pass 1: Send a compressed summary request to a cheap model (gpt-4o-mini / haiku)
   - Pass 2: Send the summary + key excerpts to the main model for extraction
3. Alternatively, chunk long conversations and extract per-chunk, then merge

**Caching** (see §4.3):
- Cache LLM responses keyed on `(conversation_hash, prompt_version, model)`
- If a conversation hasn't changed since last run, skip the LLM call entirely

**Batch processing:**
- Group conversations and send as batch requests where the API supports it (OpenAI Batch API)
- For ollama, use async concurrent requests (configurable parallelism)

### 4.3 Caching Strategy

```python
# src/llm/cache.py

class ExtractionCache:
    """SQLite-backed cache for LLM extraction results."""
    
    def __init__(self, cache_path: Path = Path(".dbp_cache/extractions.db")):
        self.db = sqlite3.connect(cache_path)
        self._init_tables()
    
    def get(
        self,
        conversation_hash: str,
        stage: str,  # "entity" | "relationship" | "classification" | "crosslink"
        prompt_version: str,
        model: str,
    ) -> CachedExtraction | None:
        """Return cached result if conversation unchanged and prompt version matches."""
        ...
    
    def put(
        self,
        conversation_hash: str,
        stage: str,
        prompt_version: str,
        model: str,
        result: BaseModel,  # The Pydantic extraction result
        response: LLMResponse,  # Raw response metadata
    ) -> None:
        ...
    
    def invalidate_conversation(self, conversation_hash: str) -> int:
        """Invalidate all cached results for a conversation. Returns count deleted."""
        ...
    
    def invalidate_stage(self, stage: str) -> int:
        """Invalidate all results for a stage (e.g., after prompt update)."""
        ...
```

**Cache key:** `sha256(conversation_text)` — if even one message changes, re-extract. Prompt versions are tracked so upgrading a prompt automatically invalidates stale cache entries.

**Cache location:** `.dbp_cache/` in project root, gitignored.

### 4.4 Incremental Processing

The pipeline already processes conversations from export files. Phase 2 adds a manifest that tracks what's been processed:

```python
class ProcessingManifest(BaseModel):
    """Tracks which conversations have been processed and with what config."""
    entries: dict[str, ManifestEntry]  # conversation_id -> entry

class ManifestEntry(BaseModel):
    conversation_hash: str
    last_processed: datetime
    extraction_mode: str
    model_used: str
    prompt_versions: dict[str, str]  # stage -> version
    cost_usd: float
```

On each run, the pipeline:
1. Loads the manifest
2. Hashes each conversation
3. Skips conversations where hash + prompt versions + model all match
4. Only sends changed/new conversations to the LLM

---

## 5. Data Flow and Validation

### 5.1 LLM Output → Pydantic Model Pipeline

```
LLM raw JSON string
    │
    ▼
Pydantic parse_raw() / model_validate_json()
    │
    ├─ Success → ExtractedEntity / ExtractedRelationship / etc.
    │
    └─ ValidationError → Retry with error feedback (1 retry)
                │
                ├─ Success on retry → continue
                └─ Fail again → log warning, fall back to rule-based result
```

### 5.2 Retry with Error Feedback

When the LLM returns invalid JSON or fails Pydantic validation:

```python
async def extract_with_retry(
    provider: LLMProvider,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    max_retries: int = 1,
) -> T | None:
    for attempt in range(max_retries + 1):
        try:
            result, response = await provider.extract_structured(
                system_prompt, user_prompt, response_model
            )
            return result
        except ValidationError as e:
            if attempt < max_retries:
                # Feed the error back to the LLM
                user_prompt = (
                    f"{user_prompt}\n\n"
                    f"Your previous response had validation errors:\n{e}\n"
                    f"Please fix these errors and return valid JSON."
                )
            else:
                logger.warning(f"LLM extraction failed after {max_retries + 1} attempts: {e}")
                return None
```

### 5.3 Merging Rule-Based + LLM Results

```python
# src/llm/merger.py

class ExtractionMerger:
    """Merges rule-based and LLM extractions into a single result set."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def merge_entities(
        self,
        rule_entities: list[Entity],
        llm_entities: list[ExtractedEntity],
    ) -> list[Entity]:
        """
        Strategy:
        1. Start with all rule-based entities (trusted baseline)
        2. For each LLM entity:
           a. If it fuzzy-matches a rule-based entity → enrich the existing one
              (add description, aliases, bump confidence)
           b. If it's new and confidence >= threshold → add it
           c. If confidence < threshold → skip
        3. Deduplicate by canonical name
        """
        ...
    
    def merge_relationships(
        self,
        rule_rels: list[Relationship],
        llm_rels: list[ExtractedRelationship],
        known_entities: list[Entity],  # Post-merge entity list
    ) -> list[Relationship]:
        """
        Only keep LLM relationships where both subject and object
        exist in the known entity list. This prevents hallucinated
        relationships referencing non-existent entities.
        """
        ...
```

**Fuzzy matching** for entity dedup uses normalized names: lowercase, strip punctuation, handle common aliases (e.g., "PyTorch" = "pytorch" = "torch"). For ambiguous cases (confidence between 0.4-0.6), flag for optional human review in the Obsidian note's frontmatter:

```yaml
---
needs_review: true
review_reason: "Ambiguous entity merge: 'torch' may refer to PyTorch or the Lua framework"
---
```

---

## 6. Configuration

### 6.1 Config Schema Extension

Add an `llm` section to `config/schema.yaml` (or a new `config/llm.yaml`):

```yaml
# config/llm.yaml

llm:
  # Which extraction mode to use
  extraction_mode: llm_augmented  # rules_only | llm_augmented | llm_primary | llm_only

  # Provider configuration
  provider:
    name: claude          # claude | openai | ollama
    model: null           # null = use provider default
    api_key_env: ANTHROPIC_API_KEY  # env var name (never put keys in config)
    base_url: null        # override for ollama or proxy
    temperature: 0.0
    max_retries: 2
    timeout_seconds: 30

  # Cost controls
  budget:
    max_cost_per_run_usd: 1.00
    max_cost_per_conversation_usd: 0.05
    warn_at_percent: 75

  # Quality thresholds
  quality:
    entity_confidence_threshold: 0.5
    relationship_confidence_threshold: 0.6
    merge_similarity_threshold: 0.85  # For fuzzy entity matching

  # Processing options
  processing:
    max_conversation_tokens: 8000    # Truncate/summarize above this
    batch_size: 10                   # Conversations per batch
    parallel_requests: 3             # Concurrent LLM calls (for ollama)
    enable_cache: true
    cache_path: .dbp_cache/

  # Prompt versioning (auto-tracked, but can pin versions)
  prompts:
    entity_extraction: v1
    relationship_mapping: v1
    concept_classification: v1
    cross_linking: v1
```

### 6.2 Environment Variables

```bash
# .env (gitignored)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
# Ollama needs no key — just a running server
```

### 6.3 CLI Extensions

Extend the existing Click CLI in `scripts/run_pipeline.py`:

```python
@cli.command()
@click.option("--mode", type=click.Choice(["rules_only", "llm_augmented", "llm_primary", "llm_only"]),
              default="llm_augmented")
@click.option("--provider", type=click.Choice(["claude", "openai", "ollama"]), default="claude")
@click.option("--model", default=None, help="Override model name")
@click.option("--budget", type=float, default=1.00, help="Max USD per run")
@click.option("--dry-run", is_flag=True, help="Estimate cost without making LLM calls")
@click.option("--force", is_flag=True, help="Ignore cache, re-extract everything")
@click.option("--show-cost", is_flag=True, help="Print cost report after run")
def extract(mode, provider, model, budget, dry_run, force, show_cost):
    """Run entity/concept extraction with optional LLM augmentation."""
    ...
```

**Dry run** estimates cost by counting tokens (via tiktoken or anthropic's token counter) without making API calls. Prints a table like:

```
Dry run estimate:
  Conversations: 47
  Cached (skip): 31
  To process:    16
  Estimated tokens: ~128,000 input / ~32,000 output
  Estimated cost:   $0.43 (claude-sonnet-4-20250514)
  Budget remaining: $0.57
```

---

## 7. Quality Evaluation

### 7.1 Evaluation Framework

The goal is measuring whether LLM extraction is actually *better* than rule-based, and by how much.

**Golden test set:** Manually annotate 10-15 diverse conversations with ground-truth entities, relationships, and concepts. Store in `tests/golden/`:

```
tests/golden/
├── conversations/
│   ├── coding_discussion.json
│   ├── ml_research.json
│   ├── medical_question.json
│   ├── brainstorm_session.json
│   └── ...
├── annotations/
│   ├── coding_discussion_entities.json
│   ├── coding_discussion_relationships.json
│   └── ...
└── eval_config.yaml
```

### 7.2 Metrics

```python
# src/eval/metrics.py

class ExtractionMetrics(BaseModel):
    """Per-conversation extraction quality metrics."""
    precision: float      # Of extracted entities, how many are correct
    recall: float         # Of ground-truth entities, how many were found
    f1: float
    entity_count_delta: int  # extracted - ground_truth (positive = over-extraction)
    
class RelationshipMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    hallucination_rate: float  # Relationships referencing non-existent entities

class QualityReport(BaseModel):
    """Comparison: rule-based vs LLM vs merged."""
    rule_based: ExtractionMetrics
    llm_only: ExtractionMetrics
    merged: ExtractionMetrics
    cost_usd: float
    cost_per_f1_point: float  # Efficiency metric
```

### 7.3 Evaluation CLI

```bash
# Run evaluation against golden set
python scripts/run_pipeline.py evaluate \
    --golden-dir tests/golden \
    --mode llm_augmented \
    --provider claude \
    --output eval_report.json
```

This runs the pipeline in all three modes (rules_only, llm_only, llm_augmented) against the golden set and produces a comparison report.

---

## 8. Testing Strategy

### 8.1 Test Layers

**Unit tests** (`tests/unit/llm/`):
- Provider interface compliance (each provider implements all methods)
- Prompt template rendering (variables substituted correctly)
- Pydantic model validation (valid and invalid LLM outputs)
- Cache operations (get/put/invalidate)
- Cost tracking arithmetic
- Merger logic (entity matching, relationship filtering)

**Integration tests** (`tests/integration/llm/`):
- Full extraction pipeline with mocked LLM responses
- Cache hit/miss behavior
- Budget enforcement (stop when budget exceeded)
- Incremental processing (skip unchanged conversations)

**LLM tests** (`tests/llm/`) — marked `@pytest.mark.llm`, skipped by default:
- Actual API calls against each provider with a tiny test conversation
- Validates that real LLM output passes Pydantic validation
- Run manually or in CI with `pytest -m llm` (costs real money)

### 8.2 Mocking LLM Calls

```python
# tests/conftest.py

@pytest.fixture
def mock_provider():
    """Returns a mock LLMProvider with canned responses."""
    provider = MockLLMProvider()
    
    # Register canned response for entity extraction
    provider.register_response(
        prompt_contains="Extract entities",
        response=LLMResponse(
            content=json.dumps({
                "entities": [
                    {"name": "PyTorch", "entity_type": "TOOL",
                     "description": "ML framework", "confidence": 0.95,
                     "aliases": ["torch"], "source_quotes": ["using PyTorch for..."]},
                ]
            }),
            model="mock-model",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100,
        )
    )
    return provider


class MockLLMProvider(LLMProvider):
    """Deterministic mock for testing."""
    
    def __init__(self):
        self._responses: list[tuple[str, LLMResponse]] = []
    
    def register_response(self, prompt_contains: str, response: LLMResponse):
        self._responses.append((prompt_contains, response))
    
    async def complete(self, system_prompt, user_prompt, **kwargs) -> LLMResponse:
        for pattern, response in self._responses:
            if pattern in user_prompt or pattern in system_prompt:
                return response
        raise ValueError(f"No mock response registered for prompt")
    
    async def extract_structured(self, system_prompt, user_prompt, response_model, **kwargs):
        response = await self.complete(system_prompt, user_prompt)
        parsed = response_model.model_validate_json(response.content)
        return parsed, response
    
    def estimate_cost(self, input_tokens, output_tokens):
        return 0.0
    
    @property
    def model_name(self):
        return "mock-model"
    
    @property
    def max_context_tokens(self):
        return 128000
```

### 8.3 Fixture Conversations

Store a set of small, representative test conversations in `tests/fixtures/conversations/` covering:
- Short coding help (3-5 messages, 1-2 entities)
- Long research discussion (20+ messages, many entities and relationships)
- Mixed-topic brainstorm (multiple domains)
- Medical/anesthesia question (domain-specific entities)
- Minimal/empty conversation (edge case)

These are used in both unit and integration tests to ensure consistent behavior.

---

## 9. Migration Path

### Phase 2a — Foundation (Week 1-2)

1. **Create `src/llm/` module structure** — provider interface, mock provider, cache
2. **Implement `CostTracker` and `TokenBudget`** — with tests
3. **Implement `ExtractionCache`** — SQLite-backed, with tests
4. **Add `ProviderConfig` to config schema** — extend `config/llm.yaml`
5. **Add `--mode rules_only` to CLI** — explicit opt-in, same behavior as today

All existing tests continue to pass. No LLM calls yet.

### Phase 2b — Entity Extraction (Week 3-4)

1. **Implement Claude provider** — `extract_structured` with tool_use
2. **Write entity extraction prompts** — with prompt versioning
3. **Build `LLMExtractor.extract_entities()`** — single-conversation extraction
4. **Build `ExtractionMerger.merge_entities()`** — combine rule-based + LLM
5. **Create golden test set** — annotate 5 conversations for entities
6. **Add `--mode llm_augmented` to CLI** — entity extraction only at first

Deliverable: can run `python scripts/run_pipeline.py extract --mode llm_augmented` and get LLM-enriched entities in Obsidian notes.

### Phase 2c — Full Extraction (Week 5-6)

1. **Add relationship mapping prompts + extraction**
2. **Add concept classification prompts + extraction**
3. **Implement `Orchestrator`** — coordinates all stages, respects budget
4. **Add incremental processing** — manifest-based skip logic
5. **Expand golden test set** — add relationships and concepts
6. **Run evaluation** — compare rule-based vs augmented quality

### Phase 2d — Cross-Linking + Polish (Week 7-8)

1. **Implement cross-linking/dedup** — across conversations
2. **Add OpenAI provider** — with batch API support
3. **Add ollama provider** — for local/free usage
4. **Implement dry-run mode** — cost estimation
5. **Write cost report output** — prints after each run
6. **Full evaluation pass** — all modes, all providers, quality report

### What Doesn't Change

- `src/models/` — Pydantic models stay exactly as-is. LLM outputs validate through them.
- `src/ingest/` — Parsers are untouched. LLM layer consumes their output.
- `src/output/obsidian.py` — Output generation stays the same. It just gets richer input data.
- `tests/` — All 26 existing tests continue to pass at every step.
- `scripts/run_pipeline.py` — Gets new options, but default behavior (`rules_only` equivalent) is preserved.

---

## 10. Cost Estimates for Typical Usage

Rough numbers for a personal knowledge base with ~50-100 conversations per batch:

| Provider | Model | 50 convos | 100 convos | Notes |
|----------|-------|-----------|------------|-------|
| Claude | sonnet | ~$0.30 | ~$0.60 | With prompt caching |
| OpenAI | gpt-4o-mini | ~$0.05 | ~$0.10 | Cheapest cloud option |
| OpenAI | gpt-4o | ~$0.40 | ~$0.80 | Batch API = half price |
| Ollama | llama3.1:8b | $0.00 | $0.00 | Slower, lower quality |

After initial processing, incremental runs (5-10 new conversations) cost pennies.

---

## 11. Open Questions

1. **Ontology evolution:** Should the LLM be able to *propose new entity types* not in the current ontology, or strictly classify into existing types? Recommendation: start strict, add a "suggest_new_type" field for review.

2. **Human-in-the-loop:** Should the pipeline generate a "review queue" in Obsidian for low-confidence extractions? Could be a special note with a dataview query.

3. **Conversation chunking strategy:** For very long conversations (50+ messages), what's the best chunking approach? Sliding window? Topic-based splits? Start with simple truncation, iterate based on quality metrics.

4. **Multi-model routing:** Route cheap/simple conversations to gpt-4o-mini and complex ones to Claude Sonnet? Could save 30-50% on costs. Add in Phase 2d if budget matters.

5. **Embedding-based search:** Phase 3 possibility — generate embeddings for each conversation/entity for semantic search across the vault. Not in scope here, but the provider interface should be extensible to support `embed()` later.

---

## Appendix A: Dependency Additions

```
# requirements.txt additions for Phase 2
anthropic>=0.40.0
openai>=1.50.0
httpx>=0.27.0          # For ollama HTTP calls
tiktoken>=0.7.0        # Token counting for cost estimation
rapidfuzz>=3.0.0       # Fuzzy string matching for entity dedup
```

## Appendix B: Ontology Mapping

How Phase 2 extraction maps to the Palantir-inspired object model:

| Ontology Concept | Phase 1 (rule-based) | Phase 2 (LLM) |
|-----------------|---------------------|----------------|
| **Object** (Entity) | Regex + keyword matching | Contextual extraction with confidence scores |
| **Property** (attributes) | Hardcoded field mapping | LLM-generated descriptions, aliases |
| **Link** (Relationship) | Co-occurrence heuristics | Typed, directional relationships with evidence |
| **Type** (Classification) | Keyword-based domain tagging | Hierarchical classification with depth assessment |
| **Cross-reference** | Exact string match | Fuzzy matching + semantic deduplication |

The fundamental objects/properties/links structure stays the same. Phase 2 just fills them with higher-quality data.
