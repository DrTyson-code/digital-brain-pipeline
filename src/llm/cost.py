"""Token budget enforcement and cost tracking for LLM calls.

Usage::

    budget = TokenBudget(max_cost_usd=1.00, max_cost_per_conversation=0.05)
    tracker = CostTracker(budget)

    est = compute_cost("claude-sonnet-4-20250514", input_tokens=500, output_tokens=200)
    if not tracker.can_afford(est):
        raise BudgetExceeded(...)

    # ... make the API call ...

    tracker.record(response, stage="entity", conversation_id="abc123")

    report = tracker.report()
    print(f"Total: ${report.total_cost_usd:.4f}")

The tracker distinguishes **actual cost** (uncached calls — actual cash spent
on the API) from **modeled cost** (all calls including cache hits — what the
calls would have cost without caching). The `total_cost_usd` in the report
is actual cash; `modeled_cost_usd` is the modeled total. Two-pass design per
audit #2 (2026-04-30): the previous tracker counted cached responses as if
they were cash, so a 100% cache-hit run reported $2+ when actual spend was $0.

The soft budget (`max_cost_usd`) is the warn/skip threshold; the hard cap
(`hard_cap_usd`) is the halt threshold. They were one field; this commit
splits them so a $5 hard ceiling can sit above a $2 soft target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.llm.provider import LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 million tokens)
# ---------------------------------------------------------------------------

PRICING: Dict[str, Dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Ollama models are always free
}

_OLLAMA_FREE_SENTINEL = "ollama"


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a given model and token counts.

    Ollama models (either the literal string "ollama" or any string containing
    a colon, like "llama3.1:8b") are always $0.00.
    """
    if _OLLAMA_FREE_SENTINEL in model or ":" in model:
        return 0.0
    pricing = PRICING.get(model)
    if pricing is None:
        logger.warning("No pricing data for model %r; treating as $0.00", model)
        return 0.0
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Budget model
# ---------------------------------------------------------------------------

class TokenBudget(BaseModel):
    """Per-run cost constraints.

    Two thresholds: a soft `max_cost_usd` (warn/skip) and a hard `hard_cap_usd`
    (halt). Audit #2 (2026-04-30) found these were one field, conflating the
    skip-stage gate with the hard-halt gate.
    """

    max_cost_usd: float = Field(
        default=1.00,
        gt=0,
        description=(
            "Soft budget per pipeline run. When actual cash cost reaches "
            "warn_at_pct of this, log a warning. Stages may opt-in to skip "
            "themselves above this via can_afford_soft()."
        ),
    )
    hard_cap_usd: float = Field(
        default=5.00,
        gt=0,
        description=(
            "Hard halt ceiling. can_afford() / assert_can_afford() use this. "
            "BudgetExceeded raised only when actual cash cost would breach "
            "this. Must be >= max_cost_usd."
        ),
    )
    max_cost_per_conversation: float = Field(
        default=0.05,
        gt=0,
        description="Per-conversation spending limit",
    )
    warn_at_pct: float = Field(
        default=0.75,
        ge=0,
        le=1.0,
        description="Emit a warning when this fraction of max_cost_usd is consumed",
    )

    def model_post_init(self, _context) -> None:  # type: ignore[override]
        """Validate that hard_cap_usd is not below max_cost_usd."""
        if self.hard_cap_usd < self.max_cost_usd:
            raise ValueError(
                f"hard_cap_usd ({self.hard_cap_usd}) must be >= "
                f"max_cost_usd ({self.max_cost_usd})"
            )


class BudgetExceeded(RuntimeError):
    """Raised when a projected LLM call would exceed the hard-cap budget."""

    def __init__(self, needed: float, remaining: float) -> None:
        self.needed = needed
        self.remaining = remaining
        super().__init__(
            f"Budget exceeded (hard cap): need ${needed:.4f} but only ${remaining:.4f} remaining"
        )


# ---------------------------------------------------------------------------
# Per-call record
# ---------------------------------------------------------------------------

@dataclass
class CostEntry:
    """One recorded LLM call."""

    stage: str
    conversation_id: str
    cost_usd: float  # what this call would have cost (modeled, includes cached)
    input_tokens: int
    output_tokens: int
    model: str
    cached: bool
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

@dataclass
class CostReport:
    """Cost breakdown for a completed pipeline run.

    Distinguishes:
    - total_cost_usd: actual cash spent (uncached calls only)
    - modeled_cost_usd: what calls would have cost without caching (all calls)

    For a 100%-cache-hit run, total_cost_usd ≈ 0 and modeled_cost_usd > 0.
    """

    total_cost_usd: float  # actual cash (uncached only)
    modeled_cost_usd: float  # all calls (cached + uncached)
    total_input_tokens: int
    total_output_tokens: int
    total_calls: int
    cached_calls: int
    cost_by_stage: Dict[str, float]  # actual cash by stage
    cost_by_conversation: Dict[str, float]  # actual cash by conversation

    @property
    def cache_hit_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.cached_calls / self.total_calls

    @property
    def cache_savings_usd(self) -> float:
        """Modeled cost minus actual cash — the value of caching."""
        return max(0.0, self.modeled_cost_usd - self.total_cost_usd)

    def __str__(self) -> str:
        # Show both numbers when caching is active so the report is honest.
        if self.cached_calls > 0 and self.modeled_cost_usd > self.total_cost_usd:
            cost_line = (
                f"Cost report: ${self.total_cost_usd:.4f} actual cash "
                f"(${self.modeled_cost_usd:.4f} modeled, "
                f"${self.cache_savings_usd:.4f} saved by cache)"
            )
        else:
            cost_line = f"Cost report: ${self.total_cost_usd:.4f} total"

        lines = [
            cost_line,
            f"  Calls: {self.total_calls} ({self.cached_calls} cached, "
            f"{self.cache_hit_rate:.0%} hit rate)",
            f"  Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out",
        ]
        if self.cost_by_stage:
            lines.append("  By stage (actual cash):")
            for stage, cost in sorted(self.cost_by_stage.items()):
                lines.append(f"    {stage}: ${cost:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Tracks LLM spending across a single pipeline run.

    Tracks two parallel totals:
    - actual_cost: uncached calls only — what's actually charged
    - modeled_cost: all calls — what the run would have cost without caching

    Budget gates use actual_cost (cash). The reported total is also
    actual_cost. Modeled cost is exposed for transparency about cache value.

    Thread-safety: not thread-safe; designed for sequential pipeline use.
    """

    def __init__(self, budget: Optional[TokenBudget] = None) -> None:
        self.budget = budget or TokenBudget()
        self.actual_cost: float = 0.0  # uncached only — actual cash spent
        self.modeled_cost: float = 0.0  # all calls — what they'd cost without cache
        self.calls: List[CostEntry] = []
        self._warned_soft = False

    @property
    def total_cost(self) -> float:
        """Backward-compat alias: actual cash spent (uncached only).

        Pre-audit-#2 tracker used `total_cost` for what is now `modeled_cost`.
        Code reading this attribute now sees real cash. The audit-#2 fix
        narrowed the semantics; this preserves the public-name surface.
        """
        return self.actual_cost

    # ------------------------------------------------------------------
    # Budget checks
    # ------------------------------------------------------------------

    def can_afford(self, estimated_cost: float) -> bool:
        """Return True if adding ``estimated_cost`` would stay under the HARD cap.

        This is the hard halt gate. Use can_afford_soft() to check the soft
        budget separately.
        """
        return (self.actual_cost + estimated_cost) <= self.budget.hard_cap_usd

    def assert_can_afford(self, estimated_cost: float) -> None:
        """Raise BudgetExceeded if the call would exceed the HARD cap."""
        if not self.can_afford(estimated_cost):
            remaining = self.budget.hard_cap_usd - self.actual_cost
            raise BudgetExceeded(needed=estimated_cost, remaining=remaining)

    def can_afford_soft(self, estimated_cost: float) -> bool:
        """Return True if adding ``estimated_cost`` would stay under the soft budget."""
        return (self.actual_cost + estimated_cost) <= self.budget.max_cost_usd

    def can_afford_conversation(self, estimated_cost: float) -> bool:
        """Return True if this call is within the per-conversation limit."""
        return estimated_cost <= self.budget.max_cost_per_conversation

    @property
    def remaining(self) -> float:
        """Remaining budget against the HARD cap."""
        return max(0.0, self.budget.hard_cap_usd - self.actual_cost)

    @property
    def remaining_soft(self) -> float:
        """Remaining budget against the soft target."""
        return max(0.0, self.budget.max_cost_usd - self.actual_cost)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        response: LLMResponse,
        stage: str,
        conversation_id: str,
    ) -> None:
        """Record a completed LLM call and update running totals.

        Cached responses contribute to `modeled_cost` (so caching value is
        visible) but NOT to `actual_cost` (since no cash was spent). Pre-audit-#2
        behavior incremented both counters identically, producing reports
        like "$2.27 total" for runs with 100% cache hit rate and $0 cash spend.
        """
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

        # Modeled total: what the call would have cost. Always increments.
        self.modeled_cost += response.cost_usd

        # Actual cash: only increment when the call hit the API.
        if not response.cached:
            self.actual_cost += response.cost_usd

        # Soft-budget warning, gated on actual cash (since the soft target is
        # the cash spending target, not the modeled-compute volume).
        if self.budget.max_cost_usd > 0:
            pct = self.actual_cost / self.budget.max_cost_usd
            if not self._warned_soft and pct >= self.budget.warn_at_pct:
                self._warned_soft = True
                logger.warning(
                    "LLM soft budget %.0f%% consumed ($%.4f / $%.4f); "
                    "hard cap is $%.4f",
                    pct * 100,
                    self.actual_cost,
                    self.budget.max_cost_usd,
                    self.budget.hard_cap_usd,
                )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> CostReport:
        """Build a CostReport from all recorded calls."""
        cost_by_stage: Dict[str, float] = {}
        cost_by_conversation: Dict[str, float] = {}
        total_in = total_out = cached = 0
        for entry in self.calls:
            # Per-stage and per-conversation counters track actual cash so
            # downstream observability matches the report's headline number.
            stage_cost = 0.0 if entry.cached else entry.cost_usd
            cost_by_stage[entry.stage] = cost_by_stage.get(entry.stage, 0.0) + stage_cost
            cost_by_conversation[entry.conversation_id] = (
                cost_by_conversation.get(entry.conversation_id, 0.0) + stage_cost
            )
            total_in += entry.input_tokens
            total_out += entry.output_tokens
            if entry.cached:
                cached += 1

        return CostReport(
            total_cost_usd=self.actual_cost,
            modeled_cost_usd=self.modeled_cost,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_calls=len(self.calls),
            cached_calls=cached,
            cost_by_stage=cost_by_stage,
            cost_by_conversation=cost_by_conversation,
        )
