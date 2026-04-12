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
    """Per-run cost constraints."""

    max_cost_usd: float = Field(default=1.00, gt=0, description="Hard ceiling per pipeline run")
    max_cost_per_conversation: float = Field(
        default=0.05, gt=0, description="Per-conversation spending limit"
    )
    warn_at_pct: float = Field(
        default=0.75, ge=0, le=1.0,
        description="Emit a warning when this fraction of the budget is consumed",
    )


class BudgetExceeded(RuntimeError):
    """Raised when a projected LLM call would exceed the budget."""

    def __init__(self, needed: float, remaining: float) -> None:
        self.needed = needed
        self.remaining = remaining
        super().__init__(
            f"Budget exceeded: need ${needed:.4f} but only ${remaining:.4f} remaining"
        )


# ---------------------------------------------------------------------------
# Per-call record
# ---------------------------------------------------------------------------


@dataclass
class CostEntry:
    """One recorded LLM call."""

    stage: str
    conversation_id: str
    cost_usd: float
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
    """Cost breakdown for a completed pipeline run."""

    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_calls: int
    cached_calls: int
    cost_by_stage: Dict[str, float]
    cost_by_conversation: Dict[str, float]

    @property
    def cache_hit_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.cached_calls / self.total_calls

    def __str__(self) -> str:
        lines = [
            f"Cost report: ${self.total_cost_usd:.4f} total",
            f"  Calls: {self.total_calls} ({self.cached_calls} cached, "
            f"{self.cache_hit_rate:.0%} hit rate)",
            f"  Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out",
        ]
        if self.cost_by_stage:
            lines.append("  By stage:")
            for stage, cost in sorted(self.cost_by_stage.items()):
                lines.append(f"    {stage}: ${cost:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Tracks LLM spending across a single pipeline run.

    Thread-safety: not thread-safe; designed for sequential pipeline use.
    """

    def __init__(self, budget: Optional[TokenBudget] = None) -> None:
        self.budget = budget or TokenBudget()
        self.total_cost: float = 0.0
        self.calls: List[CostEntry] = []
        self._warned = False

    # ------------------------------------------------------------------
    # Budget checks
    # ------------------------------------------------------------------

    def can_afford(self, estimated_cost: float) -> bool:
        """Return True if adding ``estimated_cost`` would stay within budget."""
        return (self.total_cost + estimated_cost) <= self.budget.max_cost_usd

    def assert_can_afford(self, estimated_cost: float) -> None:
        """Raise BudgetExceeded if the call would exceed the budget."""
        if not self.can_afford(estimated_cost):
            remaining = self.budget.max_cost_usd - self.total_cost
            raise BudgetExceeded(needed=estimated_cost, remaining=remaining)

    def can_afford_conversation(self, estimated_cost: float) -> bool:
        """Return True if this call is within the per-conversation limit."""
        return estimated_cost <= self.budget.max_cost_per_conversation

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget.max_cost_usd - self.total_cost)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        response: LLMResponse,
        stage: str,
        conversation_id: str,
    ) -> None:
        """Record a completed LLM call and update the running total."""
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

        pct = self.total_cost / self.budget.max_cost_usd
        if not self._warned and pct >= self.budget.warn_at_pct:
            self._warned = True
            logger.warning(
                "LLM budget %.0f%% consumed ($%.4f / $%.4f)",
                pct * 100,
                self.total_cost,
                self.budget.max_cost_usd,
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
            cost_by_stage[entry.stage] = cost_by_stage.get(entry.stage, 0.0) + entry.cost_usd
            cost_by_conversation[entry.conversation_id] = (
                cost_by_conversation.get(entry.conversation_id, 0.0) + entry.cost_usd
            )
            total_in += entry.input_tokens
            total_out += entry.output_tokens
            if entry.cached:
                cached += 1

        return CostReport(
            total_cost_usd=self.total_cost,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_calls=len(self.calls),
            cached_calls=cached,
            cost_by_stage=cost_by_stage,
            cost_by_conversation=cost_by_conversation,
        )
