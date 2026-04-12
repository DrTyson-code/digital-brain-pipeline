"""Tests for cost tracking and budget enforcement."""

from __future__ import annotations

import pytest

from src.llm.cost import (
    PRICING,
    BudgetExceeded,
    CostEntry,
    CostReport,
    CostTracker,
    TokenBudget,
    compute_cost,
)
from src.llm.provider import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    model: str = "gpt-4o-mini",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cached: bool = False,
) -> LLMResponse:
    return LLMResponse(
        content="",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=compute_cost(model, input_tokens, output_tokens),
        latency_ms=10.0,
        cached=cached,
    )


# ---------------------------------------------------------------------------
# compute_cost
# ---------------------------------------------------------------------------


def test_compute_cost_known_model():
    # gpt-4o-mini: $0.15/M input, $0.60/M output
    cost = compute_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 0.75) < 1e-9


def test_compute_cost_claude_sonnet():
    # claude-sonnet-4-20250514: $3/M input, $15/M output
    cost = compute_cost("claude-sonnet-4-20250514", input_tokens=1_000_000, output_tokens=0)
    assert abs(cost - 3.00) < 1e-9


def test_compute_cost_zero_tokens():
    cost = compute_cost("gpt-4o-mini", input_tokens=0, output_tokens=0)
    assert cost == 0.0


def test_compute_cost_ollama_free():
    assert compute_cost("llama3.1:8b", 100_000, 50_000) == 0.0
    assert compute_cost("mistral:latest", 100_000, 50_000) == 0.0
    assert compute_cost("ollama/qwen2.5", 100_000, 50_000) == 0.0


def test_compute_cost_unknown_model_zero():
    # Unknown models return 0.0 rather than crashing
    assert compute_cost("totally-fake-model-v99", 1000, 1000) == 0.0


def test_compute_cost_scales_linearly():
    c1 = compute_cost("gpt-4o-mini", input_tokens=1000, output_tokens=0)
    c2 = compute_cost("gpt-4o-mini", input_tokens=2000, output_tokens=0)
    assert abs(c2 - 2 * c1) < 1e-12


def test_pricing_table_contains_expected_models():
    assert "claude-sonnet-4-20250514" in PRICING
    assert "gpt-4o-mini" in PRICING
    assert "gpt-4o" in PRICING


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


def test_token_budget_defaults():
    budget = TokenBudget()
    assert budget.max_cost_usd == 1.00
    assert budget.max_cost_per_conversation == 0.05
    assert budget.warn_at_pct == 0.75


def test_token_budget_custom():
    budget = TokenBudget(max_cost_usd=5.00, max_cost_per_conversation=0.10)
    assert budget.max_cost_usd == 5.00
    assert budget.max_cost_per_conversation == 0.10


def test_token_budget_validation_positive():
    with pytest.raises(Exception):
        TokenBudget(max_cost_usd=0)  # must be > 0


# ---------------------------------------------------------------------------
# CostTracker — basic recording
# ---------------------------------------------------------------------------


def test_tracker_starts_at_zero():
    tracker = CostTracker()
    assert tracker.total_cost == 0.0
    assert tracker.calls == []


def test_tracker_records_call():
    tracker = CostTracker()
    response = _make_response(model="gpt-4o-mini", input_tokens=1000, output_tokens=500)
    tracker.record(response, stage="entity", conversation_id="conv-1")

    assert len(tracker.calls) == 1
    assert tracker.total_cost == pytest.approx(response.cost_usd)


def test_tracker_accumulates_multiple_calls():
    tracker = CostTracker()
    for i in range(5):
        r = _make_response(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.record(r, stage="entity", conversation_id=f"conv-{i}")

    expected = 5 * compute_cost("gpt-4o-mini", 100, 50)
    assert tracker.total_cost == pytest.approx(expected)
    assert len(tracker.calls) == 5


# ---------------------------------------------------------------------------
# CostTracker — can_afford / assert_can_afford
# ---------------------------------------------------------------------------


def test_can_afford_within_budget():
    budget = TokenBudget(max_cost_usd=1.00)
    tracker = CostTracker(budget)
    assert tracker.can_afford(0.99) is True


def test_can_afford_exactly_at_limit():
    budget = TokenBudget(max_cost_usd=1.00)
    tracker = CostTracker(budget)
    assert tracker.can_afford(1.00) is True


def test_cannot_afford_exceeds_budget():
    budget = TokenBudget(max_cost_usd=1.00)
    tracker = CostTracker(budget)
    assert tracker.can_afford(1.01) is False


def test_can_afford_accounts_for_existing_spend():
    budget = TokenBudget(max_cost_usd=1.00)
    tracker = CostTracker(budget)
    # Spend $0.80 first
    tracker.total_cost = 0.80
    assert tracker.can_afford(0.20) is True
    assert tracker.can_afford(0.21) is False


def test_assert_can_afford_raises_when_over_budget():
    budget = TokenBudget(max_cost_usd=0.10)
    tracker = CostTracker(budget)
    tracker.total_cost = 0.09
    with pytest.raises(BudgetExceeded) as exc_info:
        tracker.assert_can_afford(0.02)
    assert exc_info.value.needed == pytest.approx(0.02)
    assert exc_info.value.remaining == pytest.approx(0.01)


def test_assert_can_afford_passes_within_budget():
    budget = TokenBudget(max_cost_usd=1.00)
    tracker = CostTracker(budget)
    tracker.assert_can_afford(0.50)  # Should not raise


def test_remaining_decreases_with_spend():
    budget = TokenBudget(max_cost_usd=1.00)
    tracker = CostTracker(budget)
    assert tracker.remaining == pytest.approx(1.00)
    tracker.total_cost = 0.30
    assert tracker.remaining == pytest.approx(0.70)


def test_remaining_never_negative():
    budget = TokenBudget(max_cost_usd=1.00)
    tracker = CostTracker(budget)
    tracker.total_cost = 2.00  # over budget
    assert tracker.remaining == 0.0


# ---------------------------------------------------------------------------
# CostTracker — per-conversation limit
# ---------------------------------------------------------------------------


def test_can_afford_conversation_within_limit():
    budget = TokenBudget(max_cost_per_conversation=0.05)
    tracker = CostTracker(budget)
    assert tracker.can_afford_conversation(0.04) is True
    assert tracker.can_afford_conversation(0.05) is True
    assert tracker.can_afford_conversation(0.06) is False


# ---------------------------------------------------------------------------
# CostTracker — report
# ---------------------------------------------------------------------------


def test_report_aggregates_cost_by_stage():
    tracker = CostTracker()
    tracker.record(_make_response(model="gpt-4o-mini", input_tokens=1000, output_tokens=500),
                   stage="entity", conversation_id="c1")
    tracker.record(_make_response(model="gpt-4o-mini", input_tokens=1000, output_tokens=500),
                   stage="entity", conversation_id="c2")
    tracker.record(_make_response(model="gpt-4o-mini", input_tokens=1000, output_tokens=500),
                   stage="relationship", conversation_id="c1")

    report = tracker.report()
    assert report.total_calls == 3
    assert set(report.cost_by_stage.keys()) == {"entity", "relationship"}
    assert report.cost_by_stage["entity"] == pytest.approx(
        2 * compute_cost("gpt-4o-mini", 1000, 500)
    )


def test_report_aggregates_cost_by_conversation():
    tracker = CostTracker()
    tracker.record(_make_response(model="gpt-4o-mini", input_tokens=500, output_tokens=200),
                   stage="entity", conversation_id="conv-A")
    tracker.record(_make_response(model="gpt-4o-mini", input_tokens=500, output_tokens=200),
                   stage="relationship", conversation_id="conv-A")
    tracker.record(_make_response(model="gpt-4o-mini", input_tokens=500, output_tokens=200),
                   stage="entity", conversation_id="conv-B")

    report = tracker.report()
    assert report.cost_by_conversation["conv-A"] == pytest.approx(
        2 * compute_cost("gpt-4o-mini", 500, 200)
    )
    assert report.cost_by_conversation["conv-B"] == pytest.approx(
        compute_cost("gpt-4o-mini", 500, 200)
    )


def test_report_cache_hit_rate():
    tracker = CostTracker()
    tracker.record(_make_response(cached=False), stage="entity", conversation_id="c1")
    tracker.record(_make_response(cached=True), stage="entity", conversation_id="c2")
    tracker.record(_make_response(cached=True), stage="entity", conversation_id="c3")

    report = tracker.report()
    assert report.cached_calls == 2
    assert report.total_calls == 3
    assert report.cache_hit_rate == pytest.approx(2 / 3)


def test_report_token_totals():
    tracker = CostTracker()
    tracker.record(
        _make_response(model="gpt-4o-mini", input_tokens=300, output_tokens=100),
        stage="entity", conversation_id="c1",
    )
    tracker.record(
        _make_response(model="gpt-4o-mini", input_tokens=200, output_tokens=50),
        stage="entity", conversation_id="c2",
    )
    report = tracker.report()
    assert report.total_input_tokens == 500
    assert report.total_output_tokens == 150


def test_report_empty_tracker():
    tracker = CostTracker()
    report = tracker.report()
    assert report.total_cost_usd == 0.0
    assert report.total_calls == 0
    assert report.cache_hit_rate == 0.0


def test_report_str_is_human_readable():
    tracker = CostTracker()
    tracker.record(_make_response(), stage="entity", conversation_id="c1")
    report = tracker.report()
    text = str(report)
    assert "Cost report" in text
    assert "entity" in text


# ---------------------------------------------------------------------------
# BudgetExceeded exception
# ---------------------------------------------------------------------------


def test_budget_exceeded_message():
    exc = BudgetExceeded(needed=0.05, remaining=0.02)
    assert "0.0500" in str(exc)
    assert "0.0200" in str(exc)
    assert exc.needed == pytest.approx(0.05)
    assert exc.remaining == pytest.approx(0.02)
