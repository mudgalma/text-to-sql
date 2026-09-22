"""Tests for the deterministic explanation contract required by the assignment."""
from __future__ import annotations

from engine.phase9.explainer import Explainer
from engine.types import AnalyticalSpec, EntitySpan, OrderSpec, TaggedQuery, Token


def _tagged(conflicts: int = 0) -> TaggedQuery:
    """Return a small tagged query, optionally with no semantic ambiguity."""

    token = Token("profit", 0, 6, canonical="profit", role="metric")
    return TaggedQuery("profit", [token], [EntitySpan((token,), "profit", "metric")])


def test_explanation_covers_understanding_generation_and_confidence() -> None:
    """The output names the requested analytical operation and execution facts."""

    explanation = Explainer().explain(
        tagged=_tagged(),
        spec=AnalyticalSpec(
            operation="rank",
            metrics=["profit"],
            group_by=["city"],
            order_by=OrderSpec("profit"),
            limit=2,
        ),
        confidence=0.92,
        retries_used=0,
        rule_spec_present=True,
        llm_spec_present=False,
        specs_agree=None,
        template_path_used=True,
        result_count=2,
    )
    for fact in ("ranking", "profit", "city", "limited to 2", "no repairs", "0.92"):
        assert fact in explanation


def test_explanation_reports_defaults_and_feedback() -> None:
    """Silent defaults and actual feedback reuse are transparent to the user."""

    explanation = Explainer().explain(
        tagged=_tagged(),
        spec=AnalyticalSpec(metrics=["revenue"], defaults_applied=["default_metric"]),
        confidence=0.7,
        retries_used=1,
        rule_spec_present=True,
        llm_spec_present=True,
        specs_agree=False,
        template_path_used=False,
        result_count=1,
        feedback_applied=True,
    )
    assert "default metric" in explanation
    assert "1 repair attempt" in explanation
    assert "verified feedback correction" in explanation
    assert "available rule and LLM specifications" in explanation


class MockLLM:
    """Small offline explanation client used to exercise the LLM-primary seam."""

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.last_call: dict[str, str] | None = None

    def generate(self, *, system: str, user: str) -> str:
        """Record request context and return configured fake prose."""

        self.last_call = {"system": system, "user": user}
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_llm_explanation_is_primary_when_it_meets_the_contract() -> None:
    """A concise two-sentence LLM response replaces the deterministic fallback."""

    client = MockLLM(
        "You asked for the two cities with the highest profit. "
        "I ranked city totals and returned two rows with high confidence."
    )
    explanation = Explainer(client).explain(
        _tagged(),
        AnalyticalSpec(operation="rank", metrics=["profit"], group_by=["city"], limit=2),
        0.92,
        0,
        True,
        False,
        None,
        True,
        2,
    )
    assert explanation.startswith("You asked")
    assert client.last_call is not None
    assert "structured_summary" in client.last_call["user"]
    assert "profit" in client.last_call["user"]


def test_invalid_or_failed_llm_explanation_uses_factual_fallback() -> None:
    """Malformed and unavailable LLM responses cannot prevent an explanation."""

    for response in ("One sentence only.", TimeoutError("timed out")):
        explanation = Explainer(MockLLM(response)).explain(
            _tagged(),
            AnalyticalSpec(metrics=["profit"]),
            0.8,
            0,
            True,
            False,
            None,
            True,
            1,
        )
        assert explanation.startswith("I understood")
