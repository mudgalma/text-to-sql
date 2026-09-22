"""Tests for deterministic Phase 8 confidence scoring."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.execution.executor import ExecutionResult
from engine.scoring.scorer import ConfidenceScorer
from engine.semantic_layer import DuckDBSemanticLayer
from engine.types import AnalyticalSpec, EntitySpan, TaggedQuery, Token


@pytest.fixture(scope="module")
def semantic_layer() -> DuckDBSemanticLayer:
    """Provide real schemas for SQL identifier validation."""

    layer = DuckDBSemanticLayer(
        sales_csv="dataset/sales_data.csv",
        targets_csv="dataset/targets.csv",
        dict_json="dataset/data_dictionary.json",
    )
    yield layer
    layer.close()


@pytest.fixture(scope="module")
def scorer(semantic_layer: DuckDBSemanticLayer) -> ConfidenceScorer:
    """Provide the deterministic scorer under test."""

    return ConfidenceScorer(semantic_layer)


def _tagged_query() -> TaggedQuery:
    """Return a minimally recognized query without ambiguity."""

    token = Token("sales", 0, 5, canonical="revenue", role="metric")
    span = EntitySpan((token,), canonical="revenue", role="metric")
    return TaggedQuery(raw_query="sales", tokens=[token], spans=[span])


def _execution(sql: str, dataframe: pd.DataFrame | None = None) -> ExecutionResult:
    """Return a successful execution result with a small non-empty default result."""

    return ExecutionResult(
        success=True,
        sql=sql,
        df=dataframe if dataframe is not None else pd.DataFrame({"value": [100.0]}),
    )


def test_perfect_case_scores_high(scorer: ConfidenceScorer) -> None:
    """Executed, valid, unambiguous, agreeing signals produce high confidence."""

    breakdown = scorer.score(
        tagged=_tagged_query(),
        spec=AnalyticalSpec(metrics=["revenue"]),
        execution=_execution("SELECT SUM(revenue) AS value FROM v_sales"),
        retries_used=0,
        rule_spec_present=True,
        llm_spec_present=True,
        specs_agree=True,
    )
    assert breakdown.final == 1.0


def test_failed_execution_is_capped_low(scorer: ConfidenceScorer) -> None:
    """No confidence score can claim reliability when no answer was produced."""

    breakdown = scorer.score(
        tagged=_tagged_query(),
        spec=AnalyticalSpec(metrics=["revenue"]),
        execution=ExecutionResult(False, "SELECT bad_column FROM v_sales", error="bad"),
        retries_used=0,
        rule_spec_present=True,
        llm_spec_present=True,
        specs_agree=True,
    )
    assert breakdown.execution_success == 0.0
    assert breakdown.final <= 0.25


def test_schema_violation_reduces_schema_score(scorer: ConfidenceScorer) -> None:
    """An unrecognized column is visible in the auditable schema component."""

    breakdown = scorer.score(
        tagged=_tagged_query(),
        spec=AnalyticalSpec(metrics=["revenue"]),
        execution=_execution("SELECT SUM(revenue) FROM v_sales WHERE fake_column = 1"),
        retries_used=0,
        rule_spec_present=True,
        llm_spec_present=True,
        specs_agree=True,
    )
    assert breakdown.schema_validity < 1.0
    assert breakdown.final < 1.0


def test_string_values_and_table_aliases_are_not_false_schema_errors(
    scorer: ConfidenceScorer,
) -> None:
    """Legitimate values and aliases do not reduce a schema score."""

    breakdown = scorer.score(
        tagged=_tagged_query(),
        spec=AnalyticalSpec(metrics=["revenue"]),
        execution=_execution(
            "SELECT s.region, SUM(s.revenue) AS value FROM v_sales s "
            "WHERE s.country = 'India' GROUP BY s.region"
        ),
        retries_used=0,
        rule_spec_present=True,
        llm_spec_present=True,
        specs_agree=True,
    )
    assert breakdown.schema_validity == 1.0


def test_retries_and_empty_results_reduce_plausibility(scorer: ConfidenceScorer) -> None:
    """Repair history and missing rows lower only the plausibility component."""

    no_retry = scorer.score(
        _tagged_query(),
        AnalyticalSpec(metrics=["revenue"]),
        _execution("SELECT SUM(revenue) FROM v_sales"),
        0,
        True,
        True,
        True,
    )
    empty_after_repair = scorer.score(
        _tagged_query(),
        AnalyticalSpec(metrics=["revenue"]),
        _execution("SELECT SUM(revenue) FROM v_sales", pd.DataFrame({"value": []})),
        2,
        True,
        True,
        True,
    )
    assert empty_after_repair.result_plausibility < no_retry.result_plausibility


def test_defaults_and_actual_disagreement_lower_confidence(scorer: ConfidenceScorer) -> None:
    """Defaults and contradictory parsing paths are independently visible signals."""

    spec = AnalyticalSpec(metrics=["revenue"], defaults_applied=["default_metric"])
    breakdown = scorer.score(
        _tagged_query(),
        spec,
        _execution("SELECT SUM(revenue) FROM v_sales"),
        0,
        True,
        True,
        False,
    )
    assert breakdown.defaults_score == 0.5
    assert breakdown.agreement_signal == 0.3


@pytest.mark.parametrize("retries_used", (-1, 0, 1, 2, 99))
def test_score_stays_in_the_unit_interval(
    scorer: ConfidenceScorer, retries_used: int
) -> None:
    """Unexpected retry counts cannot produce an invalid public score."""

    breakdown = scorer.score(
        _tagged_query(),
        AnalyticalSpec(metrics=["revenue"]),
        _execution("SELECT SUM(revenue) FROM v_sales"),
        retries_used,
        True,
        False,
    )
    assert 0.0 <= breakdown.final <= 1.0
