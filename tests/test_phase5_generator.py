"""Integration and safety tests for Phase 5 SQL generation."""
from __future__ import annotations

import inspect

import pytest

from engine.phase5.generator import SQLGenerationError, SQLGenerator
from engine.semantic_layer import DuckDBSemanticLayer
from engine.types import AnalyticalSpec, Filter, MetricFilter, OrderSpec, TimeConstraint
from tests.golden_specs import GOLDEN_SPECS


@pytest.fixture(scope="module")
def semantic_layer() -> DuckDBSemanticLayer:
    """Provide the real semantic layer used by generated-query tests."""

    layer = DuckDBSemanticLayer(
        sales_csv="dataset/sales_data.csv",
        targets_csv="dataset/targets.csv",
        dict_json="dataset/data_dictionary.json",
    )
    yield layer
    layer.close()


@pytest.fixture(scope="module")
def generator(semantic_layer: DuckDBSemanticLayer) -> SQLGenerator:
    """Provide a template-only SQL generator."""

    return SQLGenerator(semantic_layer)


@pytest.mark.parametrize("golden", GOLDEN_SPECS, ids=lambda item: item.query)
def test_all_golden_specs_produce_executable_sql(
    generator: SQLGenerator, semantic_layer: DuckDBSemanticLayer, golden
) -> None:
    """All assignment specifications produce one SQL query that DuckDB can run."""

    sql = generator.generate(golden.expected)
    assert sql.upper().lstrip().startswith(("SELECT", "WITH"))
    result = semantic_layer.execute(sql)
    assert result is not None


def test_templates_do_not_embed_dataset_vocabulary() -> None:
    """Keep templates reusable; semantic names are supplied only at render time."""

    from engine.phase5 import templates

    source = inspect.getsource(templates)
    forbidden = (
        "'profit'",
        '"profit"',
        "'city'",
        '"city"',
        "'revenue'",
        '"revenue"',
        "'region'",
        '"region"',
        "'target_revenue'",
        '"target_revenue"',
        "'avg_order_value'",
        '"avg_order_value"',
        "'customer_id'",
        '"customer_id"',
        "'product_name'",
        '"product_name"',
    )
    for literal in forbidden:
        assert literal not in source


def test_rank_spec_renders_grouping_limit_and_order(generator: SQLGenerator) -> None:
    """A global ranking renders its standard query components."""

    spec = AnalyticalSpec(
        operation="rank",
        metrics=["profit"],
        group_by=["city"],
        order_by=OrderSpec(metric="profit", direction="DESC"),
        limit=2,
    )
    sql = generator.generate(spec)
    assert "SUM(profit)" in sql
    assert "GROUP BY city" in sql
    assert "ORDER BY value DESC" in sql
    assert "LIMIT 2" in sql


def test_compiled_aggregate_metric_is_used(generator: SQLGenerator) -> None:
    """Derived metrics use the semantic layer's executable definition."""

    sql = generator.generate(
        AnalyticalSpec(metrics=["avg_order_value"], group_by=["region"])
    )
    assert "SUM((quantity * unit_price * (1 - discount)))" in sql
    assert "count(order_id)" in sql


def test_compare_spec_qualifies_join_filter_and_having(generator: SQLGenerator) -> None:
    """A target comparison joins on its temporal key without ambiguity."""

    spec = AnalyticalSpec(
        operation="compare",
        metrics=["revenue"],
        group_by=["region"],
        metric_filters=[
            MetricFilter(metric="revenue", operator="<", target_metric="target_revenue")
        ],
        time_window=TimeConstraint(
            raw="feb",
            column="month",
            resolution="EXACT_PERIOD",
            resolved_value="2024-02",
        ),
    )
    sql = generator.generate(spec)
    assert "JOIN targets t" in sql
    assert "s.month = t.month" in sql
    assert "WHERE s.month = '2024-02'" in sql
    assert "HAVING SUM(s.revenue) < CAST(t.target_revenue AS DOUBLE)" in sql


def test_contribution_spec_uses_a_window_sum(generator: SQLGenerator) -> None:
    """Contribution percentages use a window total after grouping."""

    sql = generator.generate(
        AnalyticalSpec(
            metrics=["revenue"],
            group_by=["product_category"],
            transforms=["contribution_pct"],
        )
    )
    assert "SUM(SUM(revenue)) OVER ()" in sql
    assert "AS pct" in sql


def test_trend_spec_uses_lag(generator: SQLGenerator) -> None:
    """Year-over-year trend specifications render a lag window."""

    sql = generator.generate(
        AnalyticalSpec(
            operation="trend",
            metrics=["revenue"],
            group_by=["year"],
            transforms=["yoy"],
        )
    )
    assert "LAG(val) OVER (ORDER BY year)" in sql
    assert "yoy_growth" in sql


def test_partitioned_rank_uses_requested_direction(generator: SQLGenerator) -> None:
    """Top-N-within-group queries use a dense ranking window."""

    sql = generator.generate(
        AnalyticalSpec(
            operation="rank",
            metrics=["revenue"],
            group_by=["region", "product_name"],
            partition_by=["region"],
            order_by=OrderSpec(metric="revenue", direction="ASC"),
            limit=1,
        )
    )
    assert "DENSE_RANK()" in sql
    assert "PARTITION BY region" in sql
    assert "ORDER BY SUM(revenue) ASC" in sql
    assert "rnk <= 1" in sql


def test_unknown_operation_without_fallback_fails(generator: SQLGenerator) -> None:
    """Unsupported shapes have a clear error when no fallback is configured."""

    with pytest.raises(SQLGenerationError, match="No deterministic SQL template"):
        generator.generate(AnalyticalSpec(operation="unknown_op"))  # type: ignore[arg-type]


def test_unknown_column_is_rejected_before_sql_rendering(generator: SQLGenerator) -> None:
    """Specification fields cannot become identifier-injection SQL."""

    with pytest.raises(SQLGenerationError, match="Unknown sales-view column"):
        generator.generate(
            AnalyticalSpec(filters=[Filter("country; DROP TABLE targets", "=", "India")])
        )


class MockLLM:
    """Small fake used to exercise the injected fallback seam."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_call: dict[str, str] | None = None

    def generate(self, *, system: str, user: str) -> str:
        """Record a call and return configured content without external I/O."""

        self.last_call = {"system": system, "user": user}
        return self.response


def test_llm_fallback_is_schema_grounded(semantic_layer: DuckDBSemanticLayer) -> None:
    """A novel but valid shape can use the injected, offline fallback client."""

    mock = MockLLM("SELECT 1 AS value")
    sql = SQLGenerator(semantic_layer, mock).generate(
        AnalyticalSpec(operation="aggregate", metrics=["revenue"], transforms=["mom"])
    )
    assert sql == "SELECT 1 AS value"
    assert mock.last_call is not None
    assert "<analytical_spec>" in mock.last_call["user"]
    assert "v_sales" in mock.last_call["user"]


def test_llm_fallback_strips_fences_and_rejects_unsafe_sql(
    semantic_layer: DuckDBSemanticLayer,
) -> None:
    """Fallback output is cleaned but cannot return a mutating statement."""

    fenced = SQLGenerator(semantic_layer, MockLLM("```sql\nSELECT 1 AS value\n```"))
    spec = AnalyticalSpec(operation="aggregate", metrics=["revenue"], transforms=["mom"])
    assert fenced.generate(spec) == "SELECT 1 AS value"

    unsafe = SQLGenerator(semantic_layer, MockLLM("DROP TABLE targets"))
    with pytest.raises(SQLGenerationError, match="SELECT or WITH"):
        unsafe.generate(spec)
