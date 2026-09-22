"""Hand-authored expected specifications for the assignment's eight queries."""
from __future__ import annotations

from dataclasses import dataclass

from engine.types import (
    AnalyticalSpec,
    Filter,
    MetricFilter,
    OrderSpec,
    TimeComparison,
    TimeConstraint,
)


@dataclass(frozen=True)
class GoldenSpec:
    """One natural-language query and its expected semantic specification."""

    query: str
    expected: AnalyticalSpec


GOLDEN_SPECS: tuple[GoldenSpec, ...] = (
    GoldenSpec(
        query="Total sales in India for March",
        expected=AnalyticalSpec(
            operation="aggregate",
            metrics=["revenue"],
            filters=[Filter(column="country", operator="=", value="India")],
            time_window=TimeConstraint(
                raw="march",
                column="month",
                resolution="EXACT_PERIOD",
                resolved_value="2024-03",
            ),
        ),
    ),
    GoldenSpec(
        query="Top 2 cities by profit",
        expected=AnalyticalSpec(
            operation="rank",
            metrics=["profit"],
            group_by=["city"],
            order_by=OrderSpec(metric="profit", direction="DESC"),
            limit=2,
        ),
    ),
    GoldenSpec(
        query="Average order value by region",
        expected=AnalyticalSpec(
            operation="aggregate",
            metrics=["avg_order_value"],
            group_by=["region"],
        ),
    ),
    GoldenSpec(
        query="Which region missed its target in Feb?",
        expected=AnalyticalSpec(
            operation="compare",
            metrics=["revenue"],
            group_by=["region"],
            filters=[Filter(column="month", operator="=", value="2024-02")],
            metric_filters=[
                MetricFilter(
                    metric="revenue",
                    operator="<",
                    target_metric="target_revenue",
                )
            ],
            defaults_applied=["default_metric"],
        ),
    ),
    GoldenSpec(
        query="Sales contribution % by category",
        expected=AnalyticalSpec(
            operation="aggregate",
            metrics=["revenue"],
            transforms=["contribution_pct"],
            group_by=["product_category"],
        ),
    ),
    GoldenSpec(
        query="Top product in each region",
        expected=AnalyticalSpec(
            operation="rank",
            metrics=["revenue"],
            group_by=["region", "product_name"],
            partition_by=["region"],
            order_by=OrderSpec(metric="revenue", direction="DESC"),
            limit=1,
            defaults_applied=["default_metric"],
        ),
    ),
    GoldenSpec(
        query="YoY growth in revenue",
        expected=AnalyticalSpec(
            operation="trend",
            metrics=["revenue"],
            transforms=["yoy"],
            group_by=["year"],
            time_comparison=TimeComparison(
                baseline=TimeConstraint(
                    raw="current year",
                    column="year",
                    resolution="LATEST_PERIOD",
                    resolved_value="2024",
                ),
                target=TimeConstraint(
                    raw="previous year",
                    column="year",
                    resolution="PERIOD_OVER_PERIOD",
                ),
                delta_type="YoY",
            ),
        ),
    ),
    GoldenSpec(
        query="Revenue of top 3 customers per region",
        expected=AnalyticalSpec(
            operation="rank",
            metrics=["revenue"],
            group_by=["region", "customer_id"],
            partition_by=["region"],
            order_by=OrderSpec(metric="revenue", direction="DESC"),
            limit=3,
        ),
    ),
)
