"""Role-sequence rules that deterministically build analytical specifications."""
from __future__ import annotations

from engine.interfaces import SemanticLayerProtocol
from engine.types import (
    AnalyticalSpec,
    EntitySpan,
    Filter,
    MetricFilter,
    OrderSpec,
    TaggedQuery,
    TimeComparison,
    TimeConstraint,
)


CONTRIBUTION_MARKERS = frozenset({"contribution", "%"})
TARGET_COMPARISON_MARKERS = frozenset({"target", "missed"})
PARTITION_KEYWORDS = frozenset({"each", "per", "within"})
PERIOD_DELTAS = {"yoy": "YoY", "mom": "MoM", "qoq": "QoQ"}


class RuleSpecBuilder:
    """Build a spec from generic roles and query grammar, never data literals."""

    def __init__(self, semantic_layer: SemanticLayerProtocol) -> None:
        self._sl = semantic_layer

    def build(self, tagged: TaggedQuery) -> AnalyticalSpec | None:
        """Return a deterministic spec for a supported role sequence."""

        roles = tuple(span.role for span in tagged.spans)
        spans = tagged.spans
        raw_lower = tagged.raw_query.lower()
        if roles == ("metric",):
            return AnalyticalSpec(metrics=[spans[0].canonical])
        if roles == ("metric", "dimension"):
            return self._metric_dimension(spans, raw_lower)
        if roles == ("metric", "value", "time"):
            return self._metric_value_time(spans)
        if roles == ("number", "dimension", "metric"):
            return self._top_n(spans)
        if roles in (("dimension", "dimension"), ("dimension", "dimension", "metric")):
            return self._top_per_group(tagged, spans)
        if roles == ("metric", "number", "dimension", "dimension"):
            return self._nested_rank(tagged, spans)
        if roles == ("time", "metric"):
            return self._period_trend(spans)
        if roles == ("dimension", "time") and self._has_target_marker(raw_lower):
            return self._target_comparison(spans)
        return None

    def _metric_dimension(
        self, spans: list[EntitySpan], raw_lower: str
    ) -> AnalyticalSpec:
        """Build aggregation or contribution specifications by one dimension."""

        transforms = ["contribution_pct"] if self._has_contribution_marker(raw_lower) else []
        return AnalyticalSpec(
            metrics=[spans[0].canonical],
            group_by=[spans[1].canonical],
            transforms=transforms,
        )

    def _metric_value_time(self, spans: list[EntitySpan]) -> AnalyticalSpec | None:
        """Build a metric aggregation constrained by a value and a period."""

        metric, value, time = spans
        if value.matched_column is None:
            return None
        time_window = self._time_constraint(time)
        if time_window is None:
            return None
        return AnalyticalSpec(
            metrics=[metric.canonical],
            filters=[Filter(value.matched_column, "=", value.canonical)],
            time_window=time_window,
        )

    @staticmethod
    def _top_n(spans: list[EntitySpan]) -> AnalyticalSpec:
        """Build a global top-N ranking specification."""

        number, dimension, metric = spans
        return AnalyticalSpec(
            operation="rank",
            metrics=[metric.canonical],
            group_by=[dimension.canonical],
            order_by=OrderSpec(metric.canonical),
            limit=int(number.canonical),
        )

    def _top_per_group(
        self, tagged: TaggedQuery, spans: list[EntitySpan]
    ) -> AnalyticalSpec:
        """Build a top-one-per-group rank using the configured default metric."""

        dimensions = [span for span in spans if span.role == "dimension"]
        metric_spans = [span for span in spans if span.role == "metric"]
        metric = metric_spans[0].canonical if metric_spans else self._sl.get_default_metric()
        return self._partitioned_rank(tagged, dimensions, metric, limit=1)

    def _nested_rank(
        self, tagged: TaggedQuery, spans: list[EntitySpan]
    ) -> AnalyticalSpec:
        """Build a top-N-per-group ranking with an explicit metric and number."""

        metric, number, *dimensions = spans
        return self._partitioned_rank(
            tagged,
            dimensions,
            metric.canonical,
            limit=int(number.canonical),
        )

    def _partitioned_rank(
        self,
        tagged: TaggedQuery,
        dimensions: list[EntitySpan],
        metric: str,
        limit: int,
    ) -> AnalyticalSpec:
        """Build ranking fields with the partition dimension ordered first."""

        partition = self._find_partition_dimension(tagged, dimensions)
        other = next(span.canonical for span in dimensions if span.canonical != partition)
        return AnalyticalSpec(
            operation="rank",
            metrics=[metric],
            group_by=[partition, other],
            partition_by=[partition],
            order_by=OrderSpec(metric),
            limit=limit,
        )

    def _period_trend(self, spans: list[EntitySpan]) -> AnalyticalSpec | None:
        """Build a supported period-over-period trend specification."""

        time, metric = spans
        delta_type = PERIOD_DELTAS.get(time.canonical)
        if delta_type is None:
            return None
        return AnalyticalSpec(
            operation="trend",
            metrics=[metric.canonical],
            transforms=[time.canonical],
            group_by=["year"],
            time_comparison=TimeComparison(
                baseline=TimeConstraint("current year", "year", "LATEST_PERIOD"),
                target=TimeConstraint("previous year", "year", "PERIOD_OVER_PERIOD"),
                delta_type=delta_type,
            ),
        )

    def _target_comparison(self, spans: list[EntitySpan]) -> AnalyticalSpec | None:
        """Build an under-target comparison using configured target metadata."""

        dimension, time = spans
        metric = self._sl.get_default_metric()
        target_metric = self._sl.get_target_column(metric)
        time_window = self._time_constraint(time)
        if target_metric is None or time_window is None:
            return None
        return AnalyticalSpec(
            operation="compare",
            metrics=[metric],
            group_by=[dimension.canonical],
            metric_filters=[MetricFilter(metric, "<", target_metric=target_metric)],
            time_window=time_window,
        )

    def _time_constraint(self, span: EntitySpan) -> TimeConstraint | None:
        """Create an unresolved exact-period constraint for a tagged time span."""

        column = self._sl.get_time_column(span.canonical)
        if column is None:
            return None
        return TimeConstraint(span.canonical, column, "EXACT_PERIOD")

    @staticmethod
    def _has_contribution_marker(raw_lower: str) -> bool:
        """Return whether generic contribution grammar appears in the query."""

        return any(marker in raw_lower for marker in CONTRIBUTION_MARKERS)

    @staticmethod
    def _has_target_marker(raw_lower: str) -> bool:
        """Return whether generic target-comparison grammar appears in the query."""

        return any(marker in raw_lower for marker in TARGET_COMPARISON_MARKERS)

    @staticmethod
    def _find_partition_dimension(
        tagged: TaggedQuery, dimensions: list[EntitySpan]
    ) -> str:
        """Find the dimension immediately following generic partition grammar."""

        for index, token in enumerate(tagged.tokens[:-1]):
            if token.text.lower() not in PARTITION_KEYWORDS:
                continue
            next_start = tagged.tokens[index + 1].start
            for dimension in dimensions:
                if dimension.tokens[0].start == next_start:
                    return dimension.canonical
        return dimensions[-1].canonical
