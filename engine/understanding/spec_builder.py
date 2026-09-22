"""Orchestrate optional LLM and deterministic Phase 4 spec construction."""
from __future__ import annotations

from engine.interfaces import SemanticLayerProtocol
from engine.understanding.rule_builder import RuleSpecBuilder
from engine.understanding.temporal_anchor import TemporalAnchor
from engine.types import AnalyticalSpec, Filter, TaggedQuery


class SpecBuilder:
    """Build, ground, default, and time-resolve a specification from tagged text."""

    def __init__(self, semantic_layer: SemanticLayerProtocol, llm_builder: object | None = None) -> None:
        self._sl = semantic_layer
        self._anchor = TemporalAnchor(semantic_layer)
        self._rule_builder = RuleSpecBuilder(semantic_layer)
        self._llm_builder = llm_builder

    def build(self, tagged: TaggedQuery) -> AnalyticalSpec | None:
        """Prefer deterministic rules and fallback to a grounded optional LLM result."""

        spec = self._rule_builder.build(tagged)
        if spec is None:
            spec = self._build_with_llm(tagged)
        if spec is None:
            return None
        self._ensure_defaults_marked(spec, tagged)
        return self._resolve_time(spec)

    def _build_with_llm(self, tagged: TaggedQuery) -> AnalyticalSpec | None:
        """Return a grounded LLM spec when an optional builder is available."""

        if self._llm_builder is None:
            return None
        build = getattr(self._llm_builder, "build", None)
        candidate = build(tagged) if callable(build) else None
        if not isinstance(candidate, AnalyticalSpec):
            return None
        return candidate if self._grounding_ok(candidate, tagged) and self._structurally_valid(candidate) else None

    def _grounding_ok(self, spec: AnalyticalSpec, tagged: TaggedQuery) -> bool:
        """Reject specs that mention untagged schema entities or invalid fields."""

        metrics = {span.canonical for span in tagged.spans if span.role == "metric"}
        dimensions = {span.canonical for span in tagged.spans if span.role == "dimension"}
        allowed_metrics = metrics | {self._sl.get_default_metric()}
        allowed_dimensions = dimensions | {"year"}
        filter_columns = dimensions | {
            span.matched_column for span in tagged.spans if span.role == "value" and span.matched_column
        }
        valid_order = spec.order_by is None or spec.order_by.metric in allowed_metrics
        valid_filters = all(item.column in filter_columns for item in spec.filters)
        valid_metric_filters = all(
            item.metric in allowed_metrics
            and (item.target_metric is None or item.target_metric == self._sl.get_target_column(item.metric))
            for item in spec.metric_filters
        )
        valid_time = spec.time_window is None or self._sl.get_time_column(spec.time_window.raw) == spec.time_window.column
        return (
            set(spec.metrics).issubset(allowed_metrics)
            and set(spec.group_by).issubset(allowed_dimensions)
            and set(spec.partition_by).issubset(allowed_dimensions)
            and valid_order
            and valid_filters
            and valid_metric_filters
            and valid_time
            and (spec.limit is None or spec.limit > 0)
        )

    @staticmethod
    def _structurally_valid(spec: AnalyticalSpec) -> bool:
        """Reject incomplete LLM specifications before they bypass rules."""

        if not spec.metrics:
            return False
        if spec.operation == "rank":
            return bool(spec.group_by and spec.order_by and spec.limit)
        if spec.operation == "compare":
            return bool(spec.group_by and spec.metric_filters)
        if spec.operation == "trend":
            return spec.time_comparison is not None
        return True

    def _ensure_defaults_marked(self, spec: AnalyticalSpec, tagged: TaggedQuery) -> None:
        """Record a default metric consistently for both rule and LLM paths."""

        has_metric_span = any(span.role == "metric" for span in tagged.spans)
        if not has_metric_span and spec.metrics and "default_metric" not in spec.defaults_applied:
            spec.defaults_applied.append("default_metric")

    def _resolve_time(self, spec: AnalyticalSpec) -> AnalyticalSpec:
        """Resolve temporal constraints and render comparison periods as filters."""

        if spec.time_window is not None:
            spec.time_window = self._anchor.resolve(spec.time_window)
        if spec.time_comparison is not None:
            spec.time_comparison.baseline = self._anchor.resolve(spec.time_comparison.baseline)
            if spec.time_comparison.target is not None:
                spec.time_comparison.target = self._anchor.resolve(spec.time_comparison.target)
        if spec.operation == "compare" and spec.time_window is not None:
            if spec.time_window.resolved_value is not None:
                spec.filters.append(
                    Filter(spec.time_window.column, "=", spec.time_window.resolved_value)
                )
            spec.time_window = None
        return spec
