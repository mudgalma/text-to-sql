"""Generate validated DuckDB SQL from an ``AnalyticalSpec``."""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from engine.interfaces import SemanticLayerProtocol
from engine.generation.prompts import SQL_GEN_SYSTEM, build_sql_user_prompt
from engine.generation.templates import TEMPLATES
from engine.types import AnalyticalSpec, Filter, MetricFilter, TimeConstraint


LOGGER = logging.getLogger(__name__)


class SQLGenerationError(ValueError):
    """Raised when a specification cannot safely become executable SQL."""


class SQLFallbackClient(Protocol):
    """Minimal client contract for the optional SQL-generation fallback."""

    def generate(self, *, system: str, user: str) -> str:
        """Return SQL for a system and user prompt."""


class SQLGenerator:
    """Render supported specs deterministically, with an optional LLM fallback."""

    _IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _FILTER_OPERATORS = frozenset(
        {"=", "!=", ">", "<", ">=", "<=", "IN", "NOT IN", "LIKE", "IS NULL", "IS NOT NULL"}
    )
    _METRIC_FILTER_OPERATORS = frozenset({"=", "!=", ">", "<", ">=", "<="})
    _MAX_LIMIT = 10_000
    _DISALLOWED_SQL = re.compile(
        r"\b(?:ATTACH|COPY|CREATE|DELETE|DROP|EXPORT|IMPORT|INSERT|INSTALL|LOAD|UPDATE)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        semantic_layer: SemanticLayerProtocol,
        llm_client: SQLFallbackClient | None = None,
    ) -> None:
        self._sl = semantic_layer
        self._llm = llm_client
        self._view = self._validate_identifier(
            getattr(semantic_layer, "VIEW_NAME", "v_sales"), "view name"
        )
        self._view_schema = set(semantic_layer.get_view_schema())
        self._metrics = set(semantic_layer.get_metric_names())
        self._dimensions = set(semantic_layer.get_dimension_names()) | {"month", "year"}
        self._target_columns = {
            metric: target
            for metric in self._metrics
            if (target := semantic_layer.get_target_column(metric)) is not None
        }

    def generate(self, spec: AnalyticalSpec) -> str:
        """Return SQL for a valid spec, or fail before rendering unsafe SQL."""

        self._validate_spec_references(spec)
        key = self._template_key(spec)
        if key is not None:
            return self._fill_template(TEMPLATES[key], spec)
        if self._llm is None:
            raise SQLGenerationError(
                f"No deterministic SQL template supports operation {spec.operation!r}."
            )
        return self._generate_via_llm(spec)

    def _template_key(self, spec: AnalyticalSpec) -> str | None:
        """Choose a template only when it fully supports the spec's shape."""

        if spec.operation == "aggregate":
            if "contribution_pct" in spec.transforms and len(spec.group_by) == 1:
                return "contribution"
            if not spec.transforms and len(spec.group_by) == 1:
                return "group_by"
            if not spec.transforms and not spec.group_by:
                return "aggregate"
        if spec.operation == "rank":
            if spec.partition_by and len(spec.group_by) >= 2:
                return "rank_partitioned"
            if not spec.partition_by and len(spec.group_by) == 1:
                return "rank"
        if spec.operation == "compare" and len(spec.group_by) == 1:
            return "compare"
        if spec.operation == "trend" and spec.group_by == ["year"] and "yoy" in spec.transforms:
            return "trend"
        return None

    def _fill_template(self, template: str, spec: AnalyticalSpec) -> str:
        """Render one selected template after all fields have been validated."""

        comparison = spec.operation == "compare"
        dimension = self._column(spec.group_by[0]) if spec.group_by else ""
        kwargs: dict[str, Any] = {
            "view": self._view,
            "targets_table": "targets",
            "agg_expr": self._aggregate_expression(spec, "s" if comparison else None),
            "dimension": dimension,
            "group_dims": ", ".join(self._column(name) for name in spec.group_by),
            "partition_dims": ", ".join(self._column(name) for name in spec.partition_by),
            "direction": spec.order_by.direction if spec.order_by else "DESC",
            "limit": self._limit(spec),
            "where": self._build_where(spec, "s" if comparison else None),
        }
        if "{target_expr}" in template:
            target_column = self._target_column(spec)
            kwargs["target_expr"] = f"CAST(t.{target_column} AS DOUBLE)"
            kwargs["time_column"] = self._compare_time_column(spec)
            kwargs["mf_operator"] = self._metric_filter_operator(spec)
        return template.format(**kwargs).strip()

    def _validate_spec_references(self, spec: AnalyticalSpec) -> None:
        """Reject unknown fields and unsafe values before SQL rendering or prompting."""

        if not spec.metrics:
            if spec.operation in {"rank", "compare", "trend"}:
                raise SQLGenerationError("Only aggregate queries may omit a metric.")
        elif len(spec.metrics) != 1 or spec.metrics[0] not in self._metrics:
            raise SQLGenerationError("A supported query must name one configured metric.")
        for name in [*spec.group_by, *spec.partition_by]:
            self._require_dimension(name)
        if spec.order_by is not None:
            if spec.order_by.metric not in self._metrics:
                raise SQLGenerationError("Order metric is not configured.")
            if spec.order_by.direction not in {"ASC", "DESC"}:
                raise SQLGenerationError("Order direction must be ASC or DESC.")
        if spec.partition_by and not set(spec.partition_by).issubset(spec.group_by):
            raise SQLGenerationError("Partition dimensions must also be grouping dimensions.")
        self._limit(spec)
        for item in spec.filters:
            self._validate_filter(item)
        if spec.time_window is not None:
            self._validate_time_constraint(spec.time_window)
        for item in spec.metric_filters:
            self._validate_metric_filter(item)
        if spec.operation == "compare":
            self._target_column(spec)
            self._compare_time_column(spec)

    def _aggregate_expression(self, spec: AnalyticalSpec, qualifier: str | None) -> str:
        """Return a safe aggregation expression for the sole requested metric."""

        if not spec.metrics:
            return "COUNT(*)"
        metric = spec.metrics[0]
        if metric in self._view_schema:
            return f"SUM({self._qualified_column(metric, qualifier)})"
        expression = self._sl.get_metric_sql(metric)
        if expression is None:
            raise SQLGenerationError(f"No executable definition exists for metric {metric!r}.")
        return self._qualify_metric_expression(expression, qualifier)

    def _qualify_metric_expression(self, expression: str, qualifier: str | None) -> str:
        """Qualify known view columns in a semantic-layer-validated expression."""

        if qualifier is None:
            return expression
        return re.sub(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            lambda match: self._qualified_column(match.group(0), qualifier)
            if match.group(0) in self._view_schema
            else match.group(0),
            expression,
        )

    def _build_where(self, spec: AnalyticalSpec, qualifier: str | None) -> str:
        """Render row predicates from validated filters and time constraints."""

        clauses = [self._render_filter(item, qualifier) for item in spec.filters]
        if spec.time_window is not None:
            clauses.append(self._render_time(spec.time_window, qualifier))
        return f"WHERE {' AND '.join(clauses)}" if clauses else ""

    def _render_filter(self, item: Filter, qualifier: str | None) -> str:
        """Render one already-validated row predicate."""

        column = self._qualified_column(item.column, qualifier)
        if item.operator in {"IS NULL", "IS NOT NULL"}:
            return f"{column} {item.operator}"
        if item.operator in {"IN", "NOT IN"}:
            values = self._as_non_empty_values(item.value)
            return f"{column} {item.operator} ({', '.join(self._quote(value) for value in values)})"
        return f"{column} {item.operator} {self._quote(item.value)}"

    def _render_time(self, constraint: TimeConstraint, qualifier: str | None) -> str:
        """Render a resolved temporal constraint as a row predicate."""

        value = constraint.resolved_value or constraint.raw
        return f"{self._qualified_column(constraint.column, qualifier)} = {self._quote(value)}"

    def _target_column(self, spec: AnalyticalSpec) -> str:
        """Return the configured target field required by a compare specification."""

        if len(spec.metric_filters) != 1:
            raise SQLGenerationError("Compare queries require exactly one metric filter.")
        item = spec.metric_filters[0]
        expected = self._target_columns.get(item.metric)
        if expected is None or item.target_metric != expected:
            raise SQLGenerationError("Compare target must match configured target metadata.")
        return self._validate_identifier(expected, "target column")

    def _metric_filter_operator(self, spec: AnalyticalSpec) -> str:
        """Return the validated comparison operator for a compare query."""

        return spec.metric_filters[0].operator

    def _compare_time_column(self, spec: AnalyticalSpec) -> str:
        """Find the time key shared by sales and target data."""

        if spec.time_window is not None:
            return self._column(spec.time_window.column)
        for item in spec.filters:
            if item.column in {"month", "year"}:
                return self._column(item.column)
        raise SQLGenerationError("Compare queries require a month or year time constraint.")

    def _validate_filter(self, item: Filter) -> None:
        """Validate an individual row-level predicate."""

        self._require_view_column(item.column)
        if item.operator not in self._FILTER_OPERATORS:
            raise SQLGenerationError(f"Unsupported filter operator: {item.operator!r}")
        if item.operator in {"IS NULL", "IS NOT NULL"}:
            return
        if item.operator in {"IN", "NOT IN"}:
            self._as_non_empty_values(item.value)
            return
        self._validate_literal(item.value)

    def _validate_time_constraint(self, constraint: TimeConstraint) -> None:
        """Validate the time column and value before it is used as a filter."""

        self._require_view_column(constraint.column)
        self._validate_literal(constraint.resolved_value or constraint.raw)

    def _validate_metric_filter(self, item: MetricFilter) -> None:
        """Validate a HAVING-like filter used by target comparisons."""

        if item.metric not in self._metrics:
            raise SQLGenerationError("Metric filter references an unknown metric.")
        if item.operator not in self._METRIC_FILTER_OPERATORS:
            raise SQLGenerationError("Metric filter requires a scalar comparison operator.")
        if item.target_metric is None and item.target_value is None:
            raise SQLGenerationError("Metric filter requires a target metric or numeric value.")
        if item.target_value is not None:
            self._validate_literal(item.target_value)

    def _generate_via_llm(self, spec: AnalyticalSpec) -> str:
        """Ask an injected fallback client for one read-only SQL query."""

        try:
            raw = self._llm.generate(
                system=SQL_GEN_SYSTEM,
                user=build_sql_user_prompt(
                    spec,
                    self._view,
                    self._sl.get_view_schema(),
                    self._target_columns,
                ),
            )
        except (AttributeError, ConnectionError, OSError, TimeoutError, ValueError) as error:
            LOGGER.warning("sql_fallback_generation_failed", exc_info=error)
            raise SQLGenerationError("The SQL fallback could not generate a query.") from error
        return self.clean_read_only_sql(raw)

    @classmethod
    def clean_read_only_sql(cls, raw: str) -> str:
        """Remove a Markdown fence and reject non-read-only SQL."""

        if not isinstance(raw, str):
            raise SQLGenerationError("The SQL fallback returned a non-text response.")
        lines = raw.strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines).strip()
        if not sql or not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
            raise SQLGenerationError("The SQL fallback must return a SELECT or WITH query.")
        if ";" in sql or "--" in sql or "/*" in sql or cls._DISALLOWED_SQL.search(sql):
            raise SQLGenerationError("The SQL fallback returned unsafe SQL.")
        return sql

    def _limit(self, spec: AnalyticalSpec) -> int:
        """Return a bounded positive limit, using ten for ranking defaults."""

        if spec.limit is None:
            return 10
        if isinstance(spec.limit, bool) or not isinstance(spec.limit, int):
            raise SQLGenerationError("Limit must be an integer.")
        if not 1 <= spec.limit <= self._MAX_LIMIT:
            raise SQLGenerationError(f"Limit must be between 1 and {self._MAX_LIMIT}.")
        return spec.limit

    def _require_dimension(self, name: str) -> None:
        """Ensure an identifier is an allowed grouping dimension."""

        if name not in self._dimensions:
            raise SQLGenerationError(f"Unknown dimension: {name!r}")

    def _require_view_column(self, name: str) -> None:
        """Ensure an identifier exists in the canonical sales view."""

        if name not in self._view_schema:
            raise SQLGenerationError(f"Unknown sales-view column: {name!r}")

    def _column(self, name: str) -> str:
        """Return a known-safe view-column identifier."""

        self._require_view_column(name)
        return self._validate_identifier(name, "column")

    def _qualified_column(self, name: str, qualifier: str | None) -> str:
        """Return a safe view-column identifier, optionally table-qualified."""

        column = self._column(name)
        return f"{qualifier}.{column}" if qualifier else column

    @classmethod
    def _validate_identifier(cls, value: str, description: str) -> str:
        """Allow only ordinary SQL identifiers from trusted schema metadata."""

        if not isinstance(value, str) or not cls._IDENTIFIER.fullmatch(value):
            raise SQLGenerationError(f"Invalid {description}.")
        return value

    @classmethod
    def _as_non_empty_values(cls, value: Any) -> list[Any]:
        """Return a non-empty list of safe literal values for an IN predicate."""

        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise SQLGenerationError("IN filters require a non-empty list of values.")
        values = list(value)
        if not values:
            raise SQLGenerationError("IN filters require a non-empty list of values.")
        for item in values:
            cls._validate_literal(item)
        return values

    @staticmethod
    def _validate_literal(value: Any) -> None:
        """Allow only scalar literals that can be quoted without SQL syntax injection."""

        if value is None or isinstance(value, (str, int, bool)):
            return
        if isinstance(value, float) and math.isfinite(value):
            return
        raise SQLGenerationError("Filter values must be scalar text, numbers, booleans, or null.")

    @staticmethod
    def _quote(value: Any) -> str:
        """Render a validated scalar as a DuckDB SQL literal."""

        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return str(value)
