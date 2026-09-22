"""Deterministic resolution of tagged temporal constraints."""
from __future__ import annotations

import calendar
from dataclasses import replace

from engine.interfaces import SemanticLayerProtocol
from engine.types import TimeConstraint


class TemporalAnchor:
    """Resolve supported periods against the dates available in `v_sales`."""

    _MONTH_NUMBERS = {
        **{name.lower(): month for month, name in enumerate(calendar.month_name) if name},
        **{name.lower(): month for month, name in enumerate(calendar.month_abbr) if name},
        "sept": 9,
    }

    def __init__(self, semantic_layer: SemanticLayerProtocol) -> None:
        self._sl = semantic_layer

    def resolve(self, constraint: TimeConstraint) -> TimeConstraint:
        """Return a constraint with an exact/latest period resolved when possible."""

        if constraint.resolved_value is not None:
            return constraint
        if constraint.resolution == "EXACT_PERIOD":
            return self._resolve_month(constraint)
        if constraint.resolution == "LATEST_PERIOD":
            return self._resolve_latest(constraint)
        return constraint

    def _resolve_month(self, constraint: TimeConstraint) -> TimeConstraint:
        """Resolve a month name to the latest matching `YYYY-MM` dataset period."""

        month_number = self._MONTH_NUMBERS.get(constraint.raw.lower())
        if month_number is None:
            return constraint
        result = self._sl.execute(
            "SELECT MAX(month) AS period FROM v_sales "
            f"WHERE EXTRACT(MONTH FROM order_date) = {month_number}"
        )
        value = result.loc[0, "period"]
        return replace(constraint, resolved_value=str(value) if value is not None else None)

    def _resolve_latest(self, constraint: TimeConstraint) -> TimeConstraint:
        """Resolve the greatest value from a trusted time column."""

        if not constraint.column.replace("_", "").isalnum():
            raise ValueError(f"Unsafe time column: {constraint.column!r}")
        result = self._sl.execute(
            f'SELECT MAX("{constraint.column}") AS period FROM v_sales'
        )
        value = result.loc[0, "period"]
        return replace(constraint, resolved_value=str(value) if value is not None else None)
