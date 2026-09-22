"""Protocol contracts for the NLU pipeline."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from engine.types import NLUResult, TaggedQuery, ValueMatch


class SemanticLayerProtocol(Protocol):
    """Business schema provider. No indexing or search responsibilities."""

    def execute(self, sql: str) -> pd.DataFrame: ...

    def get_metric_names(self) -> list[str]: ...

    def get_metric_definition(self, name: str) -> str | None:
        """Return a metric formula, or None when the metric is unknown."""

    def get_metric_sql(self, name: str) -> str | None:
        """Return validated executable SQL for a metric, if it is known."""

    def get_dimension_names(self) -> list[str]: ...

    def get_synonyms(self) -> dict[str, str]: ...

    def get_time_mappings(self) -> dict[str, str]: ...

    def get_target_column(self, metric_name: str) -> str | None: ...

    def get_default_metric(self) -> str: ...

    def get_time_column(self, time_canonical: str) -> str | None: ...

    def get_view_schema(self) -> dict[str, str]: ...


class ValueIndexProtocol(Protocol):
    """NLU-time value lookup consumed by a tagger."""

    def match(
        self, term: str, threshold: float | None = None
    ) -> list[ValueMatch]: ...

    def is_exact_column(self, column: str) -> bool: ...

    def indexed_columns(self) -> list[str]: ...

    def cardinality(self, column: str) -> int: ...


class NLUProtocol(Protocol):
    """Top-level NLU entry point."""

    def understand(self, raw_query: str) -> NLUResult: ...


class TaggerProtocol(Protocol):
    """Lexically tag a raw analytics query before specification building."""

    def tag(self, raw_query: str) -> TaggedQuery: ...
