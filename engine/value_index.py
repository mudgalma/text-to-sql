"""Cardinality-tiered lookup for NLU entity recognition."""
from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher

from engine.interfaces import SemanticLayerProtocol
from engine.types import TIER_RANK, ValueMatch


class CardinalityTieredValueIndex:
    """Match known dimension values using exact, alias, and fuzzy lookup."""

    DEFAULT_CARDINALITY_LIMIT = 200
    DEFAULT_SAMPLE_SIZE = 1000
    DEFAULT_FUZZY_THRESHOLD = 0.85
    MIN_FUZZY_LENGTH = 4

    EXCLUDED_FROM_INDEXING = frozenset({"order_id", "customer_id"})
    DATE_LIKE = frozenset({"order_date"})

    def __init__(
        self,
        semantic_layer: SemanticLayerProtocol,
        cardinality_limit: int = DEFAULT_CARDINALITY_LIMIT,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> None:
        if cardinality_limit < 1:
            raise ValueError("cardinality_limit must be at least 1")
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0.0 and 1.0")

        self._sl = semantic_layer
        self._cardinality_limit = cardinality_limit
        self._sample_size = sample_size
        self._fuzzy_threshold = fuzzy_threshold
        self._lookup: dict[str, dict[str, str]] = {}
        self._cardinality: dict[str, int] = {}
        self._aliases = self._load_aliases()
        self._build()

    def _load_aliases(self) -> dict[str, dict[str, str]]:
        """Return optional per-column aliases exposed by the semantic layer."""

        getter = getattr(self._sl, "get_value_aliases", None)
        aliases = getter() if callable(getter) else {}
        if not isinstance(aliases, dict):
            raise ValueError("Value aliases must be a dictionary.")
        return aliases

    def _build(self) -> None:
        """Build one normalized lookup dictionary for every indexable dimension."""

        for dimension in self._sl.get_dimension_names():
            if dimension in self.EXCLUDED_FROM_INDEXING or dimension in self.DATE_LIKE:
                continue
            cardinality = self._count_distinct(dimension)
            self._cardinality[dimension] = cardinality
            self._lookup[dimension] = (
                self._load_exact(dimension)
                if cardinality <= self._cardinality_limit
                else self._load_sample(dimension)
            )

    def _count_distinct(self, column: str) -> int:
        """Count distinct values in a trusted semantic-layer column."""

        identifier = self._quote_identifier(column)
        row = self._sl.execute(f"SELECT COUNT(DISTINCT {identifier}) FROM v_sales")
        return int(row.iloc[0, 0])

    def _load_exact(self, column: str) -> dict[str, str]:
        """Load all distinct values for a low-cardinality column."""

        identifier = self._quote_identifier(column)
        dataframe = self._sl.execute(f"SELECT DISTINCT {identifier} FROM v_sales")
        return self._to_lookup(dataframe.iloc[:, 0].dropna().tolist())

    def _load_sample(self, column: str) -> dict[str, str]:
        """Load a bounded distinct sample for a high-cardinality column."""

        identifier = self._quote_identifier(column)
        dataframe = self._sl.execute(
            f"SELECT DISTINCT {identifier} FROM v_sales LIMIT {self._sample_size}"
        )
        return self._to_lookup(dataframe.iloc[:, 0].dropna().tolist())

    @staticmethod
    def _to_lookup(values: list[object]) -> dict[str, str]:
        """Normalize a sequence of stored values for case-insensitive lookup."""

        return {str(value).lower(): str(value) for value in values}

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Quote a schema identifier after rejecting unsafe characters."""

        if not identifier or not identifier.replace("_", "").isalnum():
            raise ValueError(f"Unsafe dimension identifier: {identifier!r}")
        return f'"{identifier}"'

    def match(
        self, term: str, threshold: float | None = None
    ) -> list[ValueMatch]:
        """Return ranked matches for one user-provided value candidate."""

        term_clean = term.strip().lower()
        if not term_clean:
            return []
        active_threshold = self._fuzzy_threshold if threshold is None else threshold
        if not 0.0 <= active_threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        matches = [
            match
            for column, lookup in self._lookup.items()
            if (match := self._match_column(term_clean, column, lookup, active_threshold))
            is not None
        ]
        return sorted(
            matches,
            key=lambda match: (match.score, TIER_RANK[match.match_type]),
            reverse=True,
        )

    def _match_column(
        self,
        term: str,
        column: str,
        lookup: dict[str, str],
        threshold: float,
    ) -> ValueMatch | None:
        """Apply exact, alias, then guarded fuzzy matching for one column."""

        if term in lookup:
            return ValueMatch(column, lookup[term], 1.0, "exact")

        alias_map = self._aliases.get(column, {})
        alias_value = alias_map.get(term)
        if isinstance(alias_value, str) and alias_value.lower() in lookup:
            return ValueMatch(
                column, lookup[alias_value.lower()], 1.0, "morphological"
            )

        if len(term) < self.MIN_FUZZY_LENGTH:
            return None
        best_value, best_score = self._best_similarity(term, lookup)
        if best_value is not None and best_score >= threshold:
            return ValueMatch(column, lookup[best_value], best_score, "fuzzy")
        return None

    @staticmethod
    def _best_similarity(
        term: str, candidates: Iterable[str]
    ) -> tuple[str | None, float]:
        """Return the highest-ratio candidate for a normalized term."""

        best_value: str | None = None
        best_score = 0.0
        for candidate in candidates:
            if candidate.startswith(term) or term.startswith(candidate):
                continue
            score = SequenceMatcher(None, term, candidate).ratio()
            if score > best_score:
                best_value = candidate
                best_score = score
        return best_value, best_score

    def is_exact_column(self, column: str) -> bool:
        """Return whether a column was loaded fully rather than sampled."""

        return (
            column in self._cardinality
            and self._cardinality[column] <= self._cardinality_limit
        )

    def indexed_columns(self) -> list[str]:
        """Return indexed dimensions in deterministic order."""

        return sorted(self._lookup)

    def cardinality(self, column: str) -> int:
        """Return a column's observed cardinality, or -1 when unindexed."""

        return self._cardinality.get(column, -1)
