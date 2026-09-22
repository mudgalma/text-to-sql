"""Compute confidence from observable NLU, SQL, and execution signals."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from engine.interfaces import SemanticLayerProtocol
from engine.execution.executor import ExecutionResult
from engine.types import AnalyticalSpec, TaggedQuery


W_EXECUTION = 0.30
W_SCHEMA = 0.20
W_NLU = 0.15
W_AGREEMENT = 0.15
W_PLAUSIBILITY = 0.10
W_DEFAULTS = 0.10


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Individual normalized signals and their final confidence score."""

    execution_success: float
    schema_validity: float
    nlu_confidence: float
    agreement_signal: float
    result_plausibility: float
    defaults_score: float
    final: float


class ConfidenceScorer:
    """Return a reproducible confidence score from already-observed pipeline facts."""

    _KEYWORDS = frozenset(
        {
            "all", "and", "as", "asc", "between", "by", "case", "cast", "count",
            "date", "decimal", "dense_rank", "desc", "distinct", "double", "else",
            "end", "exists", "extract", "false", "filter", "float", "from", "group",
            "having", "in", "inner", "int", "interval", "is", "join", "lag", "lead",
            "left", "like", "limit", "max", "min", "not", "null", "nullif", "on",
            "or", "order", "outer", "over", "partition", "pct", "rank", "round", "select",
            "sum", "then", "true", "union", "using", "val", "value", "varchar", "when",
            "where", "with", "year", "month", "yoy_growth",
        }
    )
    _IDENTIFIER = re.compile(r"\b[a-z_][a-z0-9_]*\b", re.IGNORECASE)
    _STRING_LITERAL = re.compile(r"'(?:''|[^'])*'")
    _TABLE_ALIAS = re.compile(
        r"\b(?:from|join)\s+[a-z_][a-z0-9_]*(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?",
        re.IGNORECASE,
    )
    _CTE_ALIAS = re.compile(r"\bwith\s+([a-z_][a-z0-9_]*)\s+as\b", re.IGNORECASE)
    _OUTPUT_ALIAS = re.compile(r"\bas\s+([a-z_][a-z0-9_]*)\b", re.IGNORECASE)

    def __init__(self, semantic_layer: SemanticLayerProtocol) -> None:
        self._semantic_layer = semantic_layer
        self._valid_columns = {name.lower() for name in semantic_layer.get_view_schema()}
        targets = semantic_layer.execute("DESCRIBE targets")
        self._valid_columns.update(targets["column_name"].str.lower())
        self._relation_names = {
            getattr(semantic_layer, "VIEW_NAME", "v_sales").lower(),
            "targets",
        }

    def score(
        self,
        tagged: TaggedQuery,
        spec: AnalyticalSpec,
        execution: ExecutionResult,
        retries_used: int,
        rule_spec_present: bool,
        llm_spec_present: bool,
        specs_agree: bool | None = None,
    ) -> ConfidenceBreakdown:
        """Score one completed query using explicit, inspectable pipeline signals."""

        execution_score = self._score_execution(execution.success)
        schema_score = self._score_schema(execution.sql)
        nlu_score = self._score_nlu(tagged)
        agreement_score = self._score_agreement(
            rule_spec_present, llm_spec_present, specs_agree
        )
        plausibility_score = self._score_plausibility(execution.df, retries_used)
        defaults_score = self._score_defaults(spec)
        weighted_score = (
            W_EXECUTION * execution_score
            + W_SCHEMA * schema_score
            + W_NLU * nlu_score
            + W_AGREEMENT * agreement_score
            + W_PLAUSIBILITY * plausibility_score
            + W_DEFAULTS * defaults_score
        )
        final = min(weighted_score, 0.25) if not execution.success else weighted_score
        return ConfidenceBreakdown(
            execution_success=round(execution_score, 3),
            schema_validity=round(schema_score, 3),
            nlu_confidence=round(nlu_score, 3),
            agreement_signal=round(agreement_score, 3),
            result_plausibility=round(plausibility_score, 3),
            defaults_score=round(defaults_score, 3),
            final=round(max(0.0, min(1.0, final)), 3),
        )

    @staticmethod
    def _score_execution(success: bool) -> float:
        """Score execution as a necessary condition for a reliable result."""

        return 1.0 if success else 0.0

    def _score_schema(self, sql: str) -> float:
        """Estimate identifier validity while ignoring literals, aliases, and SQL grammar."""

        if not sql:
            return 0.0
        normalized = self._STRING_LITERAL.sub("", sql.lower())
        aliases = self._aliases(normalized)
        identifiers = set(self._IDENTIFIER.findall(normalized))
        function_names = {
            match.group(1)
            for match in re.finditer(r"\b([a-z_][a-z0-9_]*)\s*\(", normalized)
        }
        candidates = (
            identifiers
            - self._KEYWORDS
            - self._relation_names
            - aliases
            - function_names
        )
        if not candidates:
            return 1.0
        return sum(name in self._valid_columns for name in candidates) / len(candidates)

    def _aliases(self, sql: str) -> set[str]:
        """Find relation and projection aliases that are not schema columns."""

        aliases = {match.group(1) for match in self._CTE_ALIAS.finditer(sql)}
        aliases.update(match.group(1) for match in self._OUTPUT_ALIAS.finditer(sql))
        aliases.update(
            match.group(1)
            for match in self._TABLE_ALIAS.finditer(sql)
            if match.group(1) is not None and match.group(1) not in self._KEYWORDS
        )
        return aliases

    @staticmethod
    def _score_nlu(tagged: TaggedQuery) -> float:
        """Reward recognized spans and reduce confidence for recorded ambiguity."""

        if not tagged.spans:
            return 0.0
        return max(0.0, 1.0 - 0.3 * len(tagged.conflicts))

    @staticmethod
    def _score_agreement(
        rule_present: bool, llm_present: bool, specs_agree: bool | None
    ) -> float:
        """Score actual rule/LLM agreement, not merely the existence of both paths."""

        if rule_present and llm_present:
            if specs_agree is True:
                return 1.0
            if specs_agree is False:
                return 0.3
        if rule_present or llm_present:
            return 0.7
        return 0.3

    @staticmethod
    def _score_plausibility(result_df: pd.DataFrame | None, retries_used: int) -> float:
        """Lower confidence for empty results and for results that needed repair."""

        base = 1.0 if result_df is not None and not result_df.empty else 0.4
        retries = max(0, retries_used)
        return max(0.0, base - min(0.3, 0.15 * retries))

    @staticmethod
    def _score_defaults(spec: AnalyticalSpec) -> float:
        """Reduce confidence when the user omitted information filled by a default."""

        return 0.5 if spec.defaults_applied else 1.0
