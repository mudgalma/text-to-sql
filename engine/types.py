"""Shared data contracts for the NLU pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Lexical Layer
# ---------------------------------------------------------------------------

TokenType = Literal[
    "metric",
    "dimension",
    "value",
    "time",
    "number",
    "operator",
    "unknown",
]


@dataclass(frozen=True)
class Token:
    """Atomic lexical unit. Diagnostic use only — downstream reads spans."""

    text: str
    start: int
    end: int
    canonical: str | None = None
    role: TokenType | None = None
    matched_column: str | None = None


@dataclass(frozen=True)
class EntitySpan:
    """A semantic entity — one or more contiguous, ordered tokens."""

    tokens: tuple[Token, ...]
    canonical: str
    role: TokenType
    matched_column: str | None = None
    match_quality: "MatchQuality" = "exact"

    @property
    def start(self) -> int:
        return self.tokens[0].start

    @property
    def end(self) -> int:
        return self.tokens[-1].end

    @property
    def text(self) -> str:
        return " ".join(token.text for token in self.tokens)


# ---------------------------------------------------------------------------
# Value Index Result
# ---------------------------------------------------------------------------

MatchType = Literal["exact", "fuzzy", "morphological"]

MatchQuality = MatchType

TIER_RANK: dict[MatchType, int] = {
    "exact": 3,
    "morphological": 2,
    "fuzzy": 1,
}

MATCH_QUALITY_RANK: dict[MatchQuality, int] = dict(TIER_RANK)


@dataclass(frozen=True)
class ValueMatch:
    """Result of a value lookup. The tagger owns text offsets."""

    column: str
    canonical_value: str
    score: float
    match_type: MatchType


@dataclass(frozen=True)
class Conflict:
    """Competing semantic interpretations for one contiguous token range."""

    token_range: tuple[int, int]
    candidates: tuple[EntitySpan, ...]
    chosen: EntitySpan


@dataclass
class TaggedQuery:
    """Lexical tagging output before analytical intent is assembled."""

    raw_query: str
    tokens: list[Token]
    spans: list[EntitySpan]
    conflicts: list[Conflict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

ComparisonOperator = Literal[
    "=",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "IN",
    "NOT IN",
    "LIKE",
    "IS NULL",
    "IS NOT NULL",
]


@dataclass
class Filter:
    """Row-level predicate (rendered as WHERE)."""

    column: str
    operator: ComparisonOperator
    value: Any


@dataclass
class MetricFilter:
    """Aggregate-level predicate (rendered as HAVING)."""

    metric: str
    operator: ComparisonOperator
    target_value: float | None = None
    target_metric: str | None = None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

@dataclass
class OrderSpec:
    """A metric ordering direction."""

    metric: str
    direction: Literal["ASC", "DESC"] = "DESC"


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

TimeResolutionPolicy = Literal[
    "EXACT_PERIOD",
    "LATEST_PERIOD",
    "PERIOD_OVER_PERIOD",
    "ALL_PERIODS",
]

TimeDeltaType = Literal["YoY", "MoM", "QoQ", "CUSTOM"]


@dataclass
class TimeConstraint:
    """A resolved or unresolved period constraint."""

    raw: str
    column: str
    resolution: TimeResolutionPolicy
    resolved_value: str | None = None


@dataclass
class TimeComparison:
    """Two-period comparison with a baseline and optional target."""

    baseline: TimeConstraint
    target: TimeConstraint | None = None
    delta_type: TimeDeltaType = "CUSTOM"


# ---------------------------------------------------------------------------
# Analytical Specification (Composable Semantic Contract)
# ---------------------------------------------------------------------------

OperationType = Literal["aggregate", "rank", "compare", "trend"]

TransformType = Literal["contribution_pct", "yoy", "mom", "rolling_avg"]


@dataclass
class AnalyticalSpec:
    """Composable analytical intent independent of an execution engine."""

    operation: OperationType = "aggregate"
    metrics: list[str] = field(default_factory=list)
    transforms: list[TransformType] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    partition_by: list[str] = field(default_factory=list)
    order_by: OrderSpec | None = None
    filters: list[Filter] = field(default_factory=list)
    metric_filters: list[MetricFilter] = field(default_factory=list)
    limit: int | None = None
    time_window: TimeConstraint | None = None
    time_comparison: TimeComparison | None = None
    defaults_applied: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-Level Result
# ---------------------------------------------------------------------------

NLUStatus = Literal["parsed", "rejected", "ambiguous"]


@dataclass
class StructuredQuery:
    """The semantic interpretation of a natural-language analytics query."""

    raw_query: str
    tokens: list[Token]
    spans: list[EntitySpan]
    spec: AnalyticalSpec
    unresolved_terms: list[str] = field(default_factory=list)
    default_applied: list[str] = field(default_factory=list)
    nlu_confidence: float = 0.0

    @property
    def intent_label(self) -> str:
        """Return a concise label derived from operation and transforms."""

        parts = [self.spec.operation] + list(self.spec.transforms)
        return "+".join(parts)


@dataclass
class NLUResult:
    """The parsed, rejected, or ambiguous outcome of NLU processing."""

    status: NLUStatus
    structured_query: StructuredQuery | None = None
    reason: str | None = None
    alternatives: list[StructuredQuery] = field(default_factory=list)
