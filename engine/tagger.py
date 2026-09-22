"""Deterministic lexical tagger for natural-language analytics queries."""
from __future__ import annotations

import re

from engine.interfaces import SemanticLayerProtocol, ValueIndexProtocol
from engine.types import (
    Conflict,
    EntitySpan,
    MATCH_QUALITY_RANK,
    TaggedQuery,
    Token,
    TokenType,
    ValueMatch,
)


MONTH_NAMES = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)

ROLE_RANK: dict[TokenType, int] = {
    "value": 5,
    "metric": 4,
    "dimension": 3,
    "time": 2,
    "number": 1,
    "operator": 0,
    "unknown": 0,
}

WORD_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"\d+")


class TaggerError(ValueError):
    """Raised when a raw query is not suitable for lexical tagging."""


class Tagger:
    """Convert raw query text into tokens, recognized spans, and conflicts."""

    MAX_SPAN_LENGTH = 4
    MAX_QUERY_LENGTH = 2_000

    def __init__(
        self,
        semantic_layer: SemanticLayerProtocol,
        value_index: ValueIndexProtocol,
    ) -> None:
        self._sl = semantic_layer
        self._vi = value_index
        self._metric_names = set(semantic_layer.get_metric_names())
        self._dimension_names = set(semantic_layer.get_dimension_names())
        self._synonyms = semantic_layer.get_synonyms()
        self._time_terms = set(semantic_layer.get_time_mappings()) | {
            "yoy",
            "mom",
            "qoq",
        }
        self._match_cache: dict[str, list[ValueMatch]] = {}

    def tag(self, raw_query: str) -> TaggedQuery:
        """Tag one bounded raw query without inferring analytical intent."""

        self._validate_query(raw_query)
        tokens = self._tokenize(raw_query)
        spans: list[EntitySpan] = []
        conflicts: list[Conflict] = []
        claimed: set[int] = set()

        for start in range(len(tokens)):
            if start in claimed:
                continue
            match_result = self._longest_match(tokens, start, claimed)
            if match_result is None:
                continue
            end, candidates = match_result
            chosen = self._resolve(candidates)
            if len(candidates) > 1:
                conflicts.append(
                    Conflict(
                        token_range=(start, end),
                        candidates=tuple(candidates),
                        chosen=chosen,
                    )
                )
            spans.append(chosen)
            claimed.update(range(start, end))

        return TaggedQuery(
            raw_query=raw_query,
            tokens=tokens,
            spans=spans,
            conflicts=conflicts,
        )

    def _validate_query(self, raw_query: str) -> None:
        """Reject non-string or unbounded query text before processing it."""

        if not isinstance(raw_query, str):
            raise TaggerError("Query must be a string.")
        if len(raw_query) > self.MAX_QUERY_LENGTH:
            raise TaggerError(
                f"Query exceeds the {self.MAX_QUERY_LENGTH}-character limit."
            )

    @staticmethod
    def _tokenize(raw_query: str) -> list[Token]:
        """Split query words while preserving source character offsets."""

        return [
            Token(text=match.group(0), start=match.start(), end=match.end())
            for match in WORD_RE.finditer(raw_query)
        ]

    def _longest_match(
        self, tokens: list[Token], start: int, claimed: set[int]
    ) -> tuple[int, list[EntitySpan]] | None:
        """Return the longest non-overlapping candidate with any semantic role."""

        max_length = min(self.MAX_SPAN_LENGTH, len(tokens) - start)
        for length in range(max_length, 0, -1):
            end = start + length
            if claimed.intersection(range(start, end)):
                continue
            matches = self._try_match(tokens[start:end])
            if matches:
                return end, matches
        return None

    def _try_match(self, candidate: list[Token]) -> list[EntitySpan]:
        """Return every valid semantic role for one contiguous token candidate."""

        text = " ".join(token.text for token in candidate)
        text_lower = text.lower()
        matches = self._value_spans(candidate, text)
        matches.extend(self._semantic_spans(candidate, text, text_lower))
        return matches

    def _value_spans(self, candidate: list[Token], text: str) -> list[EntitySpan]:
        """Convert Phase 2 value matches into lexical entity spans."""

        return [
            EntitySpan(
                tokens=tuple(candidate),
                canonical=match.canonical_value,
                role="value",
                matched_column=match.column,
                match_quality=match.match_type,
            )
            for match in self._cached_value_matches(text)
        ]

    def _semantic_spans(
        self, candidate: list[Token], text: str, text_lower: str
    ) -> list[EntitySpan]:
        """Match metrics, dimensions, time phrases, and integer literals."""

        matches: list[EntitySpan] = []
        canonical_metric = self._synonyms.get(text_lower, text_lower)
        if canonical_metric in self._metric_names:
            matches.append(self._span(candidate, canonical_metric, "metric"))

        canonical_dimension = self._resolve_dimension(text_lower)
        if canonical_dimension is not None:
            matches.append(
                self._span(
                    candidate,
                    canonical_dimension,
                    "dimension",
                    canonical_dimension,
                )
            )

        if text_lower in self._time_terms or text_lower in MONTH_NAMES:
            matches.append(self._span(candidate, text_lower, "time"))

        if NUMBER_RE.fullmatch(text):
            matches.append(self._span(candidate, text, "number"))
        return matches

    def _resolve_dimension(self, text_lower: str) -> str | None:
        """Resolve direct and business-alias dimension names deterministically."""

        singular = self._pluralize_down(text_lower)
        candidates = (
            self._synonyms.get(text_lower),
            self._synonyms.get(singular),
            text_lower,
            singular,
        )
        return next(
            (candidate for candidate in candidates if candidate in self._dimension_names),
            None,
        )

    @staticmethod
    def _span(
        candidate: list[Token], canonical: str, role: TokenType, matched_column: str | None = None
    ) -> EntitySpan:
        """Create an exact-match span for a semantic-layer recognition."""

        return EntitySpan(
            tokens=tuple(candidate),
            canonical=canonical,
            role=role,
            matched_column=matched_column,
            match_quality="exact",
        )

    def _resolve(self, matches: list[EntitySpan]) -> EntitySpan:
        """Choose a deterministic default while preserving all alternatives."""

        return max(
            matches,
            key=lambda match: (
                MATCH_QUALITY_RANK[match.match_quality],
                ROLE_RANK[match.role],
            ),
        )

    def _cached_value_matches(self, text: str) -> list[ValueMatch]:
        """Cache case-insensitive Phase 2 lookups within this tagger instance."""

        key = text.lower()
        if key not in self._match_cache:
            self._match_cache[key] = self._vi.match(text)
        return self._match_cache[key]

    @staticmethod
    def _pluralize_down(text: str) -> str:
        """Apply a limited plural-to-singular conversion for dimensions only."""

        if text.endswith("ies") and len(text) > 3:
            return text[:-3] + "y"
        if text.endswith("es") and len(text) > 2:
            return text[:-2]
        if text.endswith("s") and len(text) > 1:
            return text[:-1]
        return text
