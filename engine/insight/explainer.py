"""Produce grounded explanations with an optional LLM-primary prose path."""
from __future__ import annotations

import logging
import re
from typing import Protocol

from engine.insight.prompts import EXPLAIN_SYSTEM, build_explanation_prompt
from engine.types import AnalyticalSpec, TaggedQuery


LOGGER = logging.getLogger(__name__)


class ExplanationClient(Protocol):
    """Minimal contract for an optional client that rewrites structured facts as prose."""

    # TODO(provider): enforce timeout and transient-network retry policy in its adapter.
    def generate(self, *, system: str, user: str) -> str:
        """Return a proposed explanation."""


class Explainer:
    """Prefer a bounded LLM explanation and fall back to deterministic facts."""

    _MAX_LLM_WORDS = 40

    def __init__(self, llm_client: ExplanationClient | None = None) -> None:
        self._llm = llm_client

    def explain(
        self,
        tagged: TaggedQuery,
        spec: AnalyticalSpec,
        confidence: float,
        retries_used: int,
        rule_spec_present: bool,
        llm_spec_present: bool,
        specs_agree: bool | None,
        template_path_used: bool,
        result_count: int | None,
        feedback_applied: bool = False,
    ) -> str:
        """Return an LLM explanation when valid, otherwise a factual template fallback."""

        confidence = max(0.0, min(1.0, confidence))
        spec_path = self._spec_path(rule_spec_present, llm_spec_present, specs_agree)
        sql_path = "a validated SQL template" if template_path_used else "a validated SQL fallback"
        if self._llm is not None:
            llm_explanation = self._explain_via_llm(
                tagged,
                spec,
                confidence,
                retries_used,
                spec_path,
                sql_path,
                result_count,
                feedback_applied,
            )
            if llm_explanation is not None:
                return llm_explanation
        understood = self._describe_understanding(tagged, spec)
        generated = self._describe_generation(
            confidence,
            retries_used,
            spec_path,
            sql_path,
            result_count,
            feedback_applied,
        )
        return f"{understood} {generated}"

    def _explain_via_llm(
        self,
        tagged: TaggedQuery,
        spec: AnalyticalSpec,
        confidence: float,
        retries_used: int,
        spec_path: str,
        sql_path: str,
        result_count: int | None,
        feedback_applied: bool,
    ) -> str | None:
        """Generate and validate concise prose, returning ``None`` on a safe fallback."""

        try:
            raw = self._llm.generate(
                system=EXPLAIN_SYSTEM,
                user=build_explanation_prompt(
                    tagged,
                    spec,
                    confidence,
                    retries_used,
                    spec_path,
                    sql_path,
                    result_count,
                    feedback_applied,
                ),
            )
            return self._clean_llm_explanation(raw)
        except Exception as error:
            LOGGER.warning(
                "llm_explanation_failed",
                extra={"error_type": type(error).__name__},
                exc_info=error,
            )
            return None

    @classmethod
    def _clean_llm_explanation(cls, raw: str) -> str | None:
        """Accept only concise two-sentence plain text from the LLM boundary."""

        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = " ".join(lines)
        text = re.sub(r"\s+", " ", text).strip()
        sentences = [item for item in re.split(r"(?<=[.!?])\s+", text) if item]
        if (
            len(sentences) != 2
            or len(text.split()) > cls._MAX_LLM_WORDS
            or any(marker in text for marker in ("#", "*", "- "))
        ):
            return None
        return text

    def _describe_understanding(self, tagged: TaggedQuery, spec: AnalyticalSpec) -> str:
        """Describe intent exclusively from the grounded analytical specification."""

        parts = [self._operation_phrase(spec), self._metric_phrase(spec)]
        if spec.group_by:
            parts.append("grouped by " + ", ".join(spec.group_by))
        if spec.partition_by:
            parts.append("within each " + ", ".join(spec.partition_by))
        if spec.transforms:
            parts.append(self._transform_phrase(spec))
        if spec.filters:
            parts.append("filtered by " + self._filter_phrase(spec))
        if spec.limit is not None:
            parts.append(f"limited to {spec.limit}")
        if spec.defaults_applied:
            parts.append("using " + ", ".join(spec.defaults_applied).replace("_", " "))
        ambiguity = (
            f" The tagger recorded {len(tagged.conflicts)} ambiguous term(s)."
            if tagged.conflicts
            else ""
        )
        return "I understood this as " + ", ".join(part for part in parts if part) + "." + ambiguity

    @staticmethod
    def _operation_phrase(spec: AnalyticalSpec) -> str:
        """Return a plain-language operation description."""

        return {
            "aggregate": "an aggregate query",
            "rank": "a ranking query",
            "compare": "a target-comparison query",
            "trend": "a time-trend query",
        }.get(spec.operation, "an analytical query")

    @staticmethod
    def _metric_phrase(spec: AnalyticalSpec) -> str:
        """Return the requested metrics in a readable phrase."""

        return "for " + ", ".join(spec.metrics) if spec.metrics else "with no explicit metric"

    @staticmethod
    def _transform_phrase(spec: AnalyticalSpec) -> str:
        """Translate configured transforms without inventing business meaning."""

        names = {
            "contribution_pct": "shown as contribution percentages",
            "yoy": "measured year-over-year",
            "mom": "measured month-over-month",
            "rolling_avg": "shown as a rolling average",
        }
        return " and ".join(names.get(item, item) for item in spec.transforms)

    @staticmethod
    def _filter_phrase(spec: AnalyticalSpec) -> str:
        """Render row predicates already grounded by earlier phases."""

        return " and ".join(
            f"{item.column} {item.operator} {item.value!r}" for item in spec.filters
        )

    @staticmethod
    def _describe_generation(
        confidence: float,
        retries_used: int,
        spec_path: str,
        sql_path: str,
        result_count: int | None,
        feedback_applied: bool,
    ) -> str:
        """Describe only the real construction, execution, and confidence facts."""

        repair_text = "no repairs" if retries_used == 0 else f"{retries_used} repair attempt(s)"
        row_text = "no result" if result_count is None else f"{result_count} row(s)"
        feedback_text = " A verified feedback correction was applied" if feedback_applied else ""
        return (
            f"I generated the SQL with {spec_path} and {sql_path}, executed it with "
            f"{repair_text}, and returned {row_text}.{feedback_text} Confidence is {confidence:.2f}."
        )

    @staticmethod
    def _spec_path(
        rule_spec_present: bool, llm_spec_present: bool, specs_agree: bool | None
    ) -> str:
        """State provenance without claiming unverified rule/LLM agreement."""

        if rule_spec_present and llm_spec_present and specs_agree is True:
            return "matching rule and LLM specifications"
        if rule_spec_present and llm_spec_present:
            return "available rule and LLM specifications"
        if rule_spec_present:
            return "the rule-based specification"
        if llm_spec_present:
            return "the LLM specification"
        return "the available specification"
