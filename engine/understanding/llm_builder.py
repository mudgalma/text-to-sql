"""Optional Anthropic-backed structured specification builder."""
from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from engine.interfaces import SemanticLayerProtocol
from engine.understanding.prompts import build_system_prompt, build_user_prompt
from engine.types import (
    AnalyticalSpec,
    Filter,
    MetricFilter,
    OrderSpec,
    TaggedQuery,
    TimeComparison,
    TimeConstraint,
)


LOGGER = logging.getLogger(__name__)


class FilterOutput(BaseModel):
    """Structured filter returned by the optional LLM path."""

    column: str
    operator: str
    value: Any


class MetricFilterOutput(BaseModel):
    """Structured aggregate filter returned by the optional LLM path."""

    metric: str
    operator: str
    target_value: float | None = None
    target_metric: str | None = None


class AnalyticalSpecOutput(BaseModel):
    """Strict structured model accepted from the optional LLM path."""

    operation: Literal["aggregate", "rank", "compare", "trend"]
    metrics: list[str]
    group_by: list[str] = Field(default_factory=list)
    partition_by: list[str] = Field(default_factory=list)
    limit: int | None = None
    transforms: list[Literal["contribution_pct", "yoy", "mom", "rolling_avg"]] = Field(default_factory=list)
    filters: list[FilterOutput] = Field(default_factory=list)
    metric_filters: list[MetricFilterOutput] = Field(default_factory=list)
    order_by: dict[str, str] | None = None
    time_window: dict[str, str] | None = None
    time_comparison: dict[str, Any] | None = None
    defaults_applied: list[str] = Field(default_factory=list)


class LLMSpecBuilder:
    """Build a spec through an injected Anthropic client, with one tool fallback."""

    MODEL = "anthropic/claude-3-haiku"

    def __init__(self, client: Any, semantic_layer: SemanticLayerProtocol) -> None:
        self._client = client
        self._sl = semantic_layer

    def build(self, tagged: TaggedQuery) -> AnalyticalSpec | None:
        """Try structured parsing, then one tool-use fallback on any client failure."""

        try:
            response = self._client.beta.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": build_system_prompt(self._sl)},
                    {"role": "user", "content": build_user_prompt(tagged)}
                ],
                response_format=AnalyticalSpecOutput,
            )
            return self._to_spec(response.choices[0].message.parsed)
        except Exception:
            LOGGER.warning("llm_structured_build_failed: falling back to tool use.")
            return self._build_via_tool_use(tagged)

    def _build_via_tool_use(self, tagged: TaggedQuery) -> AnalyticalSpec | None:
        """Use an explicit tool schema once when structured parsing is unavailable."""

        try:
            response = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": build_system_prompt(self._sl)},
                    {"role": "user", "content": build_user_prompt(tagged)}
                ],
                tools=[self._tool_definition()],
                tool_choice={"type": "function", "function": {"name": "build_analytical_spec"}},
            )
            message = response.choices[0].message
            if message.tool_calls:
                import json
                return self._to_spec(json.loads(message.tool_calls[0].function.arguments))
            return None
        except Exception:
            LOGGER.warning("llm_tool_build_failed: all fallbacks failed.")
            return None

    @staticmethod
    def _tool_definition() -> dict[str, Any]:
        """Return the tool schema for providers without parse support."""

        return {
            "type": "function",
            "function": {
                "name": "build_analytical_spec",
                "description": "Return one grounded analytical specification.",
                "parameters": AnalyticalSpecOutput.model_json_schema(),
            }
        }

    @staticmethod
    def _to_spec(output: AnalyticalSpecOutput | dict[str, Any]) -> AnalyticalSpec | None:
        """Validate provider data and convert it to domain data contracts."""

        try:
            parsed = (
                output
                if isinstance(output, AnalyticalSpecOutput)
                else AnalyticalSpecOutput.model_validate(output)
            )
            order_by = OrderSpec(**parsed.order_by) if parsed.order_by else None
            time_window = TimeConstraint(**parsed.time_window) if parsed.time_window else None
            comparison = (
                TimeComparison(
                    baseline=TimeConstraint(**parsed.time_comparison["baseline"]),
                    target=(
                        TimeConstraint(**parsed.time_comparison["target"])
                        if parsed.time_comparison.get("target")
                        else None
                    ),
                    delta_type=parsed.time_comparison.get("delta_type", "CUSTOM"),
                )
                if parsed.time_comparison
                else None
            )
            return AnalyticalSpec(
                operation=parsed.operation,
                metrics=parsed.metrics,
                group_by=parsed.group_by,
                partition_by=parsed.partition_by,
                order_by=order_by,
                filters=[Filter(**item.model_dump()) for item in parsed.filters],
                metric_filters=[MetricFilter(**item.model_dump()) for item in parsed.metric_filters],
                limit=parsed.limit,
                transforms=parsed.transforms,
                time_window=time_window,
                time_comparison=comparison,
                defaults_applied=parsed.defaults_applied,
            )
        except (KeyError, TypeError, ValidationError, ValueError):
            return None
