"""Tests for bounded SQL repair using offline, injectable fake clients."""
from __future__ import annotations

import pytest

from engine.execution.executor import Executor
from engine.correction.repairer import SelfCorrector
from engine.semantic_layer import DuckDBSemanticLayer
from engine.types import AnalyticalSpec


@pytest.fixture(scope="module")
def semantic_layer() -> DuckDBSemanticLayer:
    """Provide the real schema needed to ground repair prompts."""

    layer = DuckDBSemanticLayer(
        sales_csv="dataset/sales_data.csv",
        targets_csv="dataset/targets.csv",
        dict_json="dataset/data_dictionary.json",
    )
    yield layer
    layer.close()


@pytest.fixture
def executor(semantic_layer: DuckDBSemanticLayer) -> Executor:
    """Provide a fresh execution boundary for each correction test."""

    return Executor(semantic_layer)


class MockLLM:
    """Record repair requests and return one configured offline response."""

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.call_count = 0
        self.last_call: dict[str, str] | None = None

    def generate(self, *, system: str, user: str) -> str:
        """Return a fake response or simulate an unavailable provider."""

        self.call_count += 1
        self.last_call = {"system": system, "user": user}
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _spec() -> AnalyticalSpec:
    """Return the simple aggregate intent used across repair scenarios."""

    return AnalyticalSpec(operation="aggregate", metrics=["revenue"])


def test_successful_first_attempt_skips_repair(
    executor: Executor, semantic_layer: DuckDBSemanticLayer
) -> None:
    """A successful execution neither calls the LLM nor consumes a retry."""

    client = MockLLM("SELECT 1")
    result = SelfCorrector(executor, semantic_layer, client).execute(
        "SELECT SUM(revenue) AS value FROM v_sales", _spec()
    )
    assert result.final.success is True
    assert result.retries_used == 0
    assert client.call_count == 0
    assert len(result.attempts) == 1


def test_failed_sql_is_repaired_once(
    executor: Executor, semantic_layer: DuckDBSemanticLayer
) -> None:
    """A read-only SQL failure can be repaired and re-executed once."""

    client = MockLLM("SELECT SUM(revenue) AS value FROM v_sales")
    result = SelfCorrector(executor, semantic_layer, client).execute(
        "SELECT wrong_column FROM v_sales", _spec()
    )
    assert result.final.success is True
    assert result.retries_used == 1
    assert client.call_count == 1
    assert len(result.attempts) == 2


def test_retries_are_bounded_and_return_final_failure(
    executor: Executor, semantic_layer: DuckDBSemanticLayer
) -> None:
    """Two unsuccessful repairs are the maximum permitted cost and work."""

    client = MockLLM("SELECT still_wrong FROM v_sales")
    result = SelfCorrector(executor, semantic_layer, client, max_retries=2).execute(
        "SELECT wrong_column FROM v_sales", _spec()
    )
    assert result.final.success is False
    assert result.retries_used == 2
    assert client.call_count == 2
    assert len(result.attempts) == 3


def test_no_llm_returns_initial_execution_failure(
    executor: Executor, semantic_layer: DuckDBSemanticLayer
) -> None:
    """Without a repair client, the original execution failure is retained."""

    result = SelfCorrector(executor, semantic_layer).execute(
        "SELECT wrong_column FROM v_sales", _spec()
    )
    assert result.final.success is False
    assert result.retries_used == 0
    assert len(result.attempts) == 1


def test_repair_prompt_is_schema_grounded_and_contains_failure_context(
    executor: Executor, semantic_layer: DuckDBSemanticLayer
) -> None:
    """The LLM receives the safe schema plus the failed query and error as data."""

    client = MockLLM("SELECT SUM(revenue) FROM v_sales")
    SelfCorrector(executor, semantic_layer, client).execute(
        "SELECT bad_col FROM v_sales", _spec()
    )
    assert client.last_call is not None
    assert "v_sales" in client.last_call["user"]
    assert "bad_col" in client.last_call["user"]
    assert "<repair_payload_json>" in client.last_call["user"]


def test_fenced_repair_is_cleaned_before_execution(
    executor: Executor, semantic_layer: DuckDBSemanticLayer
) -> None:
    """Markdown formatting does not prevent an otherwise valid repair from running."""

    client = MockLLM("```sql\nSELECT SUM(revenue) FROM v_sales\n```")
    result = SelfCorrector(executor, semantic_layer, client).execute(
        "SELECT wrong_column FROM v_sales", _spec()
    )
    assert result.final.success is True


def test_provider_failure_is_returned_in_repair_history(
    executor: Executor, semantic_layer: DuckDBSemanticLayer
) -> None:
    """An unavailable LLM cannot escape the correction boundary as an exception."""

    client = MockLLM(TimeoutError("provider timed out"))
    result = SelfCorrector(executor, semantic_layer, client, max_retries=1).execute(
        "SELECT wrong_column FROM v_sales", _spec()
    )
    assert result.final.success is False
    assert result.retries_used == 1
    assert len(result.attempts) == 1
    assert result.repair_errors == [
        "SQL repair was unavailable: TimeoutError: provider timed out"
    ]


def test_unsafe_repair_is_rejected_before_execution(
    executor: Executor, semantic_layer: DuckDBSemanticLayer
) -> None:
    """A model cannot turn a failed read-only query into a mutating statement."""

    client = MockLLM("DROP TABLE targets")
    result = SelfCorrector(executor, semantic_layer, client, max_retries=1).execute(
        "SELECT wrong_column FROM v_sales", _spec()
    )
    assert result.final.success is False
    assert len(result.attempts) == 1
    assert result.repair_errors
    assert "SELECT or WITH" in result.repair_errors[0]
