"""Tests for the isolated, read-only SQL execution boundary."""
from __future__ import annotations

import pytest

from engine.phase6.executor import Executor
from engine.semantic_layer import DuckDBSemanticLayer


@pytest.fixture(scope="module")
def semantic_layer() -> DuckDBSemanticLayer:
    """Provide a real in-memory semantic database for executor tests."""

    layer = DuckDBSemanticLayer(
        sales_csv="dataset/sales_data.csv",
        targets_csv="dataset/targets.csv",
        dict_json="dataset/data_dictionary.json",
    )
    yield layer
    layer.close()


@pytest.fixture
def executor(semantic_layer: DuckDBSemanticLayer) -> Executor:
    """Provide a standalone executor."""

    return Executor(semantic_layer)


def test_successful_query_returns_dataframe(executor: Executor) -> None:
    """Valid SQL returns data without an error."""

    result = executor.run("SELECT SUM(revenue) AS total FROM v_sales")
    assert result.success is True
    assert result.df is not None
    assert "total" in result.df.columns
    assert result.error is None


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT nonexistent_column FROM v_sales",
        "SELECT FROM WHERE",
    ),
)
def test_database_failures_are_captured(executor: Executor, sql: str) -> None:
    """Binding and syntax failures become structured failures, not raised errors."""

    result = executor.run(sql)
    assert result.success is False
    assert result.df is None
    assert result.error is not None
    assert "Error" in result.error


def test_empty_result_is_still_a_success(executor: Executor) -> None:
    """A valid query with no matching rows has a successful empty dataframe."""

    result = executor.run("SELECT * FROM v_sales WHERE country = 'Nonexistent'")
    assert result.success is True
    assert result.df is not None
    assert result.df.empty


def test_mutating_sql_is_rejected_before_database_execution(executor: Executor) -> None:
    """The executor accepts only the read-only query subset used by this pipeline."""

    result = executor.run("DROP TABLE targets")
    assert result.success is False
    assert result.error is not None
    assert "SELECT or WITH" in result.error
