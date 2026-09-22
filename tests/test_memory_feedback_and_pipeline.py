"""Tests for optional feedback handling and the complete assignment pipeline."""
from __future__ import annotations

import json

import pytest

from engine.memory.feedback_store import FeedbackStore, FeedbackStoreError
from main import run_pipeline


def test_missing_feedback_is_supported() -> None:
    """No feedback file is a normal assignment input, not an error."""

    store = FeedbackStore("dataset/not-provided.csv")
    assert store.retrieve_similar("sales by country") == []
    assert store.get_exact_correction("sales by country") is None


def test_feedback_retrieval_and_exact_correction(tmp_path) -> None:
    """Only an exact query may automatically reuse verified corrected SQL."""

    path = tmp_path / "feedback.csv"
    path.write_text(
        "query,corrected_sql\nSales by region,SELECT 1 AS value\nSales by country,SELECT 2 AS value\n",
        encoding="utf-8",
    )
    store = FeedbackStore(path)
    assert store.get_exact_correction("sales BY region") == "SELECT 1 AS value"
    assert store.get_exact_correction("sales by city") is None
    assert [entry.query for entry in store.retrieve_similar("sales by region")] == [
        "Sales by region",
        "Sales by country",
    ]


def test_malformed_feedback_is_rejected(tmp_path) -> None:
    """Unexpected feedback schemas do not silently become model guidance."""

    path = tmp_path / "feedback.csv"
    path.write_text("query,answer\nSales,42\n", encoding="utf-8")
    with pytest.raises(FeedbackStoreError, match="query and corrected_sql"):
        FeedbackStore(path)


def test_full_pipeline_returns_required_output_for_assignment_queries(tmp_path) -> None:
    """All eight provided queries run from language through JSON-ready results."""

    outputs = run_pipeline("dataset", tmp_path / "missing-feedback.csv")
    expected_queries = json.loads((__import__("pathlib").Path("dataset/nl_queries.json")).read_text())
    assert [output["query"] for output in outputs] == [item["query"] for item in expected_queries]
    for output in outputs:
        assert set(output) == {
            "query", "generated_logic", "result", "confidence_score", "explanation"
        }
        assert output["generated_logic"] is not None
        assert output["result"] is not None
        assert 0.0 <= output["confidence_score"] <= 1.0
        assert "I understood" in output["explanation"]


def test_pipeline_uses_exact_verified_feedback(tmp_path) -> None:
    """Feedback can improve a matching future run without hardcoded query answers."""

    feedback = tmp_path / "feedback.csv"
    feedback.write_text(
        "query,corrected_sql\nTotal sales in India for March,SELECT 123 AS value\n",
        encoding="utf-8",
    )
    outputs = run_pipeline("dataset", feedback)
    first = outputs[0]
    assert first["generated_logic"] == "SELECT 123 AS value"
    assert first["result"] == [{"value": 123}]
    assert "verified feedback correction" in first["explanation"]
