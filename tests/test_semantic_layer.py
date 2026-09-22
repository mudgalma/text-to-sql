import json
from pathlib import Path

import pytest

from engine.semantic_layer import DuckDBSemanticLayer, SemanticLayerError


DATASET = Path("dataset")


def _write_dictionary(tmp_path: Path, contents: dict) -> Path:
    """Write a temporary dictionary and return its path."""

    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps(contents), encoding="utf-8")
    return path


def _dataset_dictionary() -> dict:
    """Return a mutable copy of the checked-in business dictionary."""

    return json.loads((DATASET / "data_dictionary.json").read_text())


def test_semantic_layer_exposes_computed_sales_view() -> None:
    layer = DuckDBSemanticLayer(
        DATASET / "sales_data.csv",
        DATASET / "targets.csv",
        DATASET / "data_dictionary.json",
    )
    try:
        result = layer.execute("SELECT COUNT(*) AS row_count, SUM(revenue) AS revenue FROM v_sales")
        assert result.loc[0, "row_count"] == 10
        assert result.loc[0, "revenue"] == pytest.approx(6134.4)
        assert "product_name" in layer.get_dimension_names()
        assert layer.get_metric_definition("revenue") == "quantity * unit_price * (1 - discount)"
        assert layer.get_default_metric() == "revenue"
        assert layer.get_target_column("revenue") == "target_revenue"
        assert layer.get_time_column("march") == "month"
        assert layer.get_time_column("last month") == "month"
        assert layer.get_time_column("yoy") is None
        assert layer.get_data_quality_report() == {
            "row_count": 10,
            "null_counts": {
                "order_id": 0,
                "order_date": 0,
                "quantity": 0,
                "unit_price": 0,
                "profit": 0,
                "revenue": 0,
            },
            "imputed_counts": {"discount": 0},
        }
    finally:
        layer.close()


def test_semantic_layer_resolves_dependent_aggregate_metrics() -> None:
    layer = DuckDBSemanticLayer(
        DATASET / "sales_data.csv",
        DATASET / "targets.csv",
        DATASET / "data_dictionary.json",
    )
    try:
        formula = layer.get_metric_sql("avg_order_value")
        assert formula is not None
        assert "orders" not in formula
        result = layer.execute(
            f"SELECT region, {formula} AS avg_order_value "
            "FROM v_sales GROUP BY region ORDER BY region"
        )
        assert result.loc[result["region"] == "APAC", "avg_order_value"].iloc[0] == pytest.approx(118.5)
        assert result.loc[result["region"] == "EMEA", "avg_order_value"].iloc[0] == pytest.approx(799.3333333333)
    finally:
        layer.close()


def test_semantic_layer_rejects_missing_source_file(tmp_path: Path) -> None:
    with pytest.raises(SemanticLayerError, match="missing or empty"):
        DuckDBSemanticLayer(
            tmp_path / "missing.csv",
            DATASET / "targets.csv",
            DATASET / "data_dictionary.json",
        )


def test_semantic_layer_rejects_invalid_json(tmp_path: Path) -> None:
    invalid_dictionary = tmp_path / "dictionary.json"
    invalid_dictionary.write_text("{invalid", encoding="utf-8")
    with pytest.raises(SemanticLayerError, match="not valid JSON"):
        DuckDBSemanticLayer(
            DATASET / "sales_data.csv",
            DATASET / "targets.csv",
            invalid_dictionary,
        )


def test_semantic_layer_rejects_missing_phase4_schema_config(tmp_path: Path) -> None:
    dictionary = _dataset_dictionary()
    dictionary.pop("default_metric")
    with pytest.raises(SemanticLayerError, match="missing keys: default_metric"):
        DuckDBSemanticLayer(
            DATASET / "sales_data.csv",
            DATASET / "targets.csv",
            _write_dictionary(tmp_path, dictionary),
        )


def test_semantic_layer_rejects_unsafe_formula(tmp_path: Path) -> None:
    dictionary = _dataset_dictionary()
    dictionary["metrics"]["profit"] = "profit); ATTACH '/tmp/x.db' AS x; --"
    with pytest.raises(SemanticLayerError, match="unsupported SQL syntax"):
        DuckDBSemanticLayer(
            DATASET / "sales_data.csv",
            DATASET / "targets.csv",
            _write_dictionary(tmp_path, dictionary),
        )


def test_semantic_layer_rejects_metric_dependency_cycle(tmp_path: Path) -> None:
    dictionary = _dataset_dictionary()
    dictionary["metrics"]["first"] = "second"
    dictionary["metrics"]["second"] = "first"
    with pytest.raises(SemanticLayerError, match="dependency cycle"):
        DuckDBSemanticLayer(
            DATASET / "sales_data.csv",
            DATASET / "targets.csv",
            _write_dictionary(tmp_path, dictionary),
        )


def test_semantic_layer_rejects_renamed_required_column(tmp_path: Path) -> None:
    sales_csv = tmp_path / "renamed_sales.csv"
    contents = (DATASET / "sales_data.csv").read_text()
    sales_csv.write_text(contents.replace("discount", "discount_pct", 1), encoding="utf-8")
    with pytest.raises(SemanticLayerError, match="missing required columns: discount"):
        DuckDBSemanticLayer(
            sales_csv,
            DATASET / "targets.csv",
            DATASET / "data_dictionary.json",
        )


def test_blank_discount_defaults_to_zero_and_is_reported(tmp_path: Path) -> None:
    sales_csv = tmp_path / "blank_discount.csv"
    contents = (DATASET / "sales_data.csv").read_text()
    sales_csv.write_text(contents.replace("2,120,0.10", "2,120,", 1), encoding="utf-8")
    layer = DuckDBSemanticLayer(
        sales_csv,
        DATASET / "targets.csv",
        DATASET / "data_dictionary.json",
    )
    try:
        row = layer.execute("SELECT discount, revenue FROM v_sales WHERE order_id = '1001'").iloc[0]
        assert row["discount"] == 0.0
        assert row["revenue"] == pytest.approx(240.0)
        assert layer.get_data_quality_report()["imputed_counts"]["discount"] == 1
    finally:
        layer.close()


def test_semantic_layer_rejects_missing_required_measure(tmp_path: Path) -> None:
    sales_csv = tmp_path / "missing_price.csv"
    contents = (DATASET / "sales_data.csv").read_text()
    sales_csv.write_text(contents.replace("2,120,0.10", "2,,0.10", 1), encoding="utf-8")
    with pytest.raises(SemanticLayerError, match="unit_price=1"):
        DuckDBSemanticLayer(
            sales_csv,
            DATASET / "targets.csv",
            DATASET / "data_dictionary.json",
        )
