"""Verify that the local dataset and Phase 2 components are runnable."""
from __future__ import annotations

from pathlib import Path

from engine.semantic_layer import DuckDBSemanticLayer
from engine.value_index import CardinalityTieredValueIndex


def main() -> None:
    """Load the sample dataset and report essential semantic-layer checks."""

    root = Path(__file__).resolve().parent.parent
    dataset = root / "dataset"
    required_paths = [
        dataset / "sales_data.csv",
        dataset / "targets.csv",
        dataset / "data_dictionary.json",
        dataset / "nl_queries.json",
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required dataset file: {path}")

    layer = DuckDBSemanticLayer(
        sales_csv=dataset / "sales_data.csv",
        targets_csv=dataset / "targets.csv",
        dict_json=dataset / "data_dictionary.json",
    )
    try:
        index = CardinalityTieredValueIndex(layer)
        total = layer.execute("SELECT SUM(revenue) AS revenue FROM v_sales").iloc[0, 0]
        print(f"metrics: {layer.get_metric_names()}")
        print(f"dimensions: {layer.get_dimension_names()}")
        print(f"sum(revenue): {total}")
        print(f"indexed columns: {index.indexed_columns()}")
        print(f"India matches: {index.match('India')}")
    finally:
        layer.close()


if __name__ == "__main__":
    main()
