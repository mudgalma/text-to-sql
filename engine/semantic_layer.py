"""DuckDB-backed business schema and validated metric definitions."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


LOGGER = logging.getLogger(__name__)


class SemanticLayerError(ValueError):
    """Raised when semantic-layer configuration or source data is invalid."""


class DuckDBSemanticLayer:
    """Expose a validated `v_sales` view and business metadata over CSV data."""

    VIEW_NAME = "v_sales"
    _REQUIRED_DICTIONARY_KEYS = frozenset(
        {
            "metrics",
            "dimensions",
            "synonyms",
            "time_mappings",
            "targets",
            "default_metric",
            "time_column_mapping",
        }
    )
    _RAW_SALES_COLUMNS = frozenset(
        {
            "order_id",
            "order_date",
            "region",
            "country",
            "city",
            "customer_id",
            "customer_segment",
            "product_category",
            "product_subcategory",
            "product_name",
            "quantity",
            "unit_price",
            "discount",
            "shipping_cost",
            "profit",
        }
    )
    _REQUIRED_NON_NULL_COLUMNS = frozenset(
        {"order_id", "order_date", "quantity", "unit_price", "profit", "revenue"}
    )
    _SAFE_FORMULA = re.compile(r"^[A-Za-z0-9_() *+\-/.]+$")
    _IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    _ALLOWED_FUNCTIONS = frozenset({"avg", "coalesce", "count", "max", "min", "sum"})
    _MONTH_TERMS = frozenset(
        {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
            "sept", "oct", "nov", "dec",
        }
    )

    def __init__(
        self,
        sales_csv: str | Path,
        targets_csv: str | Path,
        dict_json: str | Path,
    ) -> None:
        self._sales_csv = self._validate_file(sales_csv, ".csv")
        self._targets_csv = self._validate_file(targets_csv, ".csv")
        self._dict = self._load_dictionary(dict_json)
        self.conn = duckdb.connect(":memory:")

        try:
            self._register_tables()
            self._raw_schema = self._introspect_table("raw_sales")
            self._validate_raw_schema()
            self._metric_sql = self._compile_metric_sql()
            self._register_view()
            self._schema_cache = self._introspect_table(self.VIEW_NAME)
            self._validate_dimensions()
            self._validate_compiled_metrics()
            self._data_quality_cache = self._build_data_quality_report()
            self._enforce_data_quality()
        except SemanticLayerError:
            self.conn.close()
            raise
        except duckdb.Error as error:
            self.conn.close()
            raise SemanticLayerError("Could not initialize the analytics view.") from error

        LOGGER.info(
            "semantic_layer_initialized",
            extra={
                "view": self.VIEW_NAME,
                "columns": len(self._schema_cache),
                "rows": self._data_quality_cache["row_count"],
            },
        )

    def _validate_file(self, path_value: str | Path, suffix: str) -> Path:
        """Return an existing, non-empty source file with the expected suffix."""

        path = Path(path_value).expanduser().resolve()
        if path.suffix.lower() != suffix:
            raise SemanticLayerError(f"Expected a {suffix} file: {path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise SemanticLayerError(f"Source file is missing or empty: {path}")
        return path

    def _load_dictionary(self, dict_json: str | Path) -> dict[str, Any]:
        """Load and validate the business dictionary from JSON."""

        path = self._validate_file(dict_json, ".json")
        try:
            with path.open(encoding="utf-8-sig") as source:
                contents: dict[str, Any] = json.load(source)
        except json.JSONDecodeError as error:
            raise SemanticLayerError("Data dictionary is not valid JSON.") from error

        if not isinstance(contents, dict):
            raise SemanticLayerError("Data dictionary must contain a JSON object.")
        missing = self._REQUIRED_DICTIONARY_KEYS.difference(contents)
        if missing:
            names = ", ".join(sorted(missing))
            raise SemanticLayerError(f"Data dictionary is missing keys: {names}")
        self._validate_dictionary_types(contents)
        return contents

    def _validate_dictionary_types(self, contents: dict[str, Any]) -> None:
        """Ensure dictionary sections have string keys and values where required."""

        metrics = contents["metrics"]
        if not isinstance(metrics, dict) or not metrics:
            raise SemanticLayerError("Data dictionary metrics must be a non-empty object.")
        if not all(
            isinstance(name, str) and isinstance(formula, str)
            for name, formula in metrics.items()
        ):
            raise SemanticLayerError("Metric names and formulas must be strings.")
        if not isinstance(contents["dimensions"], list):
            raise SemanticLayerError("Data dictionary dimensions must be a list.")
        for section in ("synonyms", "time_mappings"):
            values = contents[section]
            if not isinstance(values, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in values.items()
            ):
                raise SemanticLayerError(
                    f"Data dictionary {section} must map strings to strings."
                )
        targets = contents["targets"]
        if not isinstance(targets, dict) or not all(
            isinstance(metric, str) and isinstance(column, str)
            for metric, column in targets.items()
        ):
            raise SemanticLayerError("Data dictionary targets must map strings to strings.")
        default_metric = contents["default_metric"]
        if not isinstance(default_metric, str) or default_metric not in metrics:
            raise SemanticLayerError("Data dictionary default_metric must name a metric.")
        time_columns = contents["time_column_mapping"]
        if not isinstance(time_columns, dict) or not all(
            isinstance(term, str) and (column is None or isinstance(column, str))
            for term, column in time_columns.items()
        ):
            raise SemanticLayerError(
                "Data dictionary time_column_mapping must map strings to strings or null."
            )

    def _register_tables(self) -> None:
        """Register validated CSV files as all-string DuckDB tables."""

        self._create_table_from_csv("raw_sales", self._sales_csv)
        self._create_table_from_csv("targets", self._targets_csv)

    def _create_table_from_csv(self, table_name: str, source: Path) -> None:
        """Create a table from a local validated CSV path without type inference."""

        escaped_path = str(source).replace("'", "''")
        self.conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto('{escaped_path}', header = true, all_varchar = true)"
        )

    def _introspect_table(self, table_name: str) -> dict[str, str]:
        """Return column names and DuckDB types for a trusted internal relation."""

        rows = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        return {name: data_type for name, data_type, *_ in rows}

    def _validate_raw_schema(self) -> None:
        """Fail clearly when a required source column is missing or renamed.

        The explicit raw-column contract prevents schema drift from becoming a
        later SQL binding failure or a silently incorrect result.
        """

        missing = self._RAW_SALES_COLUMNS.difference(self._raw_schema)
        if missing:
            names = ", ".join(sorted(missing))
            raise SemanticLayerError(f"Sales CSV is missing required columns: {names}")

    def _compile_metric_sql(self) -> dict[str, str]:
        """Resolve metric references recursively into safe executable SQL."""

        metrics: dict[str, str] = self._dict["metrics"]
        for name, formula in metrics.items():
            self._validate_formula(name, formula, metrics)

        resolved: dict[str, str] = {}
        resolving: list[str] = []

        def resolve(name: str) -> str:
            if name in resolved:
                return resolved[name]
            if name in resolving:
                cycle = " -> ".join([*resolving, name])
                raise SemanticLayerError(f"Metric dependency cycle detected: {cycle}")

            resolving.append(name)
            formula = metrics[name]

            def replace_metric(match: re.Match[str]) -> str:
                identifier = match.group(0)
                if identifier.lower() in self._ALLOWED_FUNCTIONS:
                    return identifier
                if identifier in self._raw_schema:
                    return identifier
                return f"({resolve(identifier)})"

            resolved_formula = self._IDENTIFIER.sub(replace_metric, formula)
            resolving.pop()
            resolved[name] = resolved_formula
            return resolved_formula

        return {name: resolve(name) for name in metrics}

    def _validate_formula(
        self, name: str, formula: str, metrics: dict[str, str]
    ) -> None:
        """Reject unsafe formulas and references absent from schema and metrics."""

        if not formula.strip() or not self._SAFE_FORMULA.fullmatch(formula):
            raise SemanticLayerError(f"Metric {name!r} contains unsupported SQL syntax.")
        identifiers = self._IDENTIFIER.findall(formula)
        unknown = {
            identifier
            for identifier in identifiers
            if identifier.lower() not in self._ALLOWED_FUNCTIONS
            and identifier not in self._raw_schema
            and identifier not in metrics
        }
        if unknown:
            names = ", ".join(sorted(unknown))
            raise SemanticLayerError(f"Metric {name!r} references unknown names: {names}")

    def _register_view(self) -> None:
        """Create the canonical typed view used by downstream pipeline stages."""

        revenue_formula = self._metric_sql.get("revenue")
        if revenue_formula is None:
            raise SemanticLayerError("Data dictionary requires a revenue metric.")

        self.conn.execute(
            f"""
            CREATE OR REPLACE VIEW {self.VIEW_NAME} AS
            WITH typed_sales AS (
                SELECT
                    NULLIF(TRIM(order_id), '') AS order_id,
                    CAST(NULLIF(TRIM(order_date), '') AS DATE) AS order_date,
                    NULLIF(TRIM(region), '') AS region,
                    NULLIF(TRIM(country), '') AS country,
                    NULLIF(TRIM(city), '') AS city,
                    NULLIF(TRIM(customer_id), '') AS customer_id,
                    NULLIF(TRIM(customer_segment), '') AS customer_segment,
                    NULLIF(TRIM(product_category), '') AS product_category,
                    NULLIF(TRIM(product_subcategory), '') AS product_subcategory,
                    NULLIF(TRIM(product_name), '') AS product_name,
                    CAST(NULLIF(TRIM(quantity), '') AS DOUBLE) AS quantity,
                    CAST(NULLIF(TRIM(unit_price), '') AS DOUBLE) AS unit_price,
                    CAST(NULLIF(TRIM(discount), '') AS DOUBLE) AS raw_discount,
                    CAST(NULLIF(TRIM(shipping_cost), '') AS DOUBLE) AS shipping_cost,
                    CAST(NULLIF(TRIM(profit), '') AS DOUBLE) AS profit
                FROM raw_sales
            ),
            normalized_sales AS (
                SELECT *, COALESCE(raw_discount, 0.0) AS discount
                FROM typed_sales
            )
            SELECT
                order_id,
                order_date,
                STRFTIME(order_date, '%Y-%m') AS month,
                EXTRACT(YEAR FROM order_date) AS year,
                region,
                country,
                city,
                customer_id,
                customer_segment,
                product_category,
                product_subcategory,
                product_name,
                quantity,
                unit_price,
                discount,
                shipping_cost,
                profit,
                ({revenue_formula}) AS revenue
            FROM normalized_sales
            """
        )

    def _validate_dimensions(self) -> None:
        """Ensure each configured dimension exists in the canonical view."""

        dimensions = self._dict["dimensions"]
        if not all(isinstance(dimension, str) for dimension in dimensions):
            raise SemanticLayerError("Data dictionary dimensions must contain strings.")
        unknown = set(dimensions).difference(self._schema_cache)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise SemanticLayerError(
                f"Configured dimensions are absent from v_sales: {names}"
            )

    def _validate_compiled_metrics(self) -> None:
        """Prove every compiled metric binds against the canonical view."""

        for name, formula in self._metric_sql.items():
            try:
                self.conn.execute(
                    f"EXPLAIN SELECT {formula} AS metric_value FROM {self.VIEW_NAME}"
                )
            except duckdb.Error as error:
                raise SemanticLayerError(
                    f"Metric {name!r} cannot execute against {self.VIEW_NAME}."
                ) from error

    def _build_data_quality_report(self) -> dict[str, Any]:
        """Measure nulls and discount imputations used by downstream confidence."""

        row = self.conn.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(*) FILTER (WHERE order_id IS NULL) AS null_order_id,
                COUNT(*) FILTER (WHERE order_date IS NULL) AS null_order_date,
                COUNT(*) FILTER (WHERE quantity IS NULL) AS null_quantity,
                COUNT(*) FILTER (WHERE unit_price IS NULL) AS null_unit_price,
                COUNT(*) FILTER (WHERE profit IS NULL) AS null_profit,
                COUNT(*) FILTER (WHERE revenue IS NULL) AS null_revenue
            FROM {self.VIEW_NAME}
            """
        ).fetchone()
        discount_imputations = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM raw_sales
            WHERE NULLIF(TRIM(discount), '') IS NULL
            """
        ).fetchone()[0]
        null_counts = {
            "order_id": int(row[1]),
            "order_date": int(row[2]),
            "quantity": int(row[3]),
            "unit_price": int(row[4]),
            "profit": int(row[5]),
            "revenue": int(row[6]),
        }
        return {
            "row_count": int(row[0]),
            "null_counts": null_counts,
            "imputed_counts": {"discount": int(discount_imputations)},
        }

    def _enforce_data_quality(self) -> None:
        """Reject rows missing required analytical fields after normalization."""

        null_counts: dict[str, int] = self._data_quality_cache["null_counts"]
        invalid = {
            name: null_counts[name]
            for name in self._REQUIRED_NON_NULL_COLUMNS
            if null_counts[name] > 0
        }
        if invalid:
            details = ", ".join(
                f"{name}={count}" for name, count in sorted(invalid.items())
            )
            raise SemanticLayerError(f"Required analytical values are null: {details}")

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute trusted internally generated SQL against the semantic database."""

        try:
            return self.conn.execute(sql).df()
        except duckdb.Error as error:
            raise SemanticLayerError("Analytics query execution failed.") from error

    def get_metric_names(self) -> list[str]:
        """Return the configured metric names."""

        return list(self._dict["metrics"].keys())

    def get_metric_definition(self, name: str) -> str | None:
        """Return the original business definition for explanation purposes."""

        definition = self._dict["metrics"].get(name)
        return definition if isinstance(definition, str) else None

    def get_metric_sql(self, name: str) -> str | None:
        """Return a fully resolved and validated SQL expression for a metric."""

        return self._metric_sql.get(name)

    def get_dimension_names(self) -> list[str]:
        """Return configured dimensions available to the NLU value index."""

        dimensions = self._dict["dimensions"]
        return [dimension for dimension in dimensions if isinstance(dimension, str)]

    def get_synonyms(self) -> dict[str, str]:
        """Return a defensive copy of metric synonyms."""

        return dict(self._dict["synonyms"])

    def get_time_mappings(self) -> dict[str, str]:
        """Return a defensive copy of natural-language time mappings."""

        return dict(self._dict["time_mappings"])

    def get_target_column(self, metric_name: str) -> str | None:
        """Return the configured target column for a metric, if available."""

        return self._dict["targets"].get(metric_name)

    def get_default_metric(self) -> str:
        """Return the configured metric used when a query omits one."""

        return self._dict["default_metric"]

    def get_time_column(self, time_canonical: str) -> str | None:
        """Return the analytical time column for a tagged temporal expression."""

        mapping: dict[str, str | None] = self._dict["time_column_mapping"]
        if time_canonical in self._MONTH_TERMS:
            return mapping.get("month_names")
        return mapping.get(time_canonical, mapping.get(time_canonical.replace(" ", "_")))

    def get_value_aliases(self) -> dict[str, dict[str, str]]:
        """Return optional value aliases; this dataset currently defines none."""

        return {}

    def get_view_schema(self) -> dict[str, str]:
        """Return a defensive copy of the canonical view schema."""

        return dict(self._schema_cache)

    def get_data_quality_report(self) -> dict[str, Any]:
        """Return null and imputation counts from canonical-view construction."""

        return {
            "row_count": self._data_quality_cache["row_count"],
            "null_counts": dict(self._data_quality_cache["null_counts"]),
            "imputed_counts": dict(self._data_quality_cache["imputed_counts"]),
        }

    def close(self) -> None:
        """Release the in-memory DuckDB connection."""

        self.conn.close()
