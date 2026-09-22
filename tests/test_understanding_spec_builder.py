from pathlib import Path

import pytest

from engine.understanding.spec_builder import SpecBuilder
from engine.semantic_layer import DuckDBSemanticLayer
from engine.tagger import Tagger
from engine.types import AnalyticalSpec, EntitySpan, OrderSpec, TaggedQuery, Token
from engine.value_index import CardinalityTieredValueIndex
from tests.golden_specs import GOLDEN_SPECS


@pytest.fixture(scope="module")
def semantic_layer() -> DuckDBSemanticLayer:
    """Provide the real dataset schema for Phase 4 integration tests."""

    layer = DuckDBSemanticLayer(
        "dataset/sales_data.csv",
        "dataset/targets.csv",
        "dataset/data_dictionary.json",
    )
    yield layer
    layer.close()


@pytest.fixture(scope="module")
def tagger(semantic_layer: DuckDBSemanticLayer) -> Tagger:
    """Provide the completed Phase 3 tagger."""

    return Tagger(semantic_layer, CardinalityTieredValueIndex(semantic_layer))


@pytest.mark.parametrize("golden", GOLDEN_SPECS, ids=lambda item: item.query)
def test_rule_fallback_matches_every_golden_spec(
    semantic_layer: DuckDBSemanticLayer, tagger: Tagger, golden
) -> None:
    assert SpecBuilder(semantic_layer).build(tagger.tag(golden.query)) == golden.expected


def test_hallucinated_llm_spec_falls_back_to_rules(
    semantic_layer: DuckDBSemanticLayer, tagger: Tagger
) -> None:
    class HallucinatingBuilder:
        def build(self, tagged: TaggedQuery) -> AnalyticalSpec:
            return AnalyticalSpec(metrics=["not_a_metric"], group_by=["not_a_dimension"])

    spec = SpecBuilder(semantic_layer, HallucinatingBuilder()).build(
        tagger.tag("Top 2 cities by profit")
    )
    assert spec == next(item.expected for item in GOLDEN_SPECS if item.query == "Top 2 cities by profit")


def test_valid_llm_spec_is_used(semantic_layer: DuckDBSemanticLayer, tagger: Tagger) -> None:
    expected = next(item.expected for item in GOLDEN_SPECS if item.query == "Top 2 cities by profit")

    class ValidBuilder:
        def build(self, tagged: TaggedQuery) -> AnalyticalSpec:
            return AnalyticalSpec(
                operation="rank",
                metrics=["profit"],
                group_by=["city"],
                order_by=OrderSpec("profit"),
                limit=2,
            )

    spec = SpecBuilder(semantic_layer, ValidBuilder()).build(
        tagger.tag("Top 2 cities by profit")
    )
    assert spec is not None
    assert spec.operation == "rank"
    assert spec.metrics == ["profit"]


def test_llm_default_metric_is_marked_centrally(
    semantic_layer: DuckDBSemanticLayer, tagger: Tagger
) -> None:
    class DefaultMetricBuilder:
        def build(self, tagged: TaggedQuery) -> AnalyticalSpec:
            return AnalyticalSpec(
                operation="rank",
                metrics=["revenue"],
                group_by=["region", "product_name"],
                partition_by=["region"],
                order_by=OrderSpec("revenue"),
                limit=1,
            )

    spec = SpecBuilder(semantic_layer, DefaultMetricBuilder()).build(
        tagger.tag("Top product in each region")
    )
    assert spec is not None
    assert spec.defaults_applied == ["default_metric"]


def test_generic_partitioned_rule_uses_canonical_span_values() -> None:
    class GenericSemanticLayer:
        def get_default_metric(self) -> str:
            return "sales_amount"

        def get_target_column(self, metric_name: str) -> str | None:
            return None

        def get_time_column(self, time_canonical: str) -> str | None:
            return None

    tokens = [
        Token("Top", 0, 3),
        Token("stores", 4, 10),
        Token("per", 11, 14),
        Token("zone", 15, 19),
    ]
    tagged = TaggedQuery(
        raw_query="Top stores per zone",
        tokens=tokens,
        spans=[
            EntitySpan((tokens[1],), "store_code", "dimension", "store_code"),
            EntitySpan((tokens[3],), "zone_code", "dimension", "zone_code"),
        ],
    )
    spec = SpecBuilder(GenericSemanticLayer()).build(tagged)
    assert spec == AnalyticalSpec(
        operation="rank",
        metrics=["sales_amount"],
        group_by=["zone_code", "store_code"],
        partition_by=["zone_code"],
        order_by=OrderSpec("sales_amount"),
        limit=1,
        defaults_applied=["default_metric"],
    )


def test_rule_source_has_no_dataset_entity_literals() -> None:
    source = Path("engine/understanding/rule_builder.py").read_text().lower()
    for forbidden in ("profit", "city", "revenue", "region", "target_revenue", "avg_order_value"):
        assert f'"{forbidden}"' not in source
