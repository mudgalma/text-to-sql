from typing import Any

import pytest

from engine.semantic_layer import DuckDBSemanticLayer
from engine.types import TIER_RANK
from engine.value_index import CardinalityTieredValueIndex


@pytest.fixture(scope="module")
def semantic_layer() -> DuckDBSemanticLayer:
    """Create one semantic layer over the checked-in sample dataset."""

    layer = DuckDBSemanticLayer(
        sales_csv="dataset/sales_data.csv",
        targets_csv="dataset/targets.csv",
        dict_json="dataset/data_dictionary.json",
    )
    yield layer
    layer.close()


@pytest.fixture(scope="module")
def index(semantic_layer: DuckDBSemanticLayer) -> CardinalityTieredValueIndex:
    """Build the default value index."""

    return CardinalityTieredValueIndex(semantic_layer)


@pytest.fixture
def aliased_index(
    semantic_layer: DuckDBSemanticLayer,
) -> CardinalityTieredValueIndex:
    """Build an index with an explicit country-form alias."""

    class AliasedSemanticLayer:
        def __init__(self, base: DuckDBSemanticLayer) -> None:
            self._base = base

        def __getattr__(self, name: str) -> Any:
            return getattr(self._base, name)

        def get_value_aliases(self) -> dict[str, dict[str, str]]:
            return {"country": {"indian": "India"}}

    return CardinalityTieredValueIndex(AliasedSemanticLayer(semantic_layer))


def test_excluded_columns_not_indexed(index: CardinalityTieredValueIndex) -> None:
    assert "order_id" not in index.indexed_columns()
    assert "customer_id" not in index.indexed_columns()


def test_date_columns_not_indexed(index: CardinalityTieredValueIndex) -> None:
    assert "order_date" not in index.indexed_columns()


def test_expected_dimensions_indexed(index: CardinalityTieredValueIndex) -> None:
    expected = {
        "region",
        "country",
        "city",
        "customer_segment",
        "product_category",
        "product_subcategory",
        "product_name",
    }
    assert expected.issubset(index.indexed_columns())


def test_all_columns_are_exact_for_small_data(index: CardinalityTieredValueIndex) -> None:
    assert all(index.is_exact_column(column) for column in index.indexed_columns())


def test_cardinality_is_reported(index: CardinalityTieredValueIndex) -> None:
    assert index.cardinality("country") >= 1
    assert index.cardinality("region") >= 1
    assert index.cardinality("order_id") == -1


def test_exact_country_match(index: CardinalityTieredValueIndex) -> None:
    country = [match for match in index.match("India") if match.column == "country"]
    assert len(country) == 1
    assert country[0].canonical_value == "India"
    assert country[0].match_type == "exact"
    assert country[0].score == 1.0


def test_exact_matching_is_case_insensitive(index: CardinalityTieredValueIndex) -> None:
    assert index.match("india") == index.match("INDIA") == index.match("India")


@pytest.mark.parametrize(
    ("term", "column", "value"),
    [
        ("APAC", "region", "APAC"),
        ("San Francisco", "city", "San Francisco"),
        ("MacBook Air", "product_name", "MacBook Air"),
        ("Home Office", "customer_segment", "Home Office"),
    ],
)
def test_exact_values_match_expected_column(
    index: CardinalityTieredValueIndex,
    term: str,
    column: str,
    value: str,
) -> None:
    matches = [match for match in index.match(term) if match.column == column]
    assert len(matches) == 1
    assert matches[0].canonical_value == value
    assert matches[0].match_type == "exact"


def test_alias_driven_morphological_match(
    aliased_index: CardinalityTieredValueIndex,
) -> None:
    country = [
        match for match in aliased_index.match("Indian") if match.column == "country"
    ]
    assert len(country) == 1
    assert country[0].canonical_value == "India"
    assert country[0].match_type == "morphological"


def test_morphology_does_not_match_without_an_explicit_alias(
    index: CardinalityTieredValueIndex,
) -> None:
    assert not any(match.column == "country" for match in index.match("Indian"))


def test_unrelated_word_is_not_an_exact_or_morphological_match(
    index: CardinalityTieredValueIndex,
) -> None:
    assert not any(
        match.match_type in ("exact", "morphological")
        for match in index.match("clean")
    )


def test_short_terms_only_match_exact_values(index: CardinalityTieredValueIndex) -> None:
    assert index.match("cat") == []
    assert index.match("van") == []
    assert index.match("cat", threshold=0.5) == []
    assert any(
        match.column == "country" and match.match_type == "exact"
        for match in index.match("UK")
    )


def test_results_are_ranked_deterministically(index: CardinalityTieredValueIndex) -> None:
    matches = index.match("India")
    assert all(
        (first.score, TIER_RANK[first.match_type])
        >= (second.score, TIER_RANK[second.match_type])
        for first, second in zip(matches, matches[1:])
    )


def test_exact_matches_rank_above_morphological_and_fuzzy_matches(
    aliased_index: CardinalityTieredValueIndex,
) -> None:
    matches = aliased_index.match("Indian")
    assert all(
        (first.score, TIER_RANK[first.match_type])
        >= (second.score, TIER_RANK[second.match_type])
        for first, second in zip(matches, matches[1:])
    )


def test_empty_and_unknown_terms_return_no_matches(
    index: CardinalityTieredValueIndex,
) -> None:
    assert index.match("") == []
    assert index.match("   ") == []
    assert index.match("Qwertyuiop") == []


def test_unknown_term_returns_no_matches(index: CardinalityTieredValueIndex) -> None:
    assert index.match("Qwertyuiop") == []


def test_case_insensitivity_is_preserved(index: CardinalityTieredValueIndex) -> None:
    assert index.match("india") == index.match("India")


def test_low_limit_uses_sample_tier(semantic_layer: DuckDBSemanticLayer) -> None:
    sampled_index = CardinalityTieredValueIndex(semantic_layer, cardinality_limit=2)
    sampled_columns = [
        column
        for column in sampled_index.indexed_columns()
        if not sampled_index.is_exact_column(column)
    ]
    assert sampled_columns

    column = sampled_columns[0]
    value = str(semantic_layer.execute(f'SELECT "{column}" FROM v_sales LIMIT 1').iloc[0, 0])
    assert any(
        match.column == column and match.match_type == "exact"
        for match in sampled_index.match(value)
    )


def test_sample_tier_deduplicates_normalized_values(
    semantic_layer: DuckDBSemanticLayer,
) -> None:
    sampled_index = CardinalityTieredValueIndex(semantic_layer, cardinality_limit=2)
    for column in sampled_index.indexed_columns():
        if not sampled_index.is_exact_column(column):
            lookup = sampled_index._lookup[column]
            assert len(lookup) == len(set(lookup))


def test_sample_tier_returns_exact_sampled_values(
    semantic_layer: DuckDBSemanticLayer,
) -> None:
    sampled_index = CardinalityTieredValueIndex(
        semantic_layer,
        cardinality_limit=2,
        sample_size=1000,
    )
    sampled_columns = [
        column
        for column in sampled_index.indexed_columns()
        if not sampled_index.is_exact_column(column)
    ]
    column = sampled_columns[0]
    value = str(semantic_layer.execute(f'SELECT "{column}" FROM v_sales LIMIT 1').iloc[0, 0])
    assert any(
        match.column == column and match.match_type == "exact"
        for match in sampled_index.match(value)
    )


def test_invalid_configuration_is_rejected(
    semantic_layer: DuckDBSemanticLayer,
) -> None:
    with pytest.raises(ValueError, match="cardinality_limit"):
        CardinalityTieredValueIndex(semantic_layer, cardinality_limit=0)
