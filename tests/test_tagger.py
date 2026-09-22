import pytest

from engine.semantic_layer import DuckDBSemanticLayer
from engine.tagger import Tagger, TaggerError
from engine.types import ValueMatch
from engine.value_index import CardinalityTieredValueIndex


@pytest.fixture(scope="module")
def semantic_layer() -> DuckDBSemanticLayer:
    """Provide the real semantic layer used by lexical tagging tests."""

    layer = DuckDBSemanticLayer(
        sales_csv="dataset/sales_data.csv",
        targets_csv="dataset/targets.csv",
        dict_json="dataset/data_dictionary.json",
    )
    yield layer
    layer.close()


@pytest.fixture(scope="module")
def value_index(
    semantic_layer: DuckDBSemanticLayer,
) -> CardinalityTieredValueIndex:
    """Provide the real data-value lookup service."""

    return CardinalityTieredValueIndex(semantic_layer)


@pytest.fixture(scope="module")
def tagger(
    semantic_layer: DuckDBSemanticLayer,
    value_index: CardinalityTieredValueIndex,
) -> Tagger:
    """Provide the tagger under test."""

    return Tagger(semantic_layer, value_index)


def test_tokenize_preserves_offsets(tagger: Tagger) -> None:
    tagged = tagger.tag("Top 2 cities by profit")
    assert [token.text for token in tagged.tokens] == ["Top", "2", "cities", "by", "profit"]
    assert [(token.start, token.end) for token in tagged.tokens] == [
        (0, 3),
        (4, 5),
        (6, 12),
        (13, 15),
        (16, 22),
    ]


def test_tokenize_drops_punctuation(tagger: Tagger) -> None:
    tagged = tagger.tag("Top 2 cities, by profit?")
    assert [token.text for token in tagged.tokens] == ["Top", "2", "cities", "by", "profit"]


def test_rank_query_tags_number_dimension_and_metric(tagger: Tagger) -> None:
    tagged = tagger.tag("Top 2 cities by profit")
    spans = {(span.canonical, span.role, span.matched_column) for span in tagged.spans}
    assert ("2", "number", None) in spans
    assert ("city", "dimension", "city") in spans
    assert ("profit", "metric", None) in spans


@pytest.mark.parametrize(
    ("query", "canonical_value", "column"),
    [
        ("Revenue in San Francisco", "San Francisco", "city"),
        ("Revenue for Home Office", "Home Office", "customer_segment"),
        ("Show me MacBook Air sales", "MacBook Air", "product_name"),
    ],
)
def test_multi_word_values_claim_one_span(
    tagger: Tagger,
    query: str,
    canonical_value: str,
    column: str,
) -> None:
    tagged = tagger.tag(query)
    assert any(
        span.canonical == canonical_value
        and span.role == "value"
        and span.matched_column == column
        for span in tagged.spans
    )


@pytest.mark.parametrize(
    ("query", "canonical_metric"),
    [
        ("Total sales in India", "revenue"),
        ("AOV by region", "avg_order_value"),
        ("Average order value by region", "avg_order_value"),
    ],
)
def test_synonyms_resolve_to_canonical_metrics(
    tagger: Tagger,
    query: str,
    canonical_metric: str,
) -> None:
    tagged = tagger.tag(query)
    assert any(
        span.role == "metric" and span.canonical == canonical_metric
        for span in tagged.spans
    )


@pytest.mark.parametrize(
    ("query", "canonical_dimension"),
    [
        ("customer", "customer_id"),
        ("customers", "customer_id"),
        ("product", "product_name"),
        ("products", "product_name"),
        ("category", "product_category"),
    ],
)
def test_dimension_business_aliases_resolve_to_schema_columns(
    tagger: Tagger,
    query: str,
    canonical_dimension: str,
) -> None:
    tagged = tagger.tag(query)
    assert [(span.role, span.canonical) for span in tagged.spans] == [
        ("dimension", canonical_dimension)
    ]


@pytest.mark.parametrize("query", ["Sales in India for March", "YoY growth in revenue"])
def test_time_terms_are_tagged(tagger: Tagger, query: str) -> None:
    tagged = tagger.tag(query)
    assert any(span.role == "time" for span in tagged.spans)


def test_unknown_words_are_left_untagged(tagger: Tagger) -> None:
    tagged = tagger.tag("Top 2 cities by profit")
    tagged_words = {token.text for span in tagged.spans for token in span.tokens}
    assert "Top" not in tagged_words
    assert "by" not in tagged_words


def test_simple_query_has_no_conflicts(tagger: Tagger) -> None:
    assert tagger.tag("Top 2 cities by profit").conflicts == []


def test_nested_customer_ranking_tags_all_structural_roles_in_order(
    tagger: Tagger,
) -> None:
    tagged = tagger.tag("Revenue of top 3 customers per region")
    assert [(span.role, span.canonical) for span in tagged.spans] == [
        ("metric", "revenue"),
        ("number", "3"),
        ("dimension", "customer_id"),
        ("dimension", "region"),
    ]


def test_conflict_is_recorded_with_deterministic_default(
    semantic_layer: DuckDBSemanticLayer,
) -> None:
    class ConflictingValueIndex:
        def match(self, term: str, threshold: float | None = None) -> list[ValueMatch]:
            if term.lower() == "profit":
                return [ValueMatch("city", "Profit", 1.0, "exact")]
            return []

    tagged = Tagger(semantic_layer, ConflictingValueIndex()).tag("profit")
    assert len(tagged.conflicts) == 1
    assert tagged.conflicts[0].chosen.role == "value"
    assert {span.role for span in tagged.conflicts[0].candidates} == {"metric", "value"}


def test_oversized_query_is_rejected(tagger: Tagger) -> None:
    with pytest.raises(TaggerError, match="character limit"):
        tagger.tag("x" * (Tagger.MAX_QUERY_LENGTH + 1))


@pytest.mark.parametrize(
    ("query", "expected_roles"),
    [
        ("Total sales in India for March", {"metric", "value", "time"}),
        ("Top 2 cities by profit", {"number", "dimension", "metric"}),
        ("Average order value by region", {"dimension"}),
        ("Which region missed its target in Feb?", {"dimension", "time"}),
        ("Sales contribution % by category", {"metric"}),
        ("Top product in each region", {"dimension"}),
        ("YoY growth in revenue", {"time", "metric"}),
        ("Revenue of top 3 customers per region", {"metric", "number", "dimension"}),
    ],
)
def test_assignment_queries_produce_at_least_one_expected_role(
    tagger: Tagger,
    query: str,
    expected_roles: set[str],
) -> None:
    actual_roles = {span.role for span in tagger.tag(query).spans}
    assert actual_roles & expected_roles
