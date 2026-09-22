from engine.types import (
    AnalyticalSpec,
    EntitySpan,
    Filter,
    MetricFilter,
    NLUResult,
    OrderSpec,
    StructuredQuery,
    TimeComparison,
    TimeConstraint,
    Token,
    ValueMatch,
)


def test_entity_span_computed_offsets() -> None:
    t1 = Token(text="United", start=0, end=6)
    t2 = Token(text="States", start=7, end=13)
    span = EntitySpan(
        tokens=(t1, t2),
        canonical="United States",
        role="value",
        matched_column="country",
    )
    assert span.start == 0
    assert span.end == 13
    assert span.text == "United States"
    assert span.canonical == "United States"
    assert span.matched_column == "country"


def test_entity_span_single_token() -> None:
    token = Token(text="India", start=0, end=5)
    span = EntitySpan(
        tokens=(token,),
        canonical="India",
        role="value",
        matched_column="country",
    )
    assert span.start == 0
    assert span.end == 5
    assert span.text == "India"


def test_entity_span_is_hashable() -> None:
    token = Token(text="India", start=0, end=5)
    span = EntitySpan(tokens=(token,), canonical="India", role="value")
    assert {span}


def test_value_match_carries_match_type() -> None:
    match = ValueMatch(
        column="country",
        canonical_value="India",
        score=1.0,
        match_type="exact",
    )
    assert match.match_type == "exact"
    assert match.column == "country"
    assert match.canonical_value == "India"


def test_value_match_fuzzy_variant() -> None:
    match = ValueMatch(
        column="city",
        canonical_value="Bangalore",
        score=0.91,
        match_type="fuzzy",
    )
    assert match.match_type == "fuzzy"
    assert 0.0 < match.score < 1.0


def test_value_match_morphological_variant() -> None:
    match = ValueMatch(
        column="country",
        canonical_value="India",
        score=0.95,
        match_type="morphological",
    )
    assert match.match_type == "morphological"


def test_structured_query_compare_with_having() -> None:
    spec = AnalyticalSpec(
        operation="compare",
        metrics=["revenue"],
        group_by=["region"],
        filters=[Filter(column="month", operator="=", value="2024-02")],
        metric_filters=[
            MetricFilter(
                metric="revenue",
                operator="<",
                target_metric="target_revenue",
            )
        ],
    )
    query = StructuredQuery(
        raw_query="Which region missed its target in Feb?",
        tokens=[],
        spans=[],
        spec=spec,
        nlu_confidence=0.9,
    )
    assert query.intent_label == "compare"
    assert query.spec.metric_filters[0].metric == "revenue"
    assert query.spec.metric_filters[0].operator == "<"
    assert query.spec.metric_filters[0].target_metric == "target_revenue"


def test_structured_query_rank_with_contribution() -> None:
    spec = AnalyticalSpec(
        operation="rank",
        metrics=["revenue"],
        transforms=["contribution_pct"],
        group_by=["product_name"],
        order_by=OrderSpec(metric="revenue", direction="DESC"),
        filters=[Filter(column="region", operator="=", value="APAC")],
        limit=3,
    )
    query = StructuredQuery(
        raw_query="Top 3 products by sales contribution % in APAC",
        tokens=[],
        spans=[],
        spec=spec,
        nlu_confidence=0.95,
    )
    assert query.intent_label == "rank+contribution_pct"
    assert query.spec.limit == 3
    assert query.spec.filters[0].value == "APAC"


def test_structured_query_yoy_uses_time_comparison() -> None:
    spec = AnalyticalSpec(
        operation="trend",
        metrics=["revenue"],
        transforms=["yoy"],
        time_comparison=TimeComparison(
            baseline=TimeConstraint(
                raw="current year",
                column="year",
                resolution="LATEST_PERIOD",
                resolved_value="2024",
            ),
            delta_type="YoY",
        ),
    )
    query = StructuredQuery(
        raw_query="YoY growth in revenue",
        tokens=[],
        spans=[],
        spec=spec,
        nlu_confidence=0.9,
    )
    assert query.intent_label == "trend+yoy"
    assert query.spec.time_comparison is not None
    assert query.spec.time_comparison.delta_type == "YoY"


def test_default_applied_tracking() -> None:
    spec = AnalyticalSpec(operation="rank", metrics=["revenue"], limit=5)
    query = StructuredQuery(
        raw_query="Top 5 cities",
        tokens=[],
        spans=[],
        spec=spec,
        default_applied=["default_metric_applied"],
        nlu_confidence=0.7,
    )
    assert "default_metric_applied" in query.default_applied


def test_rejection_shape() -> None:
    result = NLUResult(status="rejected", reason="Out of domain")
    assert result.status == "rejected"
    assert result.structured_query is None
    assert result.reason == "Out of domain"
    assert result.alternatives == []


def test_ambiguous_shape() -> None:
    spec_a = AnalyticalSpec(
        operation="aggregate",
        metrics=["revenue"],
        filters=[Filter(column="product_category", operator="=", value="Sales")],
    )
    spec_b = AnalyticalSpec(
        operation="aggregate",
        metrics=["revenue"],
        filters=[Filter(column="customer_segment", operator="=", value="Sales")],
    )
    query_a = StructuredQuery(raw_query="q", tokens=[], spans=[], spec=spec_a)
    query_b = StructuredQuery(raw_query="q", tokens=[], spans=[], spec=spec_b)
    result = NLUResult(
        status="ambiguous",
        reason="'Sales' matches more than one dimension",
        alternatives=[query_a, query_b],
    )
    assert result.status == "ambiguous"
    assert result.structured_query is None
    assert len(result.alternatives) == 2


def test_parsed_shape() -> None:
    spec = AnalyticalSpec(operation="aggregate", metrics=["revenue"])
    query = StructuredQuery(raw_query="q", tokens=[], spans=[], spec=spec)
    result = NLUResult(status="parsed", structured_query=query)
    assert result.status == "parsed"
    assert result.structured_query is query
    assert result.reason is None
