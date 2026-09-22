from types import SimpleNamespace

from engine.phase4.llm_builder import AnalyticalSpecOutput, LLMSpecBuilder
from engine.semantic_layer import DuckDBSemanticLayer
from engine.types import TaggedQuery


def _semantic_layer() -> DuckDBSemanticLayer:
    """Create a semantic layer for local mocked-LLM tests."""

    return DuckDBSemanticLayer(
        "dataset/sales_data.csv",
        "dataset/targets.csv",
        "dataset/data_dictionary.json",
    )


def test_parse_output_becomes_domain_spec() -> None:
    class Messages:
        def parse(self, **kwargs):
            return SimpleNamespace(
                parsed_output=AnalyticalSpecOutput(
                    operation="aggregate",
                    metrics=["profit"],
                    group_by=["city"],
                )
            )

    layer = _semantic_layer()
    try:
        spec = LLMSpecBuilder(SimpleNamespace(messages=Messages()), layer).build(
            TaggedQuery("q", [], [])
        )
        assert spec is not None
        assert spec.metrics == ["profit"]
        assert spec.group_by == ["city"]
    finally:
        layer.close()


def test_parse_exception_uses_tool_fallback_once() -> None:
    class Messages:
        def __init__(self) -> None:
            self.create_calls = 0

        def parse(self, **kwargs):
            raise RuntimeError("structured output unavailable")

        def create(self, **kwargs):
            self.create_calls += 1
            block = SimpleNamespace(
                type="tool_use",
                input={"operation": "aggregate", "metrics": ["profit"]},
            )
            return SimpleNamespace(content=[block])

    messages = Messages()
    layer = _semantic_layer()
    try:
        spec = LLMSpecBuilder(SimpleNamespace(messages=messages), layer).build(
            TaggedQuery("q", [], [])
        )
        assert spec is not None
        assert spec.metrics == ["profit"]
        assert messages.create_calls == 1
    finally:
        layer.close()


def test_invalid_provider_payload_returns_none() -> None:
    class Messages:
        def parse(self, **kwargs):
            return SimpleNamespace(parsed_output={"operation": "not_supported"})

        def create(self, **kwargs):
            return SimpleNamespace(content=[])

    layer = _semantic_layer()
    try:
        assert LLMSpecBuilder(SimpleNamespace(messages=Messages()), layer).build(
            TaggedQuery("q", [], [])
        ) is None
    finally:
        layer.close()
