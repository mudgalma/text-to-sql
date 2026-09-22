"""Run the intelligent analytics query engine over a JSON query batch."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from engine.memory.feedback_store import FeedbackStore
from engine.understanding.spec_builder import SpecBuilder
from engine.generation.generator import SQLGenerationError, SQLGenerator
from engine.execution.executor import Executor
from engine.correction.repairer import SelfCorrector
from engine.scoring.scorer import ConfidenceScorer
from engine.insight.explainer import ExplanationClient, Explainer
from engine.understanding.llm_builder import LLMSpecBuilder
import os
from dotenv import load_dotenv
from engine.semantic_layer import DuckDBSemanticLayer
from engine.tagger import Tagger, TaggerError
from engine.types import AnalyticalSpec
from engine.value_index import CardinalityTieredValueIndex


LOGGER = logging.getLogger(__name__)


class OpenRouterAdapter:
    """Adapter to make OpenAI client satisfy the generate(system, user) -> str protocol."""
    
    def __init__(self, client: Any, model: str = "anthropic/claude-3.5-haiku"):
        self._client = client
        self._model = model

    def generate(self, *, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


def load_queries(path_value: str | Path) -> list[str]:
    """Load a bounded list of query strings from the assignment JSON format."""

    path = Path(path_value).expanduser().resolve()
    try:
        with path.open(encoding="utf-8-sig") as source:
            raw_entries = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Query file could not be loaded as JSON.") from error
    if not isinstance(raw_entries, list):
        raise ValueError("Query JSON must be an array of objects.")
    queries: list[str] = []
    for entry in raw_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("query"), str):
            raise ValueError("Every query entry must contain a string query field.")
        queries.append(entry["query"])
    return queries


def run_pipeline(
    dataset_dir: str | Path = "dataset",
    feedback_path: str | Path | None = None,
    explanation_client: ExplanationClient | None = None,
) -> list[dict[str, Any]]:
    """Run all queries, optionally using an injected LLM explanation client."""

    data_dir = Path(dataset_dir).expanduser().resolve()
    layer = DuckDBSemanticLayer(
        data_dir / "sales_data.csv",
        data_dir / "targets.csv",
        data_dir / "data_dictionary.json",
    )
    try:
        llm_client = None
        adapter = None
        api_key = os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            from openai import OpenAI
            llm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            adapter = OpenRouterAdapter(llm_client)

        tagger = Tagger(layer, CardinalityTieredValueIndex(layer))
        
        llm_builder = LLMSpecBuilder(llm_client, layer) if llm_client else None
        spec_builder = SpecBuilder(layer, llm_builder=llm_builder)
        
        generator = SQLGenerator(layer, llm_client=adapter)
        executor = Executor(layer)
        corrector = SelfCorrector(executor, layer, llm_client=adapter)
        scorer = ConfidenceScorer(layer)
        explainer = Explainer(explanation_client or adapter)
        feedback = FeedbackStore(feedback_path or data_dir / "feedback_log.csv")
        return [
            _run_query(
                query,
                tagger,
                spec_builder,
                generator,
                corrector,
                scorer,
                explainer,
                feedback,
            )
            for query in load_queries(data_dir / "nl_queries.json")
        ]
    finally:
        layer.close()


def _run_query(
    query: str,
    tagger: Tagger,
    spec_builder: SpecBuilder,
    generator: SQLGenerator,
    corrector: SelfCorrector,
    scorer: ConfidenceScorer,
    explainer: Explainer,
    feedback: FeedbackStore,
) -> dict[str, Any]:
    """Run one query and return a JSON-ready result even when a stage fails."""

    try:
        tagged = tagger.tag(query)
        spec = spec_builder.build(tagged)
    except TaggerError as error:
        return _failure_output(query, f"Could not interpret this query: {error}")
    if spec is None:
        return _failure_output(query, "Could not interpret this query.")

    feedback_sql = feedback.get_exact_correction(query)
    feedback_applied = feedback_sql is not None
    try:
        sql = feedback_sql or generator.generate(spec)
    except SQLGenerationError as error:
        return _failure_output(query, f"Could not generate safe SQL: {error}")

    correction = corrector.execute(sql, spec)
    final = correction.final
    confidence = scorer.score(
        tagged=tagged,
        spec=spec,
        execution=final,
        retries_used=correction.retries_used,
        rule_spec_present=True,
        llm_spec_present=bool(spec.defaults_applied and "llm_fallback" in spec.defaults_applied),
        specs_agree=None,
    )
    explanation = explainer.explain(
        tagged=tagged,
        spec=spec,
        confidence=confidence.final,
        retries_used=correction.retries_used,
        rule_spec_present=True,
        llm_spec_present=bool(spec.defaults_applied and "llm_fallback" in spec.defaults_applied),
        specs_agree=None,
        template_path_used=not feedback_applied,
        result_count=len(final.df) if final.df is not None else None,
        feedback_applied=feedback_applied,
    )
    return {
        "query": query,
        "generated_logic": final.sql,
        "result": final.df.to_dict(orient="records") if final.df is not None else None,
        "confidence_score": confidence.final,
        "explanation": explanation,
    }


def _failure_output(query: str, explanation: str) -> dict[str, Any]:
    """Return the required output shape for an interpretation or generation failure."""

    return {
        "query": query,
        "generated_logic": None,
        "result": None,
        "confidence_score": 0.0,
        "explanation": explanation,
    }


def main() -> None:
    """Parse CLI arguments, run the batch, print it, and persist the JSON output."""

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--feedback-path")
    parser.add_argument("--output", default="outputs.json")
    args = parser.parse_args()
    outputs = run_pipeline(args.dataset_dir, args.feedback_path)
    serialized = json.dumps(outputs, indent=2, default=str)
    print(serialized)
    Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    LOGGER.info("pipeline_completed", extra={"queries": len(outputs), "output": args.output})


if __name__ == "__main__":
    main()
