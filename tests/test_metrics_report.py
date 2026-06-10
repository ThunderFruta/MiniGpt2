import csv

from llm_eval.clients import ModelConfig
from llm_eval.metrics import METRICS_COLUMNS, ResultRecord, record_to_metrics_row, tokens_per_second, write_metrics_csv
from llm_eval.report import build_summary_markdown


def test_tokens_per_second_handles_zero_and_missing_latency():
    assert tokens_per_second(100, 2.0) == 50.0
    assert tokens_per_second(100, 0) is None
    assert tokens_per_second(None, 2.0) is None


def test_record_to_metrics_row_estimates_cost():
    config = ModelConfig(
        name="api",
        model="m",
        base_url="http://test/v1",
        price_per_1k_prompt_tokens=0.01,
        price_per_1k_completion_tokens=0.02,
    )
    row = record_to_metrics_row(
        ResultRecord(
            model_name="api",
            task_id="t1",
            task_type="math",
            expected_answer="4",
            model_output="4",
            correct=True,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_sec=3.0,
        ),
        config,
    )

    assert list(row.keys()) == METRICS_COLUMNS
    assert row["tokens_per_second"] == "50.000000"
    assert row["cost_estimate"] == "0.00200000"


def test_write_metrics_csv_uses_required_columns(tmp_path):
    path = tmp_path / "metrics.csv"
    write_metrics_csv(
        path,
        [
            ResultRecord(
                model_name="m",
                task_id="t1",
                task_type="math",
                expected_answer="4",
                model_output="4",
                correct=True,
            )
        ],
    )

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == METRICS_COLUMNS
        assert next(reader)["correct"] == "true"


def test_summary_compares_models_and_states_non_training_boundary():
    summary = build_summary_markdown(
        [
            ResultRecord("m1", "t1", "math", "4", "4", True, latency_sec=1.0, total_tokens=10),
            ResultRecord("m2", "t1", "math", "4", "5", False, error="bad"),
        ]
    )

    assert "| m1 | 1 | 100.00%" in summary
    assert "| m2 | 1 | 0.00%" in summary
    assert "does not train" in summary
    assert "Cloud GPU testing" in summary

