from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clients import ModelConfig


METRICS_COLUMNS = [
    "model_name",
    "task_id",
    "task_type",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_sec",
    "tokens_per_second",
    "expected_answer",
    "model_output",
    "correct",
    "error",
    "cost_estimate",
]


@dataclass(frozen=True)
class ResultRecord:
    model_name: str
    task_id: str
    task_type: str
    expected_answer: str
    model_output: str
    correct: bool
    latency_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: str = ""


def tokens_per_second(total_tokens: int | None, latency_sec: float | None) -> float | None:
    if total_tokens is None or latency_sec is None or latency_sec <= 0:
        return None
    return total_tokens / latency_sec


def estimate_cost(record: ResultRecord, model_config: ModelConfig | None) -> float | None:
    if model_config is None:
        return None
    prompt_price = model_config.price_per_1k_prompt_tokens
    completion_price = model_config.price_per_1k_completion_tokens
    if prompt_price is None and completion_price is None:
        return None
    cost = 0.0
    if prompt_price is not None and record.prompt_tokens is not None:
        cost += (record.prompt_tokens / 1000.0) * prompt_price
    if completion_price is not None and record.completion_tokens is not None:
        cost += (record.completion_tokens / 1000.0) * completion_price
    return cost


def record_to_metrics_row(
    record: ResultRecord,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    tps = tokens_per_second(record.total_tokens, record.latency_sec)
    cost = estimate_cost(record, model_config)
    return {
        "model_name": record.model_name,
        "task_id": record.task_id,
        "task_type": record.task_type,
        "prompt_tokens": _blank_none(record.prompt_tokens),
        "completion_tokens": _blank_none(record.completion_tokens),
        "total_tokens": _blank_none(record.total_tokens),
        "latency_sec": _format_float(record.latency_sec),
        "tokens_per_second": _format_float(tps),
        "expected_answer": record.expected_answer,
        "model_output": record.model_output,
        "correct": str(record.correct).lower(),
        "error": record.error,
        "cost_estimate": _format_float(cost, places=8),
    }


def write_metrics_csv(
    path: str | Path,
    records: list[ResultRecord],
    model_configs: dict[str, ModelConfig] | None = None,
) -> None:
    config_map = model_configs or {}
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record_to_metrics_row(record, config_map.get(record.model_name)))


def _blank_none(value: Any) -> Any:
    return "" if value is None else value


def _format_float(value: float | None, places: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{places}f}"

