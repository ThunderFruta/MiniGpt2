from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .metrics import ResultRecord, estimate_cost
from .clients import ModelConfig


def build_summary_markdown(
    records: list[ResultRecord],
    model_configs: dict[str, ModelConfig] | None = None,
) -> str:
    config_map = model_configs or {}
    by_model: dict[str, list[ResultRecord]] = defaultdict(list)
    for record in records:
        by_model[record.model_name].append(record)

    lines = [
        "# LLM Evaluation Summary",
        "",
        "This report compares inference behavior only. It does not train, retrain, recover checkpoints, or validate old MiniGpt2 training runs.",
        "",
        "## Model Comparison",
        "",
        "| Model | Tasks | Accuracy | Avg latency sec | Avg total tokens | Errors | Cost per correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name in sorted(by_model):
        group = by_model[model_name]
        task_count = len(group)
        correct_count = sum(1 for item in group if item.correct)
        accuracy = correct_count / task_count if task_count else 0.0
        latencies = [item.latency_sec for item in group if item.latency_sec is not None]
        totals = [item.total_tokens for item in group if item.total_tokens is not None]
        errors = sum(1 for item in group if item.error)
        costs = [
            cost
            for item in group
            if (cost := estimate_cost(item, config_map.get(model_name))) is not None
        ]
        total_cost = sum(costs) if costs else None
        cost_per_correct = total_cost / correct_count if total_cost is not None and correct_count else None
        lines.append(
            "| {model} | {tasks} | {accuracy:.2%} | {latency} | {tokens} | {errors} | {cost} |".format(
                model=model_name,
                tasks=task_count,
                accuracy=accuracy,
                latency=_avg_or_blank(latencies),
                tokens=_avg_or_blank(totals),
                errors=errors,
                cost=_money_or_blank(cost_per_correct),
            )
        )

    lines.extend(
        [
            "",
            "## Hardware And API Notes",
            "",
            "- API access can expose strong hosted models and managed scaling, but it cannot provide low-level hardware telemetry, VRAM fit evidence, or full control over model weights.",
            "- RTX 4080 testing should be judged by endpoint success, latency, context length, error rate, and memory/fit limits reported by the local serving stack.",
            "- Cloud GPU testing is justified only when local endpoint failures or unacceptable latency are captured in the benchmark outputs.",
        ]
    )

    return "\n".join(lines) + "\n"


def write_summary(path: str | Path, records: list[ResultRecord], model_configs: dict[str, ModelConfig] | None = None) -> None:
    Path(path).write_text(build_summary_markdown(records, model_configs), encoding="utf-8")


def _avg_or_blank(values: list[float | int]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def _money_or_blank(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"

