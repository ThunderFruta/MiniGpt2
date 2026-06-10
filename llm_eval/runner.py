from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .clients import ModelConfig, ModelResponse, OpenAICompatibleClient
from .judges import judge_output
from .metrics import ResultRecord, write_metrics_csv
from .report import write_summary
from .tasks import EvalTask


ClientFactory = Callable[[ModelConfig], object]


def default_client_factory(config: ModelConfig) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(config)


def run_evaluation(
    tasks: list[EvalTask],
    model_configs: list[ModelConfig],
    output_dir: str | Path,
    client_factory: ClientFactory = default_client_factory,
) -> list[ResultRecord]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_outputs.jsonl"
    judged_path = out / "judged_results.jsonl"

    records: list[ResultRecord] = []
    config_map = {config.name: config for config in model_configs}

    with raw_path.open("w", encoding="utf-8") as raw_handle, judged_path.open("w", encoding="utf-8") as judged_handle:
        for config in model_configs:
            client = client_factory(config)
            for task in tasks:
                record = _run_one(client, config.name, task)
                records.append(record)
                raw_handle.write(json.dumps(_raw_json(record), ensure_ascii=False) + "\n")
                judged_handle.write(json.dumps(_judged_json(record), ensure_ascii=False) + "\n")

    write_metrics_csv(out / "metrics.csv", records, config_map)
    write_summary(out / "summary.md", records, config_map)
    return records


def _run_one(client: object, model_name: str, task: EvalTask) -> ResultRecord:
    try:
        response = client.generate(task.prompt)  # type: ignore[attr-defined]
        if not isinstance(response, ModelResponse):
            raise TypeError("client.generate() must return ModelResponse")
        correct = judge_output(task, response.output_text)
        return ResultRecord(
            model_name=model_name,
            task_id=task.task_id,
            task_type=task.task_type,
            expected_answer=task.expected_answer,
            model_output=response.output_text,
            correct=correct,
            latency_sec=response.latency_sec,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )
    except Exception as exc:  # Keep benchmark runs going across model/task failures.
        return ResultRecord(
            model_name=model_name,
            task_id=task.task_id,
            task_type=task.task_type,
            expected_answer=task.expected_answer,
            model_output="",
            correct=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _raw_json(record: ResultRecord) -> dict[str, object]:
    return {
        "model_name": record.model_name,
        "task_id": record.task_id,
        "task_type": record.task_type,
        "model_output": record.model_output,
        "error": record.error,
    }


def _judged_json(record: ResultRecord) -> dict[str, object]:
    return {
        "model_name": record.model_name,
        "task_id": record.task_id,
        "task_type": record.task_type,
        "expected_answer": record.expected_answer,
        "model_output": record.model_output,
        "correct": record.correct,
        "error": record.error,
    }

