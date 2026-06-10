from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


SUPPORTED_JUDGES = {"exact", "contains", "regex"}


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    task_type: str
    prompt: str
    expected_answer: str
    judge: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskValidationError(ValueError):
    pass


def _require_string(record: dict[str, Any], field_name: str, line_no: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise TaskValidationError(f"line {line_no}: '{field_name}' must be a non-empty string")
    return value


def _validate_judge(record: dict[str, Any], line_no: int) -> dict[str, Any]:
    judge = record.get("judge")
    if judge is None:
        judge = {"type": "exact"}
    if not isinstance(judge, dict):
        raise TaskValidationError(f"line {line_no}: 'judge' must be an object")
    judge_type = judge.get("type")
    if not isinstance(judge_type, str) or judge_type not in SUPPORTED_JUDGES:
        supported = ", ".join(sorted(SUPPORTED_JUDGES))
        raise TaskValidationError(f"line {line_no}: judge.type must be one of: {supported}")
    if judge_type == "regex" and not isinstance(judge.get("pattern"), str):
        raise TaskValidationError(f"line {line_no}: regex judges require a string 'pattern'")
    return dict(judge)


def parse_task(record: dict[str, Any], line_no: int = 1) -> EvalTask:
    if not isinstance(record, dict):
        raise TaskValidationError(f"line {line_no}: task must be a JSON object")

    metadata = record.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TaskValidationError(f"line {line_no}: 'metadata' must be an object when present")

    return EvalTask(
        task_id=_require_string(record, "task_id", line_no),
        task_type=_require_string(record, "task_type", line_no),
        prompt=_require_string(record, "prompt", line_no),
        expected_answer=_require_string(record, "expected_answer", line_no),
        judge=_validate_judge(record, line_no),
        metadata=dict(metadata),
    )


def load_tasks(path: str | Path) -> list[EvalTask]:
    task_path = Path(path)
    tasks: list[EvalTask] = []
    seen_ids: set[str] = set()

    with task_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TaskValidationError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            task = parse_task(record, line_no)
            if task.task_id in seen_ids:
                raise TaskValidationError(f"line {line_no}: duplicate task_id '{task.task_id}'")
            seen_ids.add(task.task_id)
            tasks.append(task)

    if not tasks:
        raise TaskValidationError(f"{task_path}: no tasks found")
    return tasks
