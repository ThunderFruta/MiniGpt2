import json

import pytest

from llm_eval.tasks import TaskValidationError, load_tasks, parse_task


def test_load_tasks_validates_jsonl(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "math_001",
                "task_type": "math",
                "prompt": "2+2?",
                "expected_answer": "4",
                "judge": {"type": "exact"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = load_tasks(path)

    assert len(tasks) == 1
    assert tasks[0].task_id == "math_001"
    assert tasks[0].judge == {"type": "exact"}


def test_load_tasks_rejects_missing_required_field(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"task_id":"x","task_type":"math","prompt":"hi"}\n', encoding="utf-8")

    with pytest.raises(TaskValidationError, match="expected_answer"):
        load_tasks(path)


def test_load_tasks_rejects_invalid_judge(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_id":"x","task_type":"math","prompt":"hi","expected_answer":"hi","judge":{"type":"llm"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(TaskValidationError, match="judge.type"):
        load_tasks(path)


def test_load_tasks_rejects_malformed_jsonl(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"task_id":\n', encoding="utf-8")

    with pytest.raises(TaskValidationError, match="invalid JSON"):
        load_tasks(path)


def test_parse_task_defaults_to_exact_judge():
    task = parse_task(
        {
            "task_id": "x",
            "task_type": "instruction_following",
            "prompt": "Return ok",
            "expected_answer": "ok",
        }
    )

    assert task.judge == {"type": "exact"}

