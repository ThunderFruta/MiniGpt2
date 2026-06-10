from __future__ import annotations

import re
from typing import Any

from .tasks import EvalTask


def normalize_text(value: str, *, case_sensitive: bool = False, strip: bool = True) -> str:
    text = value.strip() if strip else value
    text = re.sub(r"\s+", " ", text)
    return text if case_sensitive else text.casefold()


def judge_output(task: EvalTask, model_output: str) -> bool:
    config = task.judge
    judge_type = config["type"]
    case_sensitive = bool(config.get("case_sensitive", False))
    expected = str(config.get("expected", task.expected_answer))

    if judge_type == "exact":
        return normalize_text(model_output, case_sensitive=case_sensitive) == normalize_text(
            expected,
            case_sensitive=case_sensitive,
        )
    if judge_type == "contains":
        haystack = normalize_text(model_output, case_sensitive=case_sensitive)
        needle = normalize_text(expected, case_sensitive=case_sensitive)
        return needle in haystack
    if judge_type == "regex":
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.search(str(config["pattern"]), model_output, flags=flags) is not None
    raise ValueError(f"unsupported judge type: {judge_type}")

