from llm_eval.judges import judge_output
from llm_eval.tasks import EvalTask


def _task(expected: str, judge: dict):
    return EvalTask(
        task_id="t1",
        task_type="math",
        prompt="prompt",
        expected_answer=expected,
        judge=judge,
    )


def test_exact_judge_normalizes_case_and_whitespace_by_default():
    assert judge_output(_task("Hello world", {"type": "exact"}), "  hello   WORLD  ")


def test_exact_judge_can_be_case_sensitive():
    assert not judge_output(
        _task("READY", {"type": "exact", "case_sensitive": True}),
        "ready",
    )


def test_contains_judge():
    assert judge_output(_task("needle", {"type": "contains"}), "The NEEDLE is here.")


def test_regex_judge():
    assert judge_output(
        _task("unused", {"type": "regex", "pattern": r"answer\s*:\s*42"}),
        "Answer: 42",
    )


def test_regex_judge_failure():
    assert not judge_output(
        _task("unused", {"type": "regex", "pattern": r"^42$"}),
        "The answer is 42.",
    )

