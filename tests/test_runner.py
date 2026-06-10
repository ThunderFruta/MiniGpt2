import json

from llm_eval.clients import ModelConfig, ModelResponse
from llm_eval.runner import run_evaluation
from llm_eval.tasks import EvalTask


def test_runner_saves_artifacts_and_continues_after_errors(tmp_path):
    tasks = [
        EvalTask("t1", "math", "2+2", "4", {"type": "exact"}),
        EvalTask("t2", "math", "fail", "ok", {"type": "exact"}),
    ]
    configs = [
        ModelConfig(name="good", model="m", base_url="http://test/v1"),
        ModelConfig(name="mixed", model="m", base_url="http://test/v1"),
    ]

    def factory(config):
        return _FakeClient(config.name)

    records = run_evaluation(tasks, configs, tmp_path, factory)

    assert len(records) == 4
    assert (tmp_path / "raw_outputs.jsonl").exists()
    assert (tmp_path / "judged_results.jsonl").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary.md").exists()
    assert any(record.error for record in records)

    raw_lines = [
        json.loads(line)
        for line in (tmp_path / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert raw_lines[0]["model_name"] == "good"


class _FakeClient:
    def __init__(self, name):
        self.name = name

    def generate(self, prompt):
        if self.name == "mixed" and prompt == "fail":
            raise RuntimeError("boom")
        return ModelResponse(
            model_name=self.name,
            output_text="4" if prompt == "2+2" else "wrong",
            latency_sec=0.5,
            prompt_tokens=3,
            completion_tokens=1,
            total_tokens=4,
        )

