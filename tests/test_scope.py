import ast
from pathlib import Path


def test_new_package_does_not_import_legacy_training_modules():
    forbidden = {"TrainStages", "MergeStages", "Reset&Train", "LoraAdapters", "AdapterBackups"}
    package_root = Path("llm_eval")

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                assert names.isdisjoint(forbidden), f"{path} imports forbidden legacy module"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"{path} imports forbidden legacy module"


def test_docs_do_not_claim_model_training_verified():
    for path in [Path("README.md"), Path("Architecture.md")]:
        text = path.read_text(encoding="utf-8").casefold()
        assert "model training verified" not in text
        assert "training verified" not in text


def test_eval_task_file_has_at_least_50_tasks():
    lines = [
        line
        for line in Path("data/eval_tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 50

