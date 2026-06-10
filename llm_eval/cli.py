from __future__ import annotations

import argparse
from pathlib import Path

from .clients import load_model_configs
from .runner import run_evaluation
from .tasks import load_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm_eval",
        description="Run provider-neutral LLM inference evaluations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run tasks across configured models.")
    run_parser.add_argument("--tasks", required=True, help="Path to eval_tasks.jsonl")
    run_parser.add_argument("--models", required=True, help="Path to model config JSON")
    run_parser.add_argument("--out", required=True, help="Output directory for run artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        tasks = load_tasks(Path(args.tasks))
        models = load_model_configs(Path(args.models))
        run_evaluation(tasks, models, Path(args.out))
        print(f"wrote evaluation artifacts to {args.out}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2
