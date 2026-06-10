# Architecture

The active system is an inference evaluation pipeline:

```text
TaskLoader -> ModelClient -> Runner -> Judge -> Metrics -> Report
```

## Pipeline

- `TaskLoader` reads JSONL tasks from `data/eval_tasks.jsonl` and validates required fields.
- `ModelClient` calls API or local OpenAI-compatible endpoints and standardizes text, latency, token usage, and errors.
- `Runner` runs every task against every configured model and saves raw and judged artifacts.
- `Judge` scores responses with exact, contains, or regex matching.
- `Metrics` writes `metrics.csv` with latency, token, accuracy, error, and cost fields.
- `Report` writes `summary.md` for model comparison and hardware/API interpretation.

## Endpoint Model

The first supported interface is OpenAI-compatible `/chat/completions`. This covers many hosted APIs and local serving stacks without tying the harness to one provider. Provider-specific behavior belongs in config, not in benchmark logic.

## Non-Training Boundary

The evaluation harness never trains models, restores adapters, recovers checkpoints, or imports legacy MiniGpt2 training modules. Old prototype files are preserved as evidence that prior work existed, not as active dependencies.

