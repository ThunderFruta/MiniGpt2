# Clean LLM Evaluation Harness

This repository now treats the old MiniGpt2 files as historical prototype evidence only. The active project is a reproducible inference, evaluation, and systems benchmarking harness for comparing LLMs across API and local OpenAI-compatible endpoints.

This is not a training project. It does not retrain MiniGpt2, recover old checkpoints, rebuild LoRA adapters, or verify old training runs.

## What This Answers

- Which model performs better on the same tasks?
- Which model is faster?
- Which model uses fewer tokens?
- Which model is cheaper per correct answer?
- What can API access provide or not provide?
- What can the RTX 4080 handle through a local endpoint?
- What does not fit or perform well locally?
- Whether cloud GPU testing is justified by benchmark data.

## Quickstart

Run an evaluation with an OpenAI-compatible model config:

```bash
python -m llm_eval run --tasks data/eval_tasks.jsonl --models config/models.example.json --out runs/example
```

If this machine does not provide a `python` executable, use `python3 -m llm_eval` with the same arguments.

The example config includes a DeepSeek-style API profile and a local OpenAI-compatible endpoint profile. Set the configured API key environment variable only when using a remote API profile. Local endpoints can use `api_key_env: null`.

The harness writes:

- `raw_outputs.jsonl`
- `judged_results.jsonl`
- `metrics.csv`
- `summary.md`

## Metrics

`metrics.csv` contains:

```text
model_name,task_id,task_type,prompt_tokens,completion_tokens,total_tokens,latency_sec,tokens_per_second,expected_answer,model_output,correct,error,cost_estimate
```

Token counts are recorded when the endpoint returns usage data. Cost estimates come from model config pricing fields, not hardcoded provider prices.

## Scope

In scope:

- API model calls through configurable OpenAI-compatible endpoints.
- Local model endpoints that expose an OpenAI-compatible `/chat/completions` API.
- Running the same tasks across multiple models.
- Recording latency, token counts, tokens per second, accuracy, errors, and cost estimates.
- Generating CSV metrics and Markdown summaries.

Out of scope:

- Training or retraining.
- Checkpoint recovery.
- LoRA adapter restoration.
- Debugging old training scripts.
- Hardware purchases without benchmark evidence.

## Legacy Files

Directories such as `TrainStages/`, `MergeStages/`, `LoraAdapters/`, and `AdapterBackups/` are not part of the active harness. They remain in place to preserve historical context, but the new `llm_eval/` package does not import or execute them.
