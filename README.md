# MiniGpt2 LoRA RAG

This repository is an old experimental snapshot of a MiniGpt2 LoRA and RAG project. It is preserved as research/prototype code, not as a polished package or production-ready application.

The project combines staged LoRA training scripts, merge scripts, inference helpers, a simple RAG pipeline, and Flask app entrypoints for local or tunneled serving. The original experiment had instability and memory constraints, and generated artifacts are intentionally excluded from this public repo.

## What Is Included

- `Predict.py` and `MassPredict.py`: inference helpers for loading merged bases, optional LoRA adapters, and RAG-assisted prompting.
- `Reset&Train.py`: controller script for the staged LoRA training and merge pipeline.
- `TrainStages/`: stage-specific LoRA training scripts.
- `MergeStages/`: scripts for merging staged adapters into the next base.
- `Rag/Pipelines/`: document ingest, indexing, retrieval, routing, reranking, prompt packing, and post-check helpers.
- `Rag/Config/` and `Rag/Prompts/`: lightweight configuration and prompt templates.
- `Apps&Cloudflare/`: Flask app entrypoints for local and Cloudflare-tunneled use.

## What Is Not Included

The public repo intentionally ignores generated or machine-local files:

- `.venv/`
- `Rag/Cache/`
- `AdapterBackups/`
- `LoraAdapters/`
- `Rag/Indexes/`
- `Rag/Docs/`
- `Rag/Memory/`
- model weight and checkpoint files such as `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, and `*.ckpt`

That means the code will not run out of the box without recreating or restoring the required models, adapters, datasets, and RAG documents.

## Expected Shape

The older local working tree expected directories like:

```text
LoraAdapters/
Rag/Docs/
Rag/Indexes/
Rag/Cache/
Voices/
Prediction/
```

Some scripts also assume local model paths such as `LoraAdapters/merged_stage3` and `LoraAdapters/stage4-lora`. Those paths are excluded here and need to be recreated or changed before running inference.

## Rough Workflow

1. Prepare datasets under `Voices/`.
2. Train missing or new LoRA stages with:

   ```bash
   python "Reset&Train.py" --dry-run
   python "Reset&Train.py" --action missing
   ```

3. Rebuild RAG indexes after adding documents:

   ```bash
   python UpdateRag.py
   ```

4. Run inference once model and adapter paths exist:

   ```bash
   python Predict.py
   ```

## Notes

This is old prototype code, so expect hardcoded paths, missing dependency declarations, instability, memory constraints, and assumptions from the original local environment. Treat it as a reference snapshot for the LoRA/RAG experiment rather than a clean reusable library.
