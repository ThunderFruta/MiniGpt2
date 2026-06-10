import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
# Name of the directory under `LoraAdapters/` where Stage-1 adapter was saved
# Historically this script expected `neo-lora-final`. Newer training saves Stage-1
# adapters as `stage1-lora` to match the other stages. Accept the new name here.
STAGE1_ADAPTER = "stage1-lora"
repo_root = Path(__file__).resolve().parent.parent
MERGED_OUTPUT = repo_root / "LoraAdapters" / "merged_stage1"
OFFLOAD_DIR = "offload_merge"

# Resolve adapter path: accept absolute/relative provided path, or look under MiniGpt2/LoraAdapters
adapter_path = Path(STAGE1_ADAPTER)
if not adapter_path.exists():
    alt = repo_root / "LoraAdapters" / STAGE1_ADAPTER
    if alt.exists():
        adapter_path = alt
    else:
        raise FileNotFoundError(
            f"Stage-1 adapter not found. Checked: {STAGE1_ADAPTER} and {alt.as_posix()}"
        )

# Informative path
print(f"Found Stage-1 adapter at: {adapter_path}")
print("Loading base model on CPU (this may take some time and RAM)...")
# load the base model on CPU to avoid GPU memory pressure
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)

print("Attaching Stage-1 adapter to base model (CPU)...")
model = PeftModel.from_pretrained(model, str(adapter_path), device_map={"": "cpu"})

print("Merging LoRA adapter into base weights (in-place)...")
# Merge and unload adapter weights into the base model weights
try:
    # Preferred API
    model = model.merge_and_unload()
except Exception:
    try:
        # Alternative helper name
        model.merge_and_unload()
    except Exception as e:
        print("Failed to call merge_and_unload():", e)
        print("The installed PEFT version may differ. Please upgrade peft or merge manually.")
        raise

print(f"Saving merged base to `{MERGED_OUTPUT}` (this will be used for 4-bit training)...")
# Save the merged base model
model.save_pretrained(MERGED_OUTPUT)
print("Done. You can now use the merged model as the new base for 4-bit training.")
print(f"Example: in `TrainStage2.py` set `BASE_MODEL = '{MERGED_OUTPUT}'` and keep quantization enabled.")