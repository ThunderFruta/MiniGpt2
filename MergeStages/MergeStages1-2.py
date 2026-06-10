import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base for merging stage-2 should be the already-merged stage-1 model
repo_root = Path(__file__).resolve().parent.parent
BASE_MODEL = repo_root / "LoraAdapters" / "merged_stage1"
# Name of the directory under `LoraAdapters/` where Stage-2 adapter was saved
STAGE2_ADAPTER = "stage2-lora"
MERGED_OUTPUT = repo_root / "LoraAdapters" / "merged_stage2"
OFFLOAD_DIR = "offload_merge"

# Resolve adapter path: accept absolute/relative provided path, or look under MiniGpt2/LoraAdapters
adapter_path = Path(STAGE2_ADAPTER)
if not adapter_path.exists():
    alt = repo_root / "LoraAdapters" / STAGE2_ADAPTER
    if alt.exists():
        adapter_path = alt
    else:
        raise FileNotFoundError(
            f"Stage-2 adapter not found. Checked: {STAGE2_ADAPTER} and {alt.as_posix()}"
        )

# Informative path
print(f"Found Stage-2 adapter at: {adapter_path}")
print("Loading base model on CPU (this may take some time and RAM)...")
# load the base model on CPU to avoid GPU memory pressure
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)

print("Attaching Stage-2 adapter to base model (CPU)...")
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

print(f"Saving merged base to `{MERGED_OUTPUT}` (this will be used for next-stage training)...")
# Save the merged base model
model.save_pretrained(MERGED_OUTPUT)
print("Done. You can now use the merged model as the new base for the next training stage.")
print(f"Example: in your training script set `BASE_MODEL = '{MERGED_OUTPUT}'` and keep quantization if desired.")
