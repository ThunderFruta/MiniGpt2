import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import PeftModel

# -------------------------
# Config
# -------------------------
repo_root = Path(__file__).resolve().parent.parent
LORA_ROOT = repo_root / "LoraAdapters"

# Merge Stage-4: attach `stage4-lora` onto `merged_stage3` and produce `merged_stage4`.
BASE_MODEL = LORA_ROOT / "merged_stage3"
STAGE3_ADAPTER = LORA_ROOT / "stage4-lora"

MERGED_OUTPUT = LORA_ROOT / "merged_stage4"

DEVICE_MAP = {"": "cpu"}  # CPU-only merge = safe

# -------------------------
# Sanity checks
# -------------------------
def must_exist(p: Path, name: str):
	if not p.exists():
		raise FileNotFoundError(f"{name} not found: {p}")

must_exist(BASE_MODEL, "Base model (merged_stage2)")
must_exist(STAGE3_ADAPTER, "Stage-3 LoRA")

print("[INFO] Base:", BASE_MODEL)
print("[INFO] Stage-3:", STAGE3_ADAPTER)

# -------------------------
# Load base model
# -------------------------
print("[INFO] Loading base model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
	BASE_MODEL,
	device_map=DEVICE_MAP,
	torch_dtype=torch.float16,
	low_cpu_mem_usage=True,
)

# -------------------------
# Attach Stage-3 and merge into a new merged_stage3 base.
# -------------------------
print("[INFO] Attaching Stage-4 adapter and merging into base...")
model = PeftModel.from_pretrained(
	model,
	STAGE3_ADAPTER,
	device_map=DEVICE_MAP,
)
try:
	model = model.merge_and_unload()
except Exception:
	try:
		model.merge_and_unload()
	except Exception as e:
		print("[ERROR] Failed to merge Stage-3 adapter:", e)
		raise

# -------------------------
# Save merged model
# -------------------------
MERGED_OUTPUT.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Saving merged model to {MERGED_OUTPUT}")
model.save_pretrained(MERGED_OUTPUT)

print("\n✅ Merge complete.")
print("Use this as your final base model:")
print(f"   BASE_MODEL = '{MERGED_OUTPUT}'")

