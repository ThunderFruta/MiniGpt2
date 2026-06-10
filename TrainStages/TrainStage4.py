import os
# MUST be before torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import warnings
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore", message="torch.utils.checkpoint")

# -------------------------
# Config (Stage-4 training)
# -------------------------
BASE_MODEL   = "merged_stage2"   # ← assume you merged stage 2 already
repo_root = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(repo_root, ".."))
LORA_DIR     = os.path.join(repo_root, "LoraAdapters")
STAGE4_OUT   = os.path.join(LORA_DIR, "stage4-lora")
DATA_PATH    = os.path.join(repo_root, "Voices", "Stage4.jsonl")
TOKENIZER_BASE = "meta-llama/Meta-Llama-3-8B-Instruct"

MAX_LEN      = 128               # longer context helps length conditioning
EPOCHS       = 6                 # gentle refinement
LR           = 8e-5              # lower LR to avoid overwriting canon
BATCH        = 1
GRAD_ACC     = 4

os.makedirs(LORA_DIR, exist_ok=True)

# -------------------------
# Sanity checks
# -------------------------
def must_exist(path, msg):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}: {path}")

# Allow overriding the base model via env var and resolve local locations
BASE_MODEL = os.environ.get("BASE_MODEL", BASE_MODEL)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_candidates = [
    BASE_MODEL,
    os.path.join(script_dir, BASE_MODEL),
    os.path.join(repo_root, BASE_MODEL),
    os.path.join(repo_root, "LoraAdapters", BASE_MODEL),
    os.path.join(script_dir, "LoraAdapters", BASE_MODEL),
]
resolved_base = None
for cand in base_candidates:
    if cand and os.path.exists(cand):
        resolved_base = cand
        break
if resolved_base:
    BASE_MODEL = resolved_base

must_exist(BASE_MODEL, "Merged base model folder not found")
must_exist(DATA_PATH, "Stage 4 dataset not found")
if os.path.getsize(DATA_PATH) == 0:
    raise FileNotFoundError(f"Dataset is empty: {DATA_PATH}")

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_BASE,
    use_fast=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_text(ex):
    return {"text": f"{ex['prompt']}\nAssistant: {ex['response']}"}

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

def make_labels(ex):
    input_ids = ex["input_ids"]
    labels = input_ids.copy()
    assistant_ids = tokenizer("Assistant:", add_special_tokens=False)["input_ids"]

    for i in range(len(input_ids) - len(assistant_ids) + 1):
        if input_ids[i:i+len(assistant_ids)] == assistant_ids:
            labels[:i+len(assistant_ids)] = [-100] * (i+len(assistant_ids))
            break

    return {"input_ids": input_ids, "labels": labels}

# -------------------------
# Dataset
# -------------------------
ds = load_dataset("json", data_files=DATA_PATH)["train"]
ds = ds.map(build_text, remove_columns=ds.column_names)
ds = ds.map(tokenize, batched=True, remove_columns=["text"])
ds = ds.map(make_labels)

# -------------------------
# 4-bit base load
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# -------------------------
# Stage-4 LoRA (behavior only)
# -------------------------
stage4_config = LoraConfig(
    r=4,                     # smaller = subtle behavior shaping
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, stage4_config)
model.print_trainable_parameters()
model.train()

# -------------------------
# Train
# -------------------------
args = TrainingArguments(
    output_dir=STAGE4_OUT,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=default_data_collator,
)

trainer.train()
model.save_pretrained(STAGE4_OUT)

print("\n✅ Stage-4 training complete.")
