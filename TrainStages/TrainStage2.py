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
# Config
# -------------------------
BASE_MODEL   = "merged_stage1"              # your merged Stage-1 base folder
# Allow overriding the base model via env var (useful if merged model lives elsewhere)
BASE_MODEL = os.environ.get("BASE_MODEL", BASE_MODEL)
repo_root = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(repo_root, ".."))
LORA_DIR     = os.path.join(repo_root, "LoraAdapters")
os.makedirs(LORA_DIR, exist_ok=True)
STAGE2_OUT   = os.path.join(LORA_DIR, "stage2-lora")                # output adapter dir (in LoraAdapters)
DATA_PATH    = os.path.join(repo_root, "Voices", "Stage2.jsonl")
TOKENIZER_BASE = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_BASE,
    use_fast=True,
)
MAX_LEN      = 96
EPOCHS       = 5
LR           = 5e-5
BATCH        = 1
GRAD_ACC     = 4

# -------------------------
# Sanity checks
# -------------------------
def must_exist(path, msg):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}: {path}")

# Resolve BASE_MODEL to local locations if necessary (prefer MiniGpt2/LoraAdapters)
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
must_exist(DATA_PATH, "Dataset not found")
if os.path.getsize(DATA_PATH) == 0:
    raise FileNotFoundError(f"Dataset is empty (0 bytes): {DATA_PATH}")

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

    # mask everything up through "Assistant:"
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
# 4-bit load
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
# Stage-2 LoRA
# -------------------------
stage2_config = LoraConfig(
    r=8,                     # smaller = less VRAM
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, stage2_config)
model.print_trainable_parameters()
model.train()

# -------------------------
# Train
# -------------------------
args = TrainingArguments(
    output_dir=STAGE2_OUT,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=5,
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
model.save_pretrained(STAGE2_OUT)
print("\n✅ Stage-2 training complete.")
