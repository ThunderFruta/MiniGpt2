import torch
import warnings
from pathlib import Path
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore", message="torch.utils.checkpoint")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token

def build_text(example):
    return {
        "text": f"{example['prompt']}\nAssistant: {example['response']}"
    }

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

def make_labels(example):
    input_ids = example["input_ids"]
    labels = input_ids.copy()

    assistant_ids = tokenizer("Assistant:", add_special_tokens=False)["input_ids"]

    for i in range(len(input_ids) - len(assistant_ids)):
        if input_ids[i:i+len(assistant_ids)] == assistant_ids:
            labels[:i+len(assistant_ids)] = [-100] * (i+len(assistant_ids))
            break

    return {"input_ids": input_ids, "labels": labels}

# Use the repository root as base so paths point to top-level `Voices/` and `LoraAdapters/`
repo_root = Path(__file__).resolve().parent.parent

# Dataset (line-delimited JSONL)
dataset = load_dataset("json", data_files=str(repo_root / "Voices" / "Stage1.jsonl"))["train"]
dataset = dataset.map(build_text, remove_columns=dataset.column_names)
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
dataset = dataset.map(make_labels)

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.train()

# Training
LORA_DIR = repo_root / "LoraAdapters"
LORA_DIR.mkdir(parents=True, exist_ok=True)

# Build TrainingArguments in a backwards-compatible way: some older
# "transformers" releases don't accept the `evaluation_strategy` kwarg.
base_args = dict(
    output_dir=str(LORA_DIR / "stage1-lora"),
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    fp16=True,
    num_train_epochs=5,
    logging_steps=20,
    save_steps=500,
)

try:
    # Preferred: include evaluation_strategy when available
    training_args = TrainingArguments(**{**base_args, "evaluation_strategy": "no"})
except TypeError:
    # Some transformers versions (e.g., 4.57.3) use a different kwarg name
    # (`eval_strategy`). Try that before falling back to the base args.
    try:
        training_args = TrainingArguments(**{**base_args, "eval_strategy": "no"})
    except TypeError:
        # Final fallback for very old releases
        training_args = TrainingArguments(**base_args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,
)

trainer.train()
# ensure final adapter dir exists under LoraAdapters
final_dir = LORA_DIR / "stage1-lora"
final_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(final_dir))
