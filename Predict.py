import os, re, time, uuid, argparse, subprocess, json, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# User-selectable base and lora (empty string = default behavior)
CUSTOM_BASE = "LoraAdapters/merged_stage3"
CUSTOM_LORA = "LoraAdapters/stage4-lora"

# ============================================================
# Paths / Config
# ============================================================

ROOT = Path(__file__).resolve().parent
LORA_DIR = ROOT / "LoraAdapters"
OFFLOAD_DIR = ROOT / "Offload"
RAG_PIPELINE = ROOT / "Rag" / "Pipelines"
FINAL_RAG_PROMPT = ROOT / "Rag" / "Cache" / "Retrievals" / "final_prompt.txt"

DEFAULT_BASES = [
    "merged_stage4",
    "merged_stage3",
    "merged_stage2",
    "merged_stage1",
]

# ============================================================
# Utilities
# ============================================================

def exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Missing {label}: {path}")

def should_use_rag(user_text: str) -> bool:
    text = user_text.strip().lower()

    # Never RAG on greetings or tiny inputs
    if len(text.split()) <= 2:
        return False

    if text in {"hi", "hello", "hey", "yo"}:
        return False

    return True

STOP_MARKERS = [
    "\nUser:", "\nAssistant:", "\nDuk:",
    "\nUser says:", "\nDuk says:",
    "\nUser asks:", "\nDuk replies:", "\nDuk reply:",
    "User:", "Assistant:", "Duk:",
    "User says:", "Duk says:", "User asks:", "Duk replies:",
]


def truncate_on_markers(text: str) -> str:
    # Cut off if the model starts inventing new turns
    idxs = [text.find(m) for m in STOP_MARKERS if text.find(m) != -1]
    if idxs:
        text = text[:min(idxs)]
    return text.strip()

def is_chaos(text: str) -> bool:
    text = text.lower()
    return (
        re.search(r"(.)\1{3,}", text) or
        re.search(r"(we+|du+k|die){2,}", text) or
        (len(text.split()) >= 6 and len(set(text.split())) <= 3)
    )

# ============================================================
# RAG
# ============================================================

def run_rag(query: str):
    router = RAG_PIPELINE / "Router.py"
    retriever = RAG_PIPELINE / "Retrieve.py"
    packer = RAG_PIPELINE / "PackPrompt.py"

    # 1) route (writes routed.json)
    subprocess.run([sys.executable, str(router), query], check=True)

    route_file = ROOT / "Rag" / "Cache" / "Retrievals" / "routed.json"
    route_info = json.loads(route_file.read_text())

    if route_info["route"] == "none":
        return None, route_info

    # 2) retrieve (writes last_query.json)
    subprocess.run([sys.executable, str(retriever)], check=True)

    # 3) pack (writes final_prompt.txt)
    subprocess.run([sys.executable, str(packer)], check=True, stdout=subprocess.DEVNULL)

    return FINAL_RAG_PROMPT.read_text(), route_info




# ============================================================
# Prompting
# ============================================================

DUK_SYSTEM = (
    "System: You are Duk.\n"
    "You speak in broken, chaotic phrases.\n"
    "Answer the user directly.\n"
    "Do not invent dialogue or additional speakers.\n"
    "Do not write lines starting with 'User:' or 'Duk:' or 'Assistant:'.\n"
    "One response only.\n\n"
)

def build_prompt(history, neutral=False):
    convo = "\n".join(f"{r}: {t}" for r, t in history)
    if neutral:
        return f"{convo}\nAssistant:"
    return f"{DUK_SYSTEM}{convo}\nAssistant:"


# ============================================================
# Model Loading
# ============================================================

def load_model(base: str, adapter: str | None):
    print(f"[INFO] Base model: {base}")
    print(f"[INFO] Adapter: {adapter or 'none'}")

    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant,
        device_map="auto",
        dtype=torch.float16,
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)

    tokenizer = load_tokenizer(base)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer

def load_tokenizer(base: str):
    try:
        return AutoTokenizer.from_pretrained(base)
    except Exception:
        fallback = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"[WARN] Tokenizer not found in base; falling back to {fallback}")
        return AutoTokenizer.from_pretrained(fallback)

# ============================================================
# Generation
# ============================================================

def generate(model, tokenizer, prompt, **overrides):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # defaults
    params = dict(
        max_new_tokens=128,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,              # default to FACT behavior
        no_repeat_ngram_size=3,       # helps kill loops like "Demply"
    )
    params.update(overrides)

    # only attach sampling knobs if sampling is enabled
    if params.get("do_sample", False):
        params.setdefault("temperature", 0.8)
        params.setdefault("top_p", 0.9)

    with torch.no_grad():
        out = model.generate(**inputs, **params)

    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def postprocess(text: str) -> str:
    text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)
    text = re.sub(r"\w+://\S+", "", text)
    return text.strip()


# ============================================================
# Inference Policy
# ============================================================

def infer(model, tokenizer, history, *, neutral=False, use_rag=True):
    rag_prompt = None  
    route_info = None

    user_text = history[-1][1]

    # ---------------- Chaos path ----------------
    if is_chaos(user_text):
        print("[DEBUG] Chaos detected → bypassing RAG and history")

        prompt = f"""
    <SystemFacts>
    You are Duk.
    You speak simply.
    You avoid long explanations.
    You do not form new opinions during noise.
    </SystemFacts>

    User: {user_text}
    Assistant:
    """

        return generate(
            model, tokenizer, prompt,
            do_sample=True,
            max_new_tokens=24,
            temperature=1.1,
            top_p=0.85,
            repetition_penalty=1.05,
        )


    # ---------------- RAG gating ----------------
    if use_rag and should_use_rag(user_text):
        # Use only the last user turn for routing/retrieval to avoid confusing the RAG pipeline
        rag_prompt, route_info = run_rag(user_text)


        if rag_prompt:
            print("[DEBUG] RAG USED ✅")
            print(f"[DEBUG]   Route: {route_info['route']}")
            print(f"[DEBUG]   Reason: {route_info.get('reason', '')}")
            print(f"[DEBUG]   Final prompt: {FINAL_RAG_PROMPT}")
            prompt = rag_prompt
        else:
            print("[DEBUG] RAG skipped ❌ (no matching index)")
            prompt = build_prompt(history, neutral)

    else:
        print("[DEBUG] RAG disabled ❌ (policy)")
        prompt = build_prompt(history, neutral)

    if rag_prompt:
        raw = generate(
            model,
            tokenizer,
            prompt,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=128,
            repetition_penalty=1.05,
        )
    else:
        raw = generate(
            model, tokenizer, prompt,
            do_sample=True,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )


    return truncate_on_markers(postprocess(raw))



# ============================================================
# CLI
# ============================================================

def resolve_base(name: str | None) -> str:
    candidates = []

    if name:
        candidates.extend([
            Path(name),
            ROOT / name,
            LORA_DIR / name,
        ])

    for cand in DEFAULT_BASES:
        candidates.extend([
            ROOT / cand,
            LORA_DIR / cand,
        ])

    for p in candidates:
        if p and p.exists():
            return str(p.resolve())

    if name:
        print(f"[WARN] Base '{name}' not found locally; treating as remote id")
        return name

    raise FileNotFoundError(
        "No local merged model found and no base model specified"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", nargs="?", help="Base or merged checkpoint")
    parser.add_argument("adapter", nargs="?", help="LoRA adapter path")
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    args = parser.parse_args()


    # Use CUSTOM_BASE and CUSTOM_LORA if set, else fall back to CLI args
    if CUSTOM_BASE:
        base = resolve_base(CUSTOM_BASE)
    else:
        base = resolve_base(args.base)

    if CUSTOM_LORA:
        lora_path = (ROOT / CUSTOM_LORA) if CUSTOM_LORA else None
        adapter = str(lora_path.resolve()) if (lora_path and lora_path.exists()) else None
    else:
        adapter = args.adapter if args.adapter and Path(args.adapter).exists() else None

    model, tokenizer = load_model(base, adapter)

    history = []
    print("\n✅ Chat ready. Type 'exit' to quit.\n")

    while True:
        user = input("User: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        history.append(("User", user))
        history[:] = history[-8:]

        print(f"[DEBUG] id={uuid.uuid4().hex} ts={time.time():.3f}")

        reply = infer(
            model,
            tokenizer,
            history,
            neutral=args.neutral,
            use_rag=not args.no_rag,
        )

        print("Assistant:", reply, "\n")
        history.append(("Assistant", reply))
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
