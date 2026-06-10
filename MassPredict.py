#!/usr/bin/env python3
"""mass_predict.py"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Predict import load_model, run_inference
from BasePath import INPUT_JSONL, OUTPUT_JSONL


def process_file(
    model,
    tokenizer,
    input_path,
    output_path,
    *,
    neutral=False,
    max_new_tokens=None,
    sleep_between=0.0,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
    print(f"Found {total} lines to predict in {input_path}")

    start = time.time()

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Invalid JSON on line {idx}")
                continue

            prompt_text = obj.get("prompt") or obj.get("text") or obj.get("input")
            if not prompt_text:
                print(f"[WARN] Line {idx} missing prompt")
                continue

            history = [("User", prompt_text)]

            pred, raw = run_inference(
                model,
                tokenizer,
                history,
                neutral=neutral,
                max_new_tokens=max_new_tokens,
            )

            obj_out = dict(obj)
            obj_out["raw_generation"] = raw
            obj_out["prediction"] = pred
            fout.write(json.dumps(obj_out, ensure_ascii=False) + "\n")

            if idx % 10 == 0 or idx == total:
                elapsed = time.time() - start
                print(f"[{idx}/{total}] elapsed={elapsed:.1f}s pred={pred!r}")

            if sleep_between > 0:
                time.sleep(sleep_between)

    print(f"Done — predictions written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=INPUT_JSONL)
    parser.add_argument("--output", "-o", default=OUTPUT_JSONL)
    parser.add_argument("--base", default="merged_stage2")
    parser.add_argument("--adapter", default="stage3-lora")
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--no-8bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    use_8bit = not args.no_8bit
    script_dir = Path(__file__).resolve().parent

    base_path = args.base
    for cand in [args.base, script_dir / args.base, script_dir / "LoraAdapters" / args.base]:
        if os.path.exists(cand):
            base_path = str(cand)
            break

    adapter_path = None
    if not args.no_adapter:
        for cand in [
            args.adapter,
            script_dir / args.adapter,
            script_dir / "LoraAdapters" / args.adapter,
        ]:
            if os.path.exists(cand):
                adapter_path = str(cand)
                break

    print(f"Loading model: base={base_path} adapter={adapter_path} 8bit={use_8bit}")
    model, tokenizer = load_model(
        base_model_name=base_path,
        adapter_path=adapter_path,
        use_8bit=use_8bit,
    )

    if hasattr(model, "peft_config"):
        print("[INFO] LoRA attached:", list(model.peft_config.keys()))
    else:
        print("[WARN] No LoRA adapter attached")

    process_file(
        model,
        tokenizer,
        args.input,
        args.output,
        neutral=args.neutral,
        max_new_tokens=args.max_new_tokens,
        sleep_between=args.sleep,
    )


if __name__ == "__main__":
    main()
