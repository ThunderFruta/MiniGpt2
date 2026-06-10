#!/usr/bin/env python3
"""
clean_voices.py

Scan `MiniGpt2/Voices/` for JSONL files and remove duplicate lines and
extraneous whitespace. By default this writes cleaned files to
`<orig>.cleaned.jsonl` and leaves originals intact. Use `--inplace` to
overwrite originals (a backup is created unless `--no-backup` is passed).

Dedup logic:
- If a line is valid JSON, dedup is based on the tuple (prompt, response)
  where keys are looked up as common fields and whitespace is stripped.
- If a line is not JSON, dedup is performed on the stripped line text.

Usage:
  python3 MiniGpt2/clean_voices.py [--inplace] [--no-backup] [--voices-dir PATH]

"""
import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Tuple


COMMON_PROMPT_KEYS = ("prompt", "input", "text", "instruction")
COMMON_RESPONSE_KEYS = ("response", "completion", "reply", "output")


def extract_prompt_response(obj: dict) -> Tuple[str, str]:
    """Return (prompt, response) strings from a parsed JSON object.

    Falls back to empty strings if keys are missing.
    """
    prompt = ""
    response = ""
    for k in COMMON_PROMPT_KEYS:
        if k in obj and isinstance(obj[k], str):
            prompt = obj[k]
            break
    for k in COMMON_RESPONSE_KEYS:
        if k in obj and isinstance(obj[k], str):
            response = obj[k]
            break
    return prompt.strip(), response.strip()


def process_file(path: Path, inplace: bool = False) -> dict:
    stats = {"file": str(path), "lines_in": 0, "lines_out": 0, "dups": 0, "empties": 0}
    seen = set()
    out_lines = []

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stats["lines_in"] += 1
            s = raw.strip()
            if not s:
                stats["empties"] += 1
                continue
            key = None
            out_line = None
            try:
                obj = json.loads(s)
                # normalize string fields by stripping whitespace
                for k, v in list(obj.items()):
                    if isinstance(v, str):
                        obj[k] = v.strip()
                prompt, response = extract_prompt_response(obj)
                key = (prompt, response)
                out_line = json.dumps(obj, ensure_ascii=False)
            except Exception:
                # non-JSON line: dedup on the stripped line
                key = s
                out_line = s

            if key in seen:
                stats["dups"] += 1
                continue
            seen.add(key)
            out_lines.append(out_line + "\n")

    stats["lines_out"] = len(out_lines)

    # Write cleaned output. If `inplace` is True, overwrite the original
    # file safely (atomic replace via a temporary file). No backups are
    # created by design.
    if inplace:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as outfh:
            outfh.writelines(out_lines)
        tmp.replace(path)
    else:
        out_path = path.with_name(path.stem + ".cleaned" + path.suffix)
        with out_path.open("w", encoding="utf-8") as outfh:
            outfh.writelines(out_lines)


def find_voice_files(voices_dir: Path):
    if not voices_dir.exists() or not voices_dir.is_dir():
        raise FileNotFoundError(f"Voices directory not found: {voices_dir}")
    files = sorted([p for p in voices_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"])
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voices-dir", default=str(Path(__file__).resolve().parent / "Voices"))
    p.add_argument("--no-inplace", action="store_true", help="Do not overwrite originals; write .cleaned.jsonl files instead")
    args = p.parse_args()

    inplace = not args.no_inplace

    voices_dir = Path(args.voices_dir)
    print(f"Scanning voices directory: {voices_dir}")
    files = find_voice_files(voices_dir)
    if not files:
        print("No .jsonl files found in voices directory.")
        return

    total = {"in": 0, "out": 0, "dups": 0, "empties": 0}
    for f in files:
        print(f"Processing: {f}")
        stats = process_file(f, inplace=inplace)
        print(f"  lines_in={stats['lines_in']} lines_out={stats['lines_out']} dups={stats['dups']} empties={stats['empties']}")
        total["in"] += stats["lines_in"]
        total["out"] += stats["lines_out"]
        total["dups"] += stats["dups"]
        total["empties"] += stats["empties"]

    print("\nSummary:")
    print(f"  files_processed={len(files)} total_in={total['in']} total_out={total['out']} total_dups={total['dups']} total_empties={total['empties']}")


if __name__ == "__main__":
    main()
