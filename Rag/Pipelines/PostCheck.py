#!/usr/bin/env python3

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # MiniGpt2/
INDEXES = ROOT / "Rag" / "Indexes"

def check_index(name):
    print(f"🔍 Checking {name} index")

    faiss_dir = INDEXES / name / "Faiss"
    index_file = faiss_dir / "index.faiss"
    meta_file = faiss_dir / "metadata.json"

    if not index_file.exists():
        raise RuntimeError(f"{name}: index.faiss missing")

    if not meta_file.exists():
        raise RuntimeError(f"{name}: metadata.json missing")

    with meta_file.open() as f:
        meta = json.load(f)

    if not meta:
        raise RuntimeError(f"{name}: metadata empty")

    count = len(meta)
    print(f"✅ {name}: {count} vectors")

def main():
    print("\nPOST-BUILD RAG CHECK")
    print("====================")

    try:
        check_index("Canon")
        check_index("Knowledge")
    except Exception as e:
        print(f"\n❌ POST CHECK FAILED: {e}")
        sys.exit(1)

    print("\n✅ POST CHECK PASSED")
    print("RAG indexes are valid and ready.")

if __name__ == "__main__":
    main()
