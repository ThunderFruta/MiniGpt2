#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent
PIPELINES = ROOT / "Rag" / "Pipelines"

def run(step_name, cmd):
    print(f"\n🔹 {step_name}")
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"✅ {step_name} complete")
    except subprocess.CalledProcessError:
        print(f"❌ {step_name} failed")
        sys.exit(1)

def main():
    print("\nRAG UPDATE SCRIPT")
    print("====================")

    # 1) Ingest docs → chunks
    run(
        "Ingesting documents",
        [PYTHON, str(PIPELINES / "IngestDocs.py")]
    )

    # 2) Build embeddings + FAISS indexes
    run(
        "Building FAISS indexes",
        [PYTHON, str(PIPELINES / "BuildIndex.py")]
    )

    # 3) Post-build validation (optional but correct)
    postcheck = PIPELINES / "PostCheck.py"
    if postcheck.exists():
        run(
            "Post-build validation",
            [PYTHON, str(postcheck)]
        )

    print("\n🎉 RAG BUILD COMPLETE")
    print("RAG is compiled and ready for queries.")
    print("Next step: run Predict.py")

if __name__ == "__main__":
    main()
