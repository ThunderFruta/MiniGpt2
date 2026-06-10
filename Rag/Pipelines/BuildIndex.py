#!/usr/bin/env python3

import json
from pathlib import Path
import faiss
import numpy as np

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]

CACHE = ROOT / "Rag" / "Cache" / "Embeddings"
INDEXES = ROOT / "Rag" / "Indexes"

CANON_CACHE = CACHE / "Canon"
KNOWLEDGE_CACHE = CACHE / "Knowledge"

CANON_INDEX = INDEXES / "Canon" / "Faiss"
KNOWLEDGE_INDEX = INDEXES / "Knowledge" / "Faiss"

CANON_INDEX.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_INDEX.mkdir(parents=True, exist_ok=True)

# ---------------- Helpers ----------------
def build_index(src_folder, out_folder, name):
    vectors = []
    metadata = []

    for file in src_folder.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        for row in data:
            vectors.append(row["embedding"])
            metadata.append({
                "source": row["source"],
                "chunk_id": row["chunk_id"],
                "text": row["text"]
            })

    if not vectors:
        print(f"⚠ No vectors found for {name}")
        return

    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise RuntimeError(f"{name}: inconsistent embedding dimensions")

    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(out_folder / "index.faiss"))

    with open(out_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✔ {name} index built ({len(vectors)} vectors)")

# ---------------- Run ----------------
if __name__ == "__main__":
    build_index(CANON_CACHE, CANON_INDEX, "Canon")
    build_index(KNOWLEDGE_CACHE, KNOWLEDGE_INDEX, "Knowledge")
