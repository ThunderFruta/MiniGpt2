#!/usr/bin/env python3

import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]
INDEXES = ROOT / "Rag" / "Indexes"
CACHE = ROOT / "Rag" / "Cache" / "Retrievals"

CANON_INDEX = INDEXES / "Canon" / "Faiss"
KNOWLEDGE_INDEX = INDEXES / "Knowledge" / "Faiss"

CACHE.mkdir(parents=True, exist_ok=True)

_model = None

# ---------------- Helpers ----------------
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def load_index(path):
    index = faiss.read_index(str(path / "index.faiss"))
    metadata = json.loads((path / "metadata.json").read_text())

    if index.ntotal != len(metadata):
        raise RuntimeError("FAISS index / metadata size mismatch")

    return index, metadata


def retrieve(query, index, metadata, top_k=5):
    model = get_model()

    query_vec = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(
        np.asarray(query_vec, dtype="float32"),
        top_k
    )

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        src = metadata[idx].get("source", "")

        # --- SIMPLE WORKAROUND ---
        if "SystemFacts" in src:
            continue
        # ------------------------

        results.append({
            "score": float(score),
            "text": metadata[idx]["text"],
            "source": src
        })

    return results



# ---------------- Run (debug only) ----------------
if __name__ == "__main__":
    query = "What is the fruit relic?"

    canon_index, canon_meta = load_index(CANON_INDEX)
    knowledge_index, knowledge_meta = load_index(KNOWLEDGE_INDEX)

    output = {
        "query": query,
        "canon": retrieve(query, canon_index, canon_meta),
        "knowledge": retrieve(query, knowledge_index, knowledge_meta),
    }

    out_file = CACHE / "last_query.json"
    with out_file.open("w") as f:
        json.dump(output, f, indent=2)

    print("✅ Retrieval complete")
    print(json.dumps(output, indent=2))
