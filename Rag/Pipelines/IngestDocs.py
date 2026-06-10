#!/usr/bin/env python3

import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]

DOCS = ROOT / "Rag" / "Docs"
CACHE = ROOT / "Rag" / "Cache" / "Embeddings"

CANON_DOCS = DOCS / "Canon"
KNOWLEDGE_DOCS = DOCS / "Knowledge"

CANON_CACHE = CACHE / "Canon"
KNOWLEDGE_CACHE = CACHE / "Knowledge"

CANON_CACHE.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------- Load Config ----------------
with open(ROOT / "Rag" / "Config" / "Chunking.yaml") as f:
    chunk_cfg = yaml.safe_load(f)

with open(ROOT / "Rag" / "Config" / "Embedding.yaml") as f:
    embed_cfg = yaml.safe_load(f)

CHUNK_SIZE = chunk_cfg["ChunkSize"]
OVERLAP = chunk_cfg["ChunkOverlap"]

if OVERLAP >= CHUNK_SIZE:
    raise ValueError("ChunkOverlap must be smaller than ChunkSize")

_model = None

# ---------------- Helpers ----------------
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(embed_cfg["EmbeddingModel"])
    return _model


def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def ingest_folder(src_folder, out_folder, label):
    print(f"Ingesting {label} docs...")
    model = get_model()

    # ✅ FIX 1: recurse into subfolders
    for file in src_folder.rglob("*.txt"):
        text = file.read_text(encoding="utf-8")
        text = " ".join(text.split())

        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        embeddings = model.encode(chunks, normalize_embeddings=True)

        # ✅ FIX 2: preserve relative path as source
        relative_source = file.relative_to(src_folder)

        data = []
        for i, chunk in enumerate(chunks):
            data.append({
                "source": str(relative_source),  # e.g. FruitWars/FWRelic.txt
                "chunk_id": i,
                "text": chunk,
                "embedding": embeddings[i].tolist()
            })

        out_file = out_folder / f"{relative_source.as_posix().replace('/', '__')}.json"
        with out_file.open("w") as f:
            json.dump(data, f, indent=2)

        print(f"  ✔ {relative_source} → {out_file.name}")

# ---------------- Run ----------------
if __name__ == "__main__":
    ingest_folder(CANON_DOCS, CANON_CACHE, "Canon")
    ingest_folder(KNOWLEDGE_DOCS, KNOWLEDGE_CACHE, "Knowledge")
    print("✅ Ingestion complete")
