#!/usr/bin/env python3

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL = ROOT / "Rag" / "Cache" / "Retrievals" / "last_query.json"

def tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))

def score(chunk_tokens, query_tokens):
    # simple keyword overlap
    return len(chunk_tokens & query_tokens)

def main():
    if not RETRIEVAL.exists():
        print("⚠️ No retrieval file found, skipping rerank")
        return

    with RETRIEVAL.open() as f:
        data = json.load(f)

    query = data.get("query", "")
    results = data.get("results", [])

    if not query or not results:
        print("⚠️ Empty query or results, skipping rerank")
        return

    query_tokens = tokenize(query)

    for r in results:
        text = r.get("text", "")
        r["_rerank_score"] = score(tokenize(text), query_tokens)

    # sort by rerank score (desc), keep original order as tiebreaker
    results.sort(key=lambda x: x["_rerank_score"], reverse=True)

    # cleanup helper field
    for r in results:
        r.pop("_rerank_score", None)

    data["results"] = results

    with RETRIEVAL.open("w") as f:
        json.dump(data, f, indent=2)

    print("✅ Rerank complete (keyword overlap)")

if __name__ == "__main__":
    main()
