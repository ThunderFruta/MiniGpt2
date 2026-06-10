#!/usr/bin/env python3

import json
import sys
from pathlib import Path

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "Rag" / "Cache" / "Retrievals"
CACHE.mkdir(parents=True, exist_ok=True)

# ---------------- Domain Gate ----------------

FRUIT_WARS_TERMS = {
    "fruit wars",
    "fruitwars",
    "thunderfruta",
    "fruit relic",
    "banana faction",
    "orange faction",
    "frukas",
}

def is_fruit_wars_query(q: str) -> bool:
    return any(term in q for term in FRUIT_WARS_TERMS)

# ---------------- Heuristics ----------------

CHAT_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "how are you",
    "do you like",
    "what do you think",
}

# Canon = rules / mechanics / definitions
CANON_INTENT = {
    "what is",
    "define",
    "rules",
    "rank",
    "ranks",
    "relic",
    "relics",
    "currency",
    "system",
    "canon",
}

# Knowledge = history / explanation
KNOWLEDGE_INTENT = {
    "history",
    "origin",
    "background",
    "timeline",
    "why",
    "how did",
    "explain",
    "story",
}

DEFAULT_TOP_K = 6

# ---------------- Routing ----------------

def route(query: str):
    q = query.lower().strip()

    if not q:
        route_name = "none"
        reason = "empty query"

    elif any(p in q for p in CHAT_PATTERNS):
        route_name = "none"
        reason = "casual chat"

    elif not is_fruit_wars_query(q):
        route_name = "none"
        reason = "non–fruit wars query (RAG disabled)"

    else:
        canon_hits = sum(k in q for k in CANON_INTENT)
        knowledge_hits = sum(k in q for k in KNOWLEDGE_INTENT)

        if canon_hits and knowledge_hits:
            route_name = "both"
            reason = "fruit wars: mixed canon + knowledge intent"
        elif canon_hits:
            route_name = "canon"
            reason = "fruit wars: rules / definitions"
        elif knowledge_hits:
            route_name = "knowledge"
            reason = "fruit wars: history / explanation"
        else:
            route_name = "canon"
            reason = "fruit wars: default canon"

    output = {
        "query": query,
        "route": route_name,
        "reason": reason,
        "top_k": DEFAULT_TOP_K,
    }

    with open(CACHE / "routed.json", "w") as f:
        json.dump(output, f, indent=2)

    return output

# ---------------- Run ----------------

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(route(q), indent=2))
