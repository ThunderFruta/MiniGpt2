#!/usr/bin/env python3

import json
from pathlib import Path
import re
# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]

PROMPTS = ROOT / "Rag" / "Prompts"
CACHE = ROOT / "Rag" / "Cache" / "Retrievals"

SYSTEM_PROMPT = PROMPTS / "System.txt"
CANON_WRAPPER = PROMPTS / "CanonWrapper.txt"
KNOWLEDGE_WRAPPER = PROMPTS / "KnowledgeWrapper.txt"

ROUTED = CACHE / "routed.json"
RETRIEVED = CACHE / "last_query.json"
OUT = CACHE / "final_prompt.txt"


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_FACTS = (
    ROOT / "Rag" / "Docs" / "Canon" / "SystemFacts.txt"
).read_text(encoding="utf-8")


# ---------------- Load Static Prompts ----------------
system_text = SYSTEM_PROMPT.read_text()
canon_wrapper = CANON_WRAPPER.read_text()
knowledge_wrapper = KNOWLEDGE_WRAPPER.read_text()

# ---------------- Load Routing ----------------
routed = json.loads(ROUTED.read_text())
query = routed["query"]
route = routed["route"]

# ---------------- Load Retrieval ----------------
canon_context = ""
knowledge_context = ""

def clean_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r"\w+://\S+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip odd dataset markers
    text = re.sub(r"(Answer:|Question:|答:)", "", text, flags=re.IGNORECASE)
    return text.strip()


canon_context = ""
knowledge_context = ""

if RETRIEVED.exists():
    retrieved = json.loads(RETRIEVED.read_text())

    # ---------------- Canon ----------------
    if route in ("canon", "both"):
        canon_results = retrieved.get("canon", [])[:4]  # HARD CAP

        canon_chunks = [
            clean_text(r["text"])
            for r in canon_results
            if r.get("text")
        ]

        world_canon = "\n".join(canon_chunks)

        canon_context = (
            "AUTHORITATIVE CANON (must be obeyed):\n\n"
            + world_canon
        )

    # ---------------- Knowledge ----------------
    if route in ("knowledge", "both"):
        knowledge_results = retrieved.get("knowledge", [])[:2]  # smaller cap

        knowledge_chunks = [
            clean_text(r["text"])
            for r in knowledge_results
            if r.get("text")
        ]

        knowledge_context = (
            "SUPPORTING KNOWLEDGE (use only if needed):\n\n"
            + "\n".join(knowledge_chunks)
        )


# ---------------- Assemble Prompt ----------------
prompt = f"""
{system_text}

{canon_wrapper}
{canon_context}

{knowledge_wrapper}
{knowledge_context}

User Question:
{query}

Answer:
""".strip()

# ---------------- Write Output ----------------
OUT.write_text(prompt)

print("✅ Final prompt packed")
