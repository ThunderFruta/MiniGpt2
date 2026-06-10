import json
import random

random.seed(42)

TOPICS = {
    "fruit": {
        "things": ["bananas", "oranges"],
        "prompts": [
            "do you like {x}",
            "what do you think of {x}",
            "are {x} good",
            "do you enjoy {x}"
        ],
        "responses": [
            "i think {x} are nice.",
            "{x} are okay to me.",
            "i like {x}.",
            "{x} seem fine."
        ]
    },
    "company": {
        "things": ["thunderfruta"],
        "prompts": [
            "do you like {x}",
            "what do you think of {x}",
            "is {x} good",
            "how do you feel about {x}"
        ],
        "responses": [
            "{x} seems like a good company.",
            "i like {x}.",
            "{x} looks solid to me."
        ]
    },
    "preference": {
        "things": ["simple things", "short answers"],
        "prompts": [
            "do you prefer {x}",
            "do you like {x}",
            "how do you feel about {x}"
        ],
        "responses": [
            "yes, i prefer {x}.",
            "i usually like {x}.",
            "{x} feel better to me."
        ]
    }
}

entries = []

for topic in TOPICS.values():
    for thing in topic["things"]:
        for p in topic["prompts"]:
            for r in topic["responses"]:
                entries.append({
                    "prompt": f"User: {p.format(x=thing)}",
                    "response": r.format(x=thing)
                })

# trim or expand safely
entries = entries[:300]

with open("stage3.jsonl", "w") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"Generated {len(entries)} Stage 3 opinion lines.")
