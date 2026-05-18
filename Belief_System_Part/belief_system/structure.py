from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── OPENAI CLIENT ─────────────────────────────────────────────────────────────
_api_key  = os.getenv("OPENAI_API_KEY")
_base_url = os.getenv("OPENAI_BASE_URL")

client = OpenAI(api_key=_api_key, base_url=_base_url) if _base_url else OpenAI(api_key=_api_key)

MODEL       = os.getenv("OPENAI_MODEL", "gpt-5.1")
TEMPERATURE = 0.0

# ── DEDUP PROMPT ──────────────────────────────────────────────────────────────
DEDUP_PROMPT = """
You are given a list of belief statements extracted from KickstartAI's documents.
Some beliefs are semantically equivalent or redundant.

Your task:
1. Merge semantically equivalent beliefs into a single canonical belief.
2. Preserve all genuinely distinct beliefs.
3. Assign a unique ID to each canonical belief using the format B001, B002, B003, etc.
4. For each canonical belief, keep:
   - "id": unique belief identifier
   - "belief": canonical belief statement
   - "category": one of ["mission", "strategy", "domain_knowledge", "values", "stance"]
   - "sources": list of unique source_document values where this belief appeared
5. Be conservative: only merge beliefs that clearly mean the same thing.
6. Return ONLY valid JSON.

Input beliefs:
{beliefs_json}
""".strip()

# ── STRUCTURED OUTPUT SCHEMA ──────────────────────────────────────────────────
CANONICAL_SCHEMA = {
    "name": "canonical_belief_repository",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "beliefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":       {"type": "string"},
                        "belief":   {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["mission", "strategy", "domain_knowledge", "values", "stance"]
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["id", "belief", "category", "sources"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["beliefs"],
        "additionalProperties": False
    }
}


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def load_raw_beliefs(path: str | Path) -> list[dict]:
    """
    Load and clean raw beliefs from the Step 2 output JSON.
    Skips entries missing 'belief' or 'category'.
    Raises FileNotFoundError or ValueError on bad input.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 2 output: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")

    cleaned = []
    for b in data:
        belief          = str(b.get("belief", "")).strip()
        category        = str(b.get("category", "")).strip()
        source_document = str(b.get("source_document", "")).strip()

        if not belief or not category:
            continue

        cleaned.append({
            "belief":          belief,
            "category":        category,
            "source_document": source_document or "unknown",
        })

    if not cleaned:
        raise ValueError(f"No usable beliefs found in {path}")

    print(f"[load_raw_beliefs] Loaded {len(cleaned)} beliefs from {path}")
    return cleaned


def deduplicate_and_structure(beliefs: list[dict]) -> list[dict]:
    """
    Send raw beliefs to the LLM for semantic deduplication and canonical structuring.
    Returns a list of canonical belief dicts with fields: id, belief, category, sources.
    """
    beliefs_json = json.dumps(beliefs, ensure_ascii=False, indent=2)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": "You are a careful analyst that consolidates extracted beliefs into a canonical repository."
            },
            {
                "role": "user",
                "content": DEDUP_PROMPT.format(beliefs_json=beliefs_json)
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": CANONICAL_SCHEMA,
        },
    )

    content = response.choices[0].message.content
    payload = json.loads(content)
    structured = payload["beliefs"]

    print(f"[deduplicate_and_structure] {len(beliefs)} raw → {len(structured)} canonical beliefs")
    return structured


def validate_ids(structured: list[dict]) -> None:
    """
    Ensure every canonical belief has a unique ID.
    Raises ValueError if any duplicate IDs are found.
    """
    seen: set[str] = set()

    for b in structured:
        bid = b["id"]
        if bid in seen:
            raise ValueError(f"Duplicate belief ID found: {bid}")
        seen.add(bid)

    print(f"[validate_ids] All {len(structured)} IDs are unique.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RAW_PATH = Path("data/processed/beliefs_extracted.json")
    OUT_PATH = Path("data/processed/belief_repository.json")

    raw_beliefs = load_raw_beliefs(RAW_PATH)
    structured  = deduplicate_and_structure(raw_beliefs)
    validate_ids(structured)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[Step 3] Canonical repository written : {OUT_PATH}")
    print(f"[Step 3] Total canonical beliefs      : {len(structured)}")

    for b in structured:
        print(f"  {b['id']} [{b['category']}]: {b['belief']}")