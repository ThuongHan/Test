from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from extractors.base_extractor import BaseExtractor

load_dotenv()

# ── OPENAI CLIENT ─────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL       = os.getenv("OPENAI_MODEL", "gpt-5.1")
TEMPERATURE = 0.0

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert organisational analyst specialising in belief and value extraction
from institutional communications.

A BELIEF is a statement that defines what an organisation considers essential,
correct, or strategically important. Beliefs serve as principles that motivate
decisions, public positions, and behavioural patterns.

A belief is NOT:
- A factual report of an event
- A simple product announcement with no normative content
- A description of an activity without an implicit stance

Be precise, conservative, and faithful to the source text.
""".strip()

# ── PROMPT TEMPLATES ──────────────────────────────────────────────────────────
PROMPT_A_TEMPLATE = """
TASK — PRIMARY BELIEF EXTRACTION (Pass A)

You are given the following organisational text:

\"\"\"{text}\"\"\"

Your goal: identify all PRIMARY beliefs in this text.

A PRIMARY belief is one that is EXPLICITLY stated. Reliable indicators include:
- First-person declarations: "we believe", "we think", "our mission is", "we are convinced"
- Normative prescriptions directed at others: "organisations must", "the key is to", "AI should"
- Explicit value claims: "what matters is", "this is essential", "the priority is"
- Goal statements: "our aim is", "we strive to", "we are committed to"

Instructions:
- Extract only the MOST SALIENT primary beliefs.
- Maximum 3 beliefs for this text segment.
- Do not include factual observations unless they clearly express a belief.
- Do not split one idea into multiple near-duplicate beliefs.
- Write each belief as a full, standalone declarative sentence.
- Select the SHORTEST phrase from the text that triggered the belief as source_quote.
- Assign one category from: values | stance | strategy | mission | domain_knowledge
- Return an empty list if none are present.
""".strip()

PROMPT_B_TEMPLATE = """
TASK — SECONDARY BELIEF EXTRACTION (Pass B)

You are given the following organisational text:

\"\"\"{text}\"\"\"

Your goal: identify all SECONDARY beliefs in this text.

A SECONDARY belief is NOT directly stated, but is clearly implied by how the text
is written. Look for:
- Problem framing
- Causal attribution
- Prescriptive urgency
- Contrasting language
- Audience assumptions
- Metaphor and analogy choices

Instructions:
- Only include beliefs that are strongly and clearly inferable from the text.
- Extract only the MOST SALIENT secondary beliefs.
- Maximum 2 beliefs for this text segment.
- Do not include weak, speculative, repetitive, or overlapping inferences.
- Do not restate a primary belief as a secondary belief.
- Write each belief as a full, standalone declarative sentence.
- Provide one sentence of reasoning explaining the inference.
- Select the SHORTEST phrase from the text that grounds the inference as source_quote.
- Assign one category from: values | stance | strategy | mission | domain_knowledge
- Return an empty list if none are present.
""".strip()

# ── STRUCTURED OUTPUT SCHEMAS ─────────────────────────────────────────────────
PRIMARY_SCHEMA = {
    "name": "primary_beliefs",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "beliefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "belief":       {"type": "string"},
                        "category":     {"type": "string", "enum": ["values", "stance", "strategy", "mission", "domain_knowledge"]},
                        "source_quote": {"type": "string"},
                        "belief_type":  {"type": "string", "enum": ["primary"]},
                    },
                    "required": ["belief", "category", "source_quote", "belief_type"],
                    "additionalProperties": False,
                }
            }
        },
        "required": ["beliefs"],
        "additionalProperties": False,
    }
}

SECONDARY_SCHEMA = {
    "name": "secondary_beliefs",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "beliefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "belief":               {"type": "string"},
                        "category":             {"type": "string", "enum": ["values", "stance", "strategy", "mission", "domain_knowledge"]},
                        "source_quote":         {"type": "string"},
                        "inference_reasoning":  {"type": "string"},
                        "belief_type":          {"type": "string", "enum": ["secondary"]},
                    },
                    "required": ["belief", "category", "source_quote", "inference_reasoning", "belief_type"],
                    "additionalProperties": False,
                }
            }
        },
        "required": ["beliefs"],
        "additionalProperties": False,
    }
}


# ── HELPER ────────────────────────────────────────────────────────────────────
def _call_api(text: str, prompt_template: str, schema: dict) -> list[dict]:
    """Call OpenAI with structured output. Retries once on failure."""
    prompt = prompt_template.format(text=text)

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            payload = json.loads(response.choices[0].message.content)
            return payload.get("beliefs", [])

        except Exception as e:
            if attempt == 0:
                print(f"    [WARN] attempt 1 failed: {e}")
                time.sleep(1)
            else:
                print(f"    [ERROR] both attempts failed: {e}")
                return []


# ── EXTRACTOR CLASS ───────────────────────────────────────────────────────────
class LLMExtractor(BaseExtractor):
    """
    Method 1 — Multi-pass LLM extraction (primary + secondary beliefs).
    Inherits BaseExtractor; implements extract() for a single text chunk.
    run_pipeline() orchestrates the full Step 2 workflow.
    """

    def extract(self, text: str, source_label: str) -> list[dict]:
        """
        Run Pass A (primary) + Pass B (secondary) on a single text segment.
        Implements the abstract method from BaseExtractor.
        """
        primary   = _call_api(text, PROMPT_A_TEMPLATE, PRIMARY_SCHEMA)
        secondary = _call_api(text, PROMPT_B_TEMPLATE, SECONDARY_SCHEMA)

        for b in primary + secondary:
            b["source_document"] = source_label

        return primary + secondary

    def run_pipeline(
        self,
        blog_path:   str | Path,
        posts_path:  str | Path,
        output_path: str | Path,
        seed_path:   str | Path | None = None,
        max_docs:    int = 268,
    ) -> list[dict]:
        """
        Full Step 2 pipeline:
        1. Build corpus (blog chunks + LinkedIn posts)
        2. Load optional seed beliefs
        3. Run extract() on every document
        4. Deduplicate and save to output_path
        """
        from utils.text_processing import build_corpus
        from extractors import deduplicate_beliefs, load_seed_beliefs

        blog_path   = Path(blog_path)
        posts_path  = Path(posts_path)
        output_path = Path(output_path)

        corpus = build_corpus(blog_path, posts_path)
        print(f"\n[LLMExtractor] Documents prepared: {len(corpus)}")

        all_beliefs = []

        if seed_path:
            seeds = load_seed_beliefs(seed_path)
            all_beliefs.extend(seeds)
            print(f"[SEED] Loaded {len(seeds)} seed beliefs")

        for i, doc in enumerate(corpus[:max_docs], start=1):
            text   = doc["text"].strip()
            source = doc["source"]
            doc_id = doc["id"]

            if not text:
                continue

            print(f"\n[{i}/{len(corpus)}] {doc_id} | {source}")
            beliefs = self.extract(text, source_label=source)

            for b in beliefs:
                b["source_id"]   = doc_id
                b["source_text"] = text
                b["meta"]        = doc.get("meta", {})

            all_beliefs.extend(beliefs)
            time.sleep(0.4)

        print(f"\n[DEDUP] Before: {len(all_beliefs)}")
        final = deduplicate_beliefs(all_beliefs)
        print(f"[DEDUP] After : {len(final)}")

        category_counts = Counter(b.get("category", "unknown") for b in final)
        type_counts     = Counter(b.get("belief_type", "unknown") for b in final)

        print("\n[SUMMARY] By category")
        for k, v in sorted(category_counts.items()):
            print(f"  {k:<22} {v}")
        print("\n[SUMMARY] By type")
        for k, v in sorted(type_counts.items()):
            print(f"  {k:<22} {v}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[OUTPUT] Written: {output_path}")
        return final