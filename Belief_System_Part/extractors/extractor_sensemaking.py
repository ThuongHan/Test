from __future__ import annotations

import json
import os
import re
import time
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

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION A — Rule-based sensemaking signal patterns
#  Drawing on: Malik et al. (2025); Weick (1995)
# ─────────────────────────────────────────────────────────────────────────────

PRESCRIPTIVE_PATTERNS = [
    r"\borganisations? must\b",
    r"\borganisations? should\b",
    r"\bthe key is\b",
    r"\bwhat('s| is) needed\b",
    r"\bwe (must|need to|have to|should)\b",
    r"\bit is (essential|critical|imperative|necessary)\b",
    r"\bcompanies? (must|should|need to)\b",
    r"\bleaders? (must|should|need to)\b",
    r"\b(must|should) be (built|embedded|designed|adopted|implemented)\b",
]

CAUSAL_PATTERNS = [
    r"\bbecause\b",
    r"\bdue to\b",
    r"\bas a result\b",
    r"\bleads? to\b",
    r"\benables?\b",
    r"\bdriven by\b",
    r"\bthe reason\b",
    r"\bif .{0,60} then\b",
    r"\bwithout .{0,60}, .{0,60} (will|cannot|won't)\b",
]

PROBLEM_FRAMING_PATTERNS = [
    r"\bchallenge\b",
    r"\bgap\b",
    r"\bbarrier\b",
    r"\bstill\b.{0,40}(lack|behind|struggle|fail)",
    r"\bnot (yet|enough|sufficient)\b",
    r"\bproblem\b",
    r"\bstruggle\b",
    r"\bmissing\b",
    r"\blocked\b",
]

# ── PROMPTS ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert qualitative researcher specialising in organisational sensemaking theory
and digitally enabled strategic agility.

Your task is to extract implicit organisational beliefs from KickstartAI communications.
Use the academic framework from Malik et al. (2025), which theorises that digitally enabled
strategic agility is constructed through organisational sensemaking.

Core theoretical assumptions:
1. Organisations face weak signals and equivocality when interpreting digital, AI, or market change.
2. Sensemaking removes equivocality through two linked devices:
   a. Meaning / discourse: shared interpretations, cognitive frames, and digital orientation.
   b. Actions / process facilitators: routines, structures, governance, and transformation practices.
3. Digital orientation represents the discursive meaning structure.
4. Information governance and digital transformation represent action-oriented process facilitators.
5. Strategic agility emerges when meaning and action enable coordinated organisational response.

Extract only beliefs that are inferable from the text. Do not invent beliefs.
For each belief, identify the textual signal and the theoretical mechanism.

Return a valid JSON array with exactly these fields:
{
  "belief_id": "<sequential integer as string>",
  "belief_statement": "<concise declarative sentence>",
  "observed_signal": "<problem_framing | prescriptive_language | causal_attribution | strategic_orientation | governance_signal | transformation_signal>",
  "sensemaking_role": "<meaning | action | meaning_action_link | equivocality_removal>",
  "theoretical_construct": "<digital_orientation | information_governance | digital_transformation | digitally_enabled_strategic_agility | equivocality>",
  "inference_type": "<problem_framing | prescriptive | causal_attribution | theoretical_interpretation>",
  "inference_logic": "<one sentence explaining how the quote supports the belief using the theory>",
  "source_excerpt": "<verbatim short quote of 25 words or fewer from the input>",
  "confidence": "<high | medium | low>",
  "domain": "<AI_adoption | societal_impact | organisational_capability | knowledge_sharing | collaboration | responsibility>"
}

Rules:
- Do not extract generic themes; extract implicit beliefs.
- Every belief must be grounded in a source_excerpt.
- The inference_logic must explicitly connect the excerpt to sensemaking theory.
- If the text only reports an event without normative, causal, or sensemaking implication, do not extract it.
- Return ONLY valid JSON. No markdown.
"""

DEDUP_SYSTEM = """
You are a qualitative research analyst using organisational sensemaking theory.

You will receive a JSON array of extracted implicit beliefs.
Deduplicate semantically redundant beliefs only when they express the same underlying belief
AND share the same:
- theoretical_construct
- sensemaking_role
- domain

Do not merge beliefs if they refer to different sensemaking mechanisms, e.g. meaning/discourse
versus action/process facilitation.

When duplicates exist, retain the belief with:
1. the clearest belief_statement,
2. the strongest source_excerpt,
3. the most explicit inference_logic,
4. the highest confidence.

Return ONLY the deduplicated JSON array using the same schema. No markdown.
"""


# ── HELPERS ───────────────────────────────────────────────────────────────────
def count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Return total regex match count across all patterns."""
    return sum(len(re.findall(pat, text, re.IGNORECASE)) for pat in patterns)


def strip_code_fences(text: str) -> str:
    """Remove accidental markdown code fences from model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def annotate_posts_with_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add sensemaking signal count columns to a LinkedIn posts DataFrame."""
    if "Post title" not in df.columns:
        raise KeyError("Column 'Post title' not found in DataFrame")

    df = df.copy()
    df["n_prescriptive"]    = df["Post title"].apply(lambda t: count_pattern_matches(str(t), PRESCRIPTIVE_PATTERNS))
    df["n_causal"]          = df["Post title"].apply(lambda t: count_pattern_matches(str(t), CAUSAL_PATTERNS))
    df["n_problem_framing"] = df["Post title"].apply(lambda t: count_pattern_matches(str(t), PROBLEM_FRAMING_PATTERNS))
    df["sensemaking_signal_total"] = df["n_prescriptive"] + df["n_causal"] + df["n_problem_framing"]

    print("\n=== Sensemaking Signal Distribution ===")
    print(df[["n_prescriptive", "n_causal", "n_problem_framing", "sensemaking_signal_total"]].describe().round(2))
    return df


def _llm_deduplicate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Semantic deduplication via a second LLM pass. Falls back on failure."""
    
    available_cols = [
    c for c in [
        "belief_id", "belief_statement", "inference_type",
        "source_excerpt", "confidence",
        "theoretical_construct", "belief_domain",
        "sensemaking_role", "source_label", "source_document",
    ]
    if c in df_raw.columns
    ]   
    
    dedup_input = json.dumps(df_raw[available_cols].to_dict(orient="records"), ensure_ascii=False)

    print("\n[DEDUP] Running semantic deduplication via LLM ...")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": DEDUP_SYSTEM},
                {"role": "user",   "content": f"Deduplicate:\n{dedup_input[:12000]}"},
            ],
        )
        raw = strip_code_fences(response.choices[0].message.content or "")
        df_deduped = pd.DataFrame(json.loads(raw))
        print(f"  -> {len(df_deduped)} beliefs retained")
        return df_deduped

    except Exception as e:
        print(f"  [WARNING] Deduplication failed ({e}); using raw beliefs.")
        return df_raw.copy()


# ── EXTRACTOR CLASS ───────────────────────────────────────────────────────────
class SensemakingExtractor(BaseExtractor):
    """
    Method 2 — Rule-based pre-pass + LLM implicit belief extraction.
    Drawing on: Malik et al. (2025); Weick (1995) sensemaking theory.

    Section A: regex signals (prescriptive, causal, problem framing)
    Section B: LLM extracts implicit beliefs via three sensemaking lenses
    Section C: LLM semantic deduplication pass
    """

    def extract(self, text: str, source_label: str) -> list[dict]:
        """
        Extract implicit beliefs from a single text using sensemaking lenses.
        Implements the abstract method from BaseExtractor.
        """
        user_message = (
            f"Source: {source_label}\n\n"
            f"--- BEGIN TEXT ---\n{text[:4000]}\n--- END TEXT ---\n\n"
            "Extract implicit organisational beliefs using Malik et al. (2025)'s "
            "sensemaking framework. For each belief, show whether the text expresses "
            "meaning/discourse, action/process facilitation, equivocality removal, or a "
            "meaning-action link. Only extract beliefs supported by a verbatim excerpt."
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            raw = strip_code_fences(response.choices[0].message.content or "")
            beliefs = json.loads(raw)

            if not isinstance(beliefs, list):
                print(f"  [WARNING] Output for '{source_label}' is not a list.")
                return []

            for b in beliefs:
                if isinstance(b, dict):
                    b["source_label"]    = source_label
                    b["source_document"] = source_label

            return [b for b in beliefs if isinstance(b, dict)]

        except json.JSONDecodeError as e:
            print(f"  [WARNING] JSON parsing failed for '{source_label}': {e}")
            return []
        except Exception as e:
            print(f"  [WARNING] OpenAI call failed for '{source_label}': {e}")
            return []

    def extract_from_posts(self, posts: list[dict], batch_size: int = 20) -> list[dict]:
        """
        Override: process LinkedIn posts in batches instead of one by one.
        Batching reduces API calls and preserves context across titles.
        """
        post_titles = [
            str(p.get("Post title", "")).strip()
            for p in posts
            if str(p.get("Post title", "")).strip()
        ]

        batches = [post_titles[i:i + batch_size] for i in range(0, len(post_titles), batch_size)]
        all_beliefs: list[dict] = []

        for idx, batch in enumerate(batches, start=1):
            batch_text = "\n\n".join([f"Post {i + 1}: {t}" for i, t in enumerate(batch)])
            label = f"linkedin_batch_{idx:02d}"
            print(f"  Processing {label} ({len(batch)} posts) ...")
            beliefs = self.extract(text=batch_text, source_label=label)
            all_beliefs.extend(beliefs)
            time.sleep(1)

        return all_beliefs

    def run_pipeline(
        self,
        blog_path:           str | Path,
        posts_path:          str | Path,
        output_dir:          str | Path,
        output_path:         str | Path | None = None,
        batch_size:          int = 20,
        blog_char_limit:     int | None = None,
        linkedin_row_limit:  int | None = None,
    ) -> pd.DataFrame:
        """
        Full Step 2 sensemaking pipeline.
        Returns a deduplicated DataFrame of implicit beliefs.
        """
        blog_path  = Path(blog_path)
        posts_path = Path(posts_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load inputs
        blog_text_full = blog_path.read_text(encoding="utf-8").strip()
        blog_text = blog_text_full[:blog_char_limit] if blog_char_limit else blog_text_full

        df_posts = pd.read_csv(posts_path)
        if linkedin_row_limit:
            df_posts = df_posts.head(linkedin_row_limit).copy()

        print(f"[SensemakingExtractor] Blog : {len(blog_text):,} chars")
        print(f"[SensemakingExtractor] Posts: {len(df_posts):,} rows")

        # Section A: annotate posts with signal scores
        df_posts = annotate_posts_with_signals(df_posts)

        # Section B: LLM extraction
        print("\n[B.2] Blog extraction ...")
        blog_beliefs = self.extract(blog_text, source_label="blog")
        print(f"  -> {len(blog_beliefs)} beliefs")
        time.sleep(1)

        print("\n[B.3] LinkedIn posts extraction ...")
        posts_records = df_posts.to_dict(orient="records")
        linkedin_beliefs = self.extract_from_posts(posts_records, batch_size=batch_size)
        print(f"  -> {len(linkedin_beliefs)} beliefs")

        # Section C: consolidation + deduplication
        all_raw = blog_beliefs + linkedin_beliefs
        for i, b in enumerate(all_raw, start=1):
            b["belief_id"] = str(i)

        df_raw = pd.DataFrame(all_raw)
        print(f"\n=== Raw beliefs before dedup: {len(df_raw)} ===")

        if df_raw.empty:
            print("No beliefs extracted.")
            return df_raw

        df_final = _llm_deduplicate(df_raw)

        # Section D: save outputs
        if output_path is not None:
            # Use caller-supplied path (e.g. beliefs_extracted_sensemaking.json)
            output_path   = Path(output_path)
            stem          = output_path.stem                          # e.g. beliefs_extracted_sensemaking
            beliefs_out   = output_path
            raw_out       = output_dir / f"{stem}_raw.json"
            annotated_out = output_dir / f"{stem}_linkedin_annotated.json"
        else:
            # Legacy fallback — original hardcoded names
            beliefs_out   = output_dir / "beliefs_extracted_method2.json"
            raw_out       = output_dir / "beliefs_raw_method2.json"
            annotated_out = output_dir / "linkedin_posts_annotated_method2.json"

        df_final.to_json(beliefs_out,   orient="records", force_ascii=False, indent=2)
        df_raw.to_json(raw_out,         orient="records", force_ascii=False, indent=2)
        df_posts.to_json(annotated_out, orient="records", force_ascii=False, indent=2)

        print(f"\n[OUTPUT] Written:")
        print(f"  ├── {beliefs_out}")
        print(f"  ├── {raw_out}")
        print(f"  └── {annotated_out}")

        return df_final