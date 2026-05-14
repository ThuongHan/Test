"""
main.py — KickstartAI Belief System Pipeline
=============================================
Orchestrates all five steps end-to-end:

    Step 1 — Ingestion      : load raw data (blog + LinkedIn posts)
    Step 2 — Extraction     : extract beliefs from text
    Step 3 — Structuring    : deduplicate and canonicalise beliefs
    Step 4 — Vectorstore    : embed beliefs and build vector index
    Step 5 — Interface      : expose retrieval interface for Member 3

Usage:
    python main.py                         # run full pipeline (default: llm extractor)
    python main.py --extractor sensemaking # run with sensemaking extractor
    python main.py --steps 1 2             # run specific steps only
    python main.py --steps 4 5             # rebuild vectorstore + interface only
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
STORE_DIR     = Path("data/belief_store")

BLOG_PATH         = RAW_DIR       / "blog.txt"
POSTS_CSV_PATH    = RAW_DIR       / "linkedin_posts.csv"
LINKEDIN_XLSX     = RAW_DIR       / "KickstartAI_LinkedIn_post_year_to_date.xlsx"
BELIEFS_RAW_PATH  = PROCESSED_DIR / "beliefs_extracted.json"
REPO_PATH         = PROCESSED_DIR / "belief_repository.json"
STORE_PATH        = STORE_DIR     / "beliefs_with_embeddings.json"

# ── STEP RUNNERS ──────────────────────────────────────────────────────────────

def run_step1() -> None:
    """
    Step 1 — Ingestion
    Load blog.txt and LinkedIn xlsx, save cleaned outputs to data/raw/.
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Ingestion")
    print("=" * 60)

    from utils.file_io import load_txt, load_linkedin_xlsx, save_txt, save_csv

    # Load blog text
    blog_text  = load_txt(BLOG_PATH)

    # Load and filter LinkedIn posts
    df_organic = load_linkedin_xlsx(LINKEDIN_XLSX)

    # Save cleaned outputs for Step 2
    save_txt(blog_text, BLOG_PATH)
    save_csv(df_organic, POSTS_CSV_PATH)

    print(f"\n[Step 1] Done. Outputs in: {RAW_DIR}")


def run_step2(extractor_method: str = "llm") -> None:
    """
    Step 2 — Belief Extraction
    Extract beliefs from blog + LinkedIn posts using selected extractor method.
    Change extractor_method to 'sensemaking' to use Method 2.
    """
    print("\n" + "=" * 60)
    print(f"STEP 2 — Extraction  [method: {extractor_method}]")
    print("=" * 60)

    from extractors import get_extractor

    extractor = get_extractor(extractor_method)

    extractor.run_pipeline(
        blog_path   = BLOG_PATH,
        posts_path  = POSTS_CSV_PATH,
        output_path = BELIEFS_RAW_PATH,
    )

    print(f"\n[Step 2] Done. Output: {BELIEFS_RAW_PATH}")


def run_step3() -> None:
    """
    Step 3 — Structuring
    Deduplicate and canonicalise raw beliefs into belief_repository.json.
    """
    print("\n" + "=" * 60)
    print("STEP 3 — Structuring")
    print("=" * 60)

    from belief_system.structure  import load_raw_beliefs, deduplicate_and_structure, validate_ids
    from belief_system.repository  import save_beliefs

    raw_beliefs = load_raw_beliefs(BELIEFS_RAW_PATH)
    structured  = deduplicate_and_structure(raw_beliefs)
    validate_ids(structured)
    save_beliefs(structured, REPO_PATH)

    print(f"\n[Step 3] Done. {len(structured)} canonical beliefs → {REPO_PATH}")


def run_step4() -> None:
    """
    Step 4 — Vectorstore
    Embed all canonical beliefs and build the vector index.
    """
    print("\n" + "=" * 60)
    print("STEP 4 — Vectorstore")
    print("=" * 60)

    from belief_system.repository import load_beliefs
    from embeddings.index         import build_vectorstore

    beliefs = load_beliefs(REPO_PATH)
    build_vectorstore(beliefs, store_file=STORE_PATH)

    print(f"\n[Step 4] Done. Vectorstore → {STORE_PATH}")


def run_step5() -> None:
    """
    Step 5 — Interface (smoke test)
    Verify the retrieval interface works end-to-end.
    This is the public interface consumed by Member 3 (Interpreter).
    """
    print("\n" + "=" * 60)
    print("STEP 5 — Interface (smoke test)")
    print("=" * 60)

    from retrieval.retriever import (
        get_all_beliefs,
        get_beliefs_by_category,
        retrieve_relevant_beliefs,
        format_beliefs_for_prompt,
    )

    # 5a. All beliefs
    all_beliefs = get_all_beliefs(REPO_PATH)
    print(f"[Step 5] Total beliefs in repository : {len(all_beliefs)}")

    # 5b. By category
    for cat in ["values", "strategy", "mission", "stance", "domain_knowledge"]:
        filtered = get_beliefs_by_category(cat, REPO_PATH)
        print(f"  [{cat:<20}] {len(filtered)} beliefs")

    # 5c. RAG retrieval
    test_query = "How should AI contribute to Dutch society?"
    results    = retrieve_relevant_beliefs(test_query, k=3, store_path=STORE_PATH)
    print(f"\n[Step 5] Top-3 beliefs for: '{test_query}'")
    print(format_beliefs_for_prompt(results))

    print(f"\n[Step 5] Done. Retrieval interface is ready for Member 3.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(steps: list[int], extractor_method: str) -> None:

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    step_runners = {
        1: run_step1,
        2: lambda: run_step2(extractor_method),
        3: run_step3,
        4: run_step4,
        5: run_step5,
    }

    for step in steps:
        step_runners[step]()

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Steps run: {steps}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KickstartAI Belief System Pipeline")

    parser.add_argument(
        "--extractor",
        type=str,
        default="llm",
        choices=["llm", "sensemaking"],
        help="Extractor method for Step 2 (default: llm)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Steps to run (default: 1 2 3 4 5)"
    )

    args = parser.parse_args()
    main(steps=sorted(args.steps), extractor_method=args.extractor)