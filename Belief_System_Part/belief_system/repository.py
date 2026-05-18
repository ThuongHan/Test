from __future__ import annotations

import json
from pathlib import Path


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_beliefs(beliefs: list[dict], path: str | Path) -> None:
    """
    Write a list of canonical belief dicts to a JSON file.
    Creates parent directories if they do not exist.

    Args:
        beliefs: list of canonical belief dicts
                 (each must have: id, belief, category, sources)
        path:    output file path (e.g. 'data/processed/belief_repository.json')
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(beliefs, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[save_beliefs] Written {len(beliefs)} beliefs → {out_path}")


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_beliefs(path: str | Path) -> list[dict]:
    """
    Load and validate canonical beliefs from the belief_repository.json.
    Skips entries missing required fields: id, belief, category.

    Args:
        path: path to belief_repository.json

    Returns:
        List of cleaned canonical belief dicts.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the file is empty or contains no valid beliefs.
    """
    in_path = Path(path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Belief repository not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {in_path}, got {type(data).__name__}")

    cleaned = []
    for b in data:
        belief_id   = str(b.get("id",       "")).strip()
        belief_text = str(b.get("belief",   "")).strip()
        category    = str(b.get("category", "")).strip()
        sources     = b.get("sources", [])

        if not belief_id or not belief_text or not category:
            continue

        # normalise sources to a clean list of strings
        if not isinstance(sources, list):
            sources = [str(sources)]
        sources = [str(s).strip() for s in sources if str(s).strip()]

        cleaned.append({
            "id":       belief_id,
            "belief":   belief_text,
            "category": category,
            "sources":  sources,
        })

    if not cleaned:
        raise ValueError(f"No usable beliefs found in {in_path}")

    print(f"[load_beliefs] Loaded {len(cleaned)} beliefs from {in_path}")
    return cleaned


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REPO_PATH = Path("data/processed/belief_repository.json")

    beliefs = load_beliefs(REPO_PATH)

    print(f"\n[repository] {len(beliefs)} beliefs loaded")
    print(f"{'ID':<8} {'Category':<20} Belief")
    print("-" * 80)

    for b in beliefs:
        print(f"{b['id']:<8} {b['category']:<20} {b['belief'][:60]}")