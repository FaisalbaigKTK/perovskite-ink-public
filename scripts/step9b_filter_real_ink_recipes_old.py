import re
import pandas as pd
from pathlib import Path

INFILE = "final_perovskite_ink_recipes_only.csv"
OUTFILE = "final_perovskite_ink_recipes_REAL_ONLY.csv"

RE_RATIO = re.compile(r"\b\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\b")
RE_MOLARITY_TOKEN = re.compile(r"\b\d+(?:\.\d+)?\s*M\b", re.IGNORECASE)
RE_FLOAT = re.compile(r"\b\d+(?:\.\d+)?\b")

RECIPE_WORDS = [
    "dissolved", "prepared", "solution", "precursor", "stir", "stirred",
    "heated", "mixed", "filter", "filtered", "aged", "shake", "shaken",
    "added", "add", "dropwise"
]

BAD_CONTEXT = [
    "wash", "rins", "clean", "antisolvent", "anti-solvent", "spin-coat", "spin coat",
    "anneal", "substrate", "electrode"
]

def looks_like_real_recipe(row) -> bool:
    # IMPORTANT: use the columns that actually exist in your pipeline
    text = " ".join([
        str(row.get("evidence_block", "")),
        str(row.get("best_recipe_block", "")),   # if present
        str(row.get("precursors", "")),
        str(row.get("solvents", "")),
        str(row.get("molarity_M", "")),
        str(row.get("solvent_ratio", "")),
        str(row.get("mix_temp_C", "")),
        str(row.get("times_found", "")),
        str(row.get("filter_um", "")),
    ]).lower()

    has_recipe_word = any(w in text for w in RECIPE_WORDS)
    has_ratio = bool(RE_RATIO.search(text))
    has_molarity = bool(RE_MOLARITY_TOKEN.search(text))
    has_any_number = bool(RE_FLOAT.search(text))

    has_bad_context = any(b in text for b in BAD_CONTEXT)

    # Strong keep condition: recipe verb + (ratio or molarity)
    if has_recipe_word and (has_ratio or has_molarity):
        return True

    # If it looks like cleaning-only and has weak numeric evidence, reject
    if has_bad_context and not (has_ratio or has_molarity):
        return False

    # Otherwise keep if there's at least a ratio or molarity (still meaningful)
    if has_ratio or has_molarity:
        return True

    # Very weak evidence: drop
    return False

def main():
    infile = Path(INFILE)
    if not infile.exists():
        print(f"Missing input: {INFILE}")
        return

    df = pd.read_csv(INFILE)
    before = len(df)

    df2 = df[df.apply(looks_like_real_recipe, axis=1)].copy()
    after = len(df2)

    df2.to_csv(OUTFILE, index=False)
    print(f"Kept rows: {after} out of {before}")

    preview_cols = [c for c in ["year", "doi", "pdf_file", "molarity_M", "solvent_ratio", "solvents", "precursors"] if c in df2.columns]
    if preview_cols:
        print(df2[preview_cols].head(20).to_string(index=False))
    else:
        print(df2.head(20).to_string(index=False))

    print(f"\nSaved: {OUTFILE}")

if __name__ == "__main__":
    main()
