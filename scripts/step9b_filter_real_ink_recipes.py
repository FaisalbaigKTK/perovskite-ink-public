import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INFILE = ROOT / "final_perovskite_ink_recipes_only.csv"   # run_all expects root outputs
OUT_REAL = ROOT / "final_perovskite_ink_recipes_REAL_ONLY.csv"
OUT_SEMI = ROOT / "final_perovskite_ink_recipes_SEMI_REAL.csv"
OUT_ALL  = ROOT / "final_perovskite_ink_recipes_SCORED_ALL.csv"

# Patterns
RE_RATIO = re.compile(r"\b\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\b")
RE_VOLRATIO = re.compile(r"\b(v\/v|vol\/vol|volume\/volume)\b", re.IGNORECASE)
RE_WTRATIO = re.compile(r"\b(w\/w|wt\/wt|weight\/weight)\b", re.IGNORECASE)

RE_MOLARITY_M = re.compile(r"\b\d+(?:\.\d+)?\s*M\b", re.IGNORECASE)
RE_MOLARITY_MOL_L = re.compile(r"\b\d+(?:\.\d+)?\s*(mol\s*L[-−]?1|mol\/L)\b", re.IGNORECASE)
RE_MM = re.compile(r"\b\d+(?:\.\d+)?\s*mM\b", re.IGNORECASE)

RE_TEMP = re.compile(r"\b\d+(?:\.\d+)?\s*°\s*C\b")
RE_TIME = re.compile(r"\b\d+(?:\.\d+)?\s*(s|sec|secs|seconds|min|mins|minutes|h|hr|hrs|hours)\b", re.IGNORECASE)
RE_FILTER = re.compile(r"\b0\.\d+\s*(µm|um)\b", re.IGNORECASE)

RE_MASSVOL = re.compile(r"\b\d+(?:\.\d+)?\s*(mg|g|µg|ug|mL|ml|L)\b", re.IGNORECASE)

# Chemistry keywords (broad to avoid dropping true positives)
PB_SALTS = ["pbi2", "pbbr2", "pbcl2", "pbn", "pb(ac)2", "lead(ii)"]
A_SITE = ["mai", "mabr", "macl", "fai", "fabr", "csi", "csbr", "formamidinium", "methylammonium", "cesium"]
SOLVENTS = ["dmso", "dmf", "gbl", "gvl", "nmp", "acn", "acetonitrile", "ipa", "isopropanol"]

RECIPE_VERBS = [
    "dissolv", "prepare", "precursor", "solution", "ink",
    "mix", "stir", "sonicat", "heat", "filter", "add", "dropwise"
]

BAD_CONTEXT = [
    "wash", "rins", "clean", "etch",
    "antisolvent", "anti-solvent",
    "substrate", "electrode"
]

def build_text(row) -> str:
    # Works with your pipeline column names (and tolerates missing ones)
    parts = [
        row.get("evidence_block", ""),
        row.get("best_recipe_block", ""),
        row.get("precursors", ""),
        row.get("solvents", ""),
        row.get("molarity_M", ""),
        row.get("molarity", ""),
        row.get("solvent_ratio", ""),
        row.get("ratios", ""),
        row.get("mix_temp_C", ""),
        row.get("temps_C", ""),
        row.get("times_found", ""),
        row.get("times", ""),
        row.get("filter_um", ""),
    ]
    return " ".join([str(p) for p in parts if p is not None]).lower()

def recipe_confidence_score(text: str) -> int:
    score = 0

    # Strong numeric evidence
    if RE_RATIO.search(text):
        score += 25
    if RE_VOLRATIO.search(text) or RE_WTRATIO.search(text):
        score += 10

    if RE_MOLARITY_M.search(text) or RE_MOLARITY_MOL_L.search(text) or RE_MM.search(text):
        score += 25

    # Helpful process evidence
    if RE_TEMP.search(text):
        score += 8
    if RE_TIME.search(text):
        score += 6
    if RE_FILTER.search(text):
        score += 8

    # Units often appear in real recipes
    if RE_MASSVOL.search(text):
        score += 8

    # Chemistry presence
    if any(k in text for k in PB_SALTS):
        score += 12
    if any(k in text for k in A_SITE):
        score += 10
    if any(k in text for k in SOLVENTS):
        score += 10

    # Recipe verbs
    if any(v in text for v in RECIPE_VERBS):
        score += 12

    # Penalize bad context ONLY if recipe evidence is weak
    bad = any(b in text for b in BAD_CONTEXT)
    strong_numeric = (RE_RATIO.search(text) is not None) or (RE_MOLARITY_M.search(text) is not None) or (RE_MOLARITY_MOL_L.search(text) is not None) or (RE_MM.search(text) is not None)
    if bad and not strong_numeric:
        score -= 15

    # Clamp
    if score < 0:
        score = 0
    if score > 100:
        score = 100
    return score

def main():
    if not INFILE.exists():
        print(f"Missing input: {INFILE}")
        return

    df = pd.read_csv(INFILE)
    if df.empty:
        print("Input file is empty.")
        df.to_csv(OUT_ALL, index=False)
        df.to_csv(OUT_REAL, index=False)
        df.to_csv(OUT_SEMI, index=False)
        return

    texts = df.apply(lambda r: build_text(r), axis=1)
    df["recipe_confidence"] = texts.apply(recipe_confidence_score)

    # Save all scored
    df.to_csv(OUT_ALL, index=False)

    # Thresholds tuned for early-stage dataset building
    # With only 3 rows, you need recall. We keep semi-real too.
    df_real = df[df["recipe_confidence"] >= 60].copy()
    df_semi = df[(df["recipe_confidence"] >= 35) & (df["recipe_confidence"] < 60)].copy()

    df_real.to_csv(OUT_REAL, index=False)
    df_semi.to_csv(OUT_SEMI, index=False)

    print(f"Rows total: {len(df)}")
    print(f"REAL_ONLY (>=60): {len(df_real)}")
    print(f"SEMI_REAL (35-59): {len(df_semi)}")
    print(f"\nSaved: {OUT_ALL.name}, {OUT_REAL.name}, {OUT_SEMI.name}")

    # Show top few with confidence
    show_cols = [c for c in ["pdf_file", "doi", "molarity_M", "solvent_ratio", "solvents", "precursors", "recipe_confidence"] if c in df.columns]
    print("\nTop scored rows:")
    print(df.sort_values("recipe_confidence", ascending=False)[show_cols].head(15).to_string(index=False))

if __name__ == "__main__":
    main()
