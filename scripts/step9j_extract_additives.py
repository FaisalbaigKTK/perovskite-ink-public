import re
import sys
import pandas as pd
from pathlib import Path

# --- Make Windows console printing robust to unicode ---
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# We will auto-pick input from these candidates (first existing wins)
INPUT_CANDIDATES = [
    DATA / "05_final" / "final_perovskite_ink_recipes_ML_READY4.csv",
    DATA / "05_final" / "final_perovskite_ink_recipes_ML_READY.csv",
    DATA / "05_final" / "final_perovskite_ink_recipes_STRUCTURED.csv",
    DATA / "04_clean" / "final_perovskite_ink_recipes_STRUCTURED.csv",
    DATA / "04_clean" / "final_perovskite_ink_recipes_CLEAN.csv",
    DATA / "04_clean" / "final_perovskite_ink_recipes_REAL_ONLY.csv",
]

OUTFILE = DATA / "05_final" / "final_perovskite_ink_recipes_ADDATIVES.csv"

# Additive keywords (expand anytime)
ADD = [
    "MACl", "PEABr", "PEAI", "NH4Cl", "NH4SCN", "SCN", "KSCN",
    "HI", "HBr", "CsI", "RbI", "guanidinium", "GAI", "GABr",
    "Pb(SCN)2", "thiocyanate", "tBP", "4-tert-Butylpyridine",
    "DIO", "NMP", "toluene", "chlorobenzene"
]

# Amount patterns
RE_MMMOL = re.compile(r"(\d+(?:\.\d+)?)\s*(mmol|mol)\b", re.IGNORECASE)
RE_MG = re.compile(r"(\d+(?:\.\d+)?)\s*mg\b", re.IGNORECASE)
RE_ML = re.compile(r"(\d+(?:\.\d+)?)\s*(mL|ml|µL|uL)\b", re.IGNORECASE)
RE_VOLPCT = re.compile(r"(\d+(?:\.\d+)?)\s*vol\s*%|\b(\d+(?:\.\d+)?)\s*%\s*vol", re.IGNORECASE)


def pick_input_file() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    # If none exist, show the user what we tried
    tried = "\n".join(str(p) for p in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "Step9J could not find an input CSV. Tried:\n" + tried
    )


def extract_additives(text: str) -> str:
    if not isinstance(text, str):
        return ""
    hits = []
    for a in ADD:
        if re.search(rf"\b{re.escape(a)}\b", text):
            hits.append(a)
    return ";".join(sorted(set(hits)))


def extract_amounts(text: str) -> str:
    if not isinstance(text, str):
        return ""

    parts = []

    for m in RE_MMMOL.findall(text):
        parts.append(f"{m[0]} {m[1]}")

    for m in RE_MG.findall(text):
        parts.append(f"{m} mg")

    for m in RE_ML.findall(text):
        parts.append(f"{m[0]} {m[1]}")

    for m in RE_VOLPCT.findall(text):
        v = m[0] if m[0] else m[1]
        if v:
            parts.append(f"{v} vol%")

    parts = list(dict.fromkeys(parts))[:12]
    return ";".join(parts)


def choose_text_column(df: pd.DataFrame) -> str:
    """
    Prefer the richest block of text if available.
    """
    for c in ["evidence_block", "best_recipe_block", "recipe_block", "method_block", "text", "raw_text"]:
        if c in df.columns:
            return c
    # if none exist, we still won't crash; we will create an empty text field
    return ""


def safe_cols(df: pd.DataFrame, wanted):
    return [c for c in wanted if c in df.columns]


def main():
    infile = pick_input_file()
    print(f"Step9J input: {infile}")

    df = pd.read_csv(infile)

    text_col = choose_text_column(df)
    if text_col == "":
        df["_tmp_text"] = ""
        text_col = "_tmp_text"
        print("Warning: no text column found (evidence_block/recipe_block/etc). Additives will be empty.")

    df[text_col] = df[text_col].fillna("").astype(str)

    df["additives_found"] = df[text_col].apply(extract_additives)
    df["amount_tokens"] = df[text_col].apply(extract_amounts)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTFILE, index=False)

    preview_cols = safe_cols(df, [
        "doi", "pdf_file",
        "additives_found", "amount_tokens", text_col
    ])

    print("\nPreview (first 8 rows):")
    if preview_cols:
        print(df[preview_cols].head(8).to_string(index=False))
    else:
        print(df.head(8).to_string(index=False))

    print("\nSaved:", OUTFILE)


if __name__ == "__main__":
    main()
