import re
import pandas as pd
from pathlib import Path

INFILE = "final_perovskite_ink_recipes_CLEAN.csv"
OUTFILE = "final_perovskite_ink_recipes_STRUCTURED.csv"

# Catch common perovskite formulas like:
# MAPbI3, FAPbI3, CsPbI3, MAPbBr3, Cs0.05FA0.81MA0.14PbI2.55Br0.45, etc.
RE_FORMULA = re.compile(
    r"\b("
    r"(?:Cs|FA|MA|Rb|GA|PEA)?\s*\d*(?:\.\d+)?"
    r"(?:\([A-Za-z0-9\.\s]+\)\d*(?:\.\d+)?)?"
    r"(?:Cs|FA|MA|Rb|GA|PEA)?\s*\d*(?:\.\d+)?"
    r")?Pb"
    r"(?:I|Br|Cl)\d*(?:\.\d+)?"
    r"(?:(?:I|Br|Cl)\d*(?:\.\d+)?)?"
    r"\d*"
    r"\b",
    re.IGNORECASE
)

RE_MOLARITY = re.compile(r"(\d+(?:\.\d+)?)\s*M\b", re.IGNORECASE)

def safe_cols(df, wanted):
    return [c for c in wanted if c in df.columns]

def extract_first_formula(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = RE_FORMULA.search(text.replace(" ", ""))
    return m.group(0) if m else ""

def extract_molarity(text: str) -> str:
    if not isinstance(text, str):
        return ""
    vals = RE_MOLARITY.findall(text)
    # Keep unique, order-preserving
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return ";".join(out)

def main():
    infile = Path(INFILE)
    if not infile.exists():
        print(f"Missing input: {INFILE}")
        return

    df = pd.read_csv(INFILE)

    # Where do we pull evidence text from?
    # Prefer best_recipe_block if present, else fallback to any available text column
    text_col = None
    for c in ["best_recipe_block", "evidence_block", "best_recipe_text"]:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        # fallback: combine some columns
        df["_text"] = (
            df.get("precursors", "").astype(str) + " " +
            df.get("solvents", "").astype(str) + " " +
            df.get("solvent_ratio", "").astype(str)
        )
        text_col = "_text"

    df["perovskite_formula"] = df[text_col].astype(str).apply(extract_first_formula)

    # Keep molarity in a dedicated field. If you already have "molarity" from step4,
    # we also try to build a normalized molarity_M field.
    if "molarity_M" not in df.columns:
        # Derive from either 'molarity' field or the evidence text
        if "molarity" in df.columns:
            df["molarity_M"] = df["molarity"].astype(str).apply(lambda s: s.replace(";", ";"))
        else:
            df["molarity_M"] = df[text_col].astype(str).apply(extract_molarity)
    else:
        # Ensure it's string
        df["molarity_M"] = df["molarity_M"].astype(str)

    df.to_csv(OUTFILE, index=False)

    preview = safe_cols(df, ["year", "doi", "pdf_file", "perovskite_formula", "solvent_ratio", "molarity_M"])
    if preview:
        print(df[preview].head(25).to_string(index=False))
    else:
        print(df.head(25).to_string(index=False))

    print(f"\nSaved: {OUTFILE}")

if __name__ == "__main__":
    main()
