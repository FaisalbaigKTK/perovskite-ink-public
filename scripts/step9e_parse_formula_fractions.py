import re
import pandas as pd
from pathlib import Path

INFILE = "final_perovskite_ink_recipes_STRUCTURED.csv"
OUTFILE = "final_perovskite_ink_recipes_ML_READY.csv"

# --- Helpers ---
def safe_cols(df, wanted):
    return [c for c in wanted if c in df.columns]

def _parse_token_amount(formula: str, token: str) -> float:
    """
    Find amount after token, e.g. Cs0.05 -> 0.05, MA -> default 1 if present without number.
    """
    # token followed by optional number
    m = re.search(rf"{token}(\d+(?:\.\d+)?)", formula)
    if m:
        return float(m.group(1))
    # token present but no explicit coefficient => 1
    if re.search(rf"{token}\b", formula):
        return 1.0
    return 0.0

def parse_perovskite_fractions(formula: str):
    """
    Returns (Cs, FA, MA, I, Br, Cl) fractions normalized within their sublattices:
    - A-site normalized over Cs+FA+MA (if any present)
    - X-site normalized over I+Br+Cl (if any present)

    If formula missing / cannot parse, returns NaNs.
    """
    if not isinstance(formula, str) or not formula.strip():
        return (None, None, None, None, None, None)

    f = formula.replace(" ", "")

    # Handle parentheses like Cs0.05(FA0.87MA0.13)0.95Pb(I0.9Br0.1)3
    # Expand A-site group if present
    # A-group pattern: (FA0.87MA0.13)0.95
    a_group = re.search(r"\((FA\d+(?:\.\d+)?)(MA\d+(?:\.\d+)?)\)(\d+(?:\.\d+)?)", f)
    Cs = _parse_token_amount(f, "Cs")
    FA = _parse_token_amount(f, "FA")
    MA = _parse_token_amount(f, "MA")

    if a_group:
        FA_in = float(re.search(r"FA(\d+(?:\.\d+)?)", a_group.group(1)).group(1))
        MA_in = float(re.search(r"MA(\d+(?:\.\d+)?)", a_group.group(2)).group(1))
        mult = float(a_group.group(3))
        # If FA/MA outside group exist, still add them (rare)
        FA = FA + FA_in * mult
        MA = MA + MA_in * mult

    # X-site group: (I0.9Br0.1)3
    x_group = re.search(r"\((I\d+(?:\.\d+)?)(Br\d+(?:\.\d+)?)(?:Cl\d+(?:\.\d+)?)?\)(\d+(?:\.\d+)?)", f)
    I = _parse_token_amount(f, "I")
    Br = _parse_token_amount(f, "Br")
    Cl = _parse_token_amount(f, "Cl")

    if x_group:
        I_in = float(re.search(r"I(\d+(?:\.\d+)?)", x_group.group(1)).group(1))
        Br_in = float(re.search(r"Br(\d+(?:\.\d+)?)", x_group.group(2)).group(1))
        mult = float(x_group.group(3))
        I = I + I_in * mult
        Br = Br + Br_in * mult
        # Cl in group (optional)
        mcl = re.search(r"Cl(\d+(?:\.\d+)?)", x_group.group(0))
        if mcl:
            Cl = Cl + float(mcl.group(1)) * mult

    # Normalize A-site
    a_sum = Cs + FA + MA
    if a_sum > 0:
        Cs_f = Cs / a_sum
        FA_f = FA / a_sum
        MA_f = MA / a_sum
    else:
        Cs_f = FA_f = MA_f = None

    # Normalize X-site
    x_sum = I + Br + Cl
    if x_sum > 0:
        I_f = I / x_sum
        Br_f = Br / x_sum
        Cl_f = Cl / x_sum
    else:
        I_f = Br_f = Cl_f = None

    return (Cs_f, FA_f, MA_f, I_f, Br_f, Cl_f)

def main():
    infile = Path(INFILE)
    if not infile.exists():
        print(f"Missing input: {INFILE}")
        return

    df = pd.read_csv(INFILE)

    # Ensure perovskite_formula exists
    if "perovskite_formula" not in df.columns:
        df["perovskite_formula"] = ""

    # Compute fractions
    frac = df["perovskite_formula"].apply(parse_perovskite_fractions)
    df["Cs_frac"] = frac.apply(lambda x: x[0])
    df["FA_frac"] = frac.apply(lambda x: x[1])
    df["MA_frac"] = frac.apply(lambda x: x[2])
    df["I_frac"]  = frac.apply(lambda x: x[3])
    df["Br_frac"] = frac.apply(lambda x: x[4])
    df["Cl_frac"] = frac.apply(lambda x: x[5])

    df.to_csv(OUTFILE, index=False)

    preview_cols = safe_cols(df, [
        "year","doi","pdf_file","perovskite_formula",
        "Cs_frac","FA_frac","MA_frac","I_frac","Br_frac","Cl_frac"
    ])
    if preview_cols:
        print(df[preview_cols].head(20).to_string(index=False))
    else:
        print(df.head(20).to_string(index=False))

    print(f"\nSaved: {OUTFILE}")

if __name__ == "__main__":
    main()
