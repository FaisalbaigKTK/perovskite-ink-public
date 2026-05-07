import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFILE = ROOT / "final_perovskite_ink_recipes_REAL_ONLY.csv"
OUTFILE = ROOT / "final_perovskite_ink_recipes_CLEAN.csv"

RE_TIME_HHMMSS = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")   # 100:01:00
RE_TIME_MMSS = re.compile(r"^\d{1,3}:\d{2}$")          # 0:39 (usually time)
RE_DECIMAL = re.compile(r"^\d+\.\d+$")                 # 4.031944444
RE_RATIO = re.compile(r"^\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?$")  # 4:1, 2.5:1.0

def normalize_ratio(s: str) -> str:
    s = s.strip()
    s = s.replace(" ", "")

    # strip weird leading/trailing punctuation
    s = s.strip(";,:|")

    # reject time-like and decimal-like
    if RE_TIME_HHMMSS.match(s) or RE_TIME_MMSS.match(s) or RE_DECIMAL.match(s):
        return ""

    # must look like a ratio
    if not RE_RATIO.match(s):
        return ""

    a, b = s.split(":")
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return ""

    # sanity bounds for solvent mixture ratios
    if af <= 0 or bf <= 0:
        return ""
    if af < 0.05 or bf < 0.05:
        return ""
    if af > 50 or bf > 50:
        return ""

    # normalize 4:01 -> 4:1, 1.00 -> 1
    def tidy_num(x):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x).rstrip("0").rstrip(".")

    return f"{tidy_num(af)}:{tidy_num(bf)}"


def main():
    if not INFILE.exists():
        print(f"Missing input: {INFILE}")
        pd.DataFrame([]).to_csv(OUTFILE, index=False)
        return

    df = pd.read_csv(INFILE)

    if "solvent_ratio" not in df.columns:
        # If upstream used a different name, still write file
        df.to_csv(OUTFILE, index=False)
        print("No solvent_ratio column found; saved unchanged CLEAN file.")
        return

    df["solvent_ratio"] = df["solvent_ratio"].fillna("").astype(str)
    df["solvent_ratio_clean"] = df["solvent_ratio"].apply(normalize_ratio)

    # keep original but prefer clean downstream
    df.to_csv(OUTFILE, index=False)

    kept = (df["solvent_ratio_clean"].astype(str).str.strip() != "").sum()
    print(f"Rows: {len(df)} | ratios kept (clean): {kept}")
    print(f"Saved: {OUTFILE.name}")


if __name__ == "__main__":
    main()
