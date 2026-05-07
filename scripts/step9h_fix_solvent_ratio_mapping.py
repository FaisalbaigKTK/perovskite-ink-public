import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INFILE = ROOT / "final_perovskite_ink_recipes_ML_READY3.csv"
OUTFILE = ROOT / "final_perovskite_ink_recipes_ML_READY4.csv"

RE_TIME_HHMMSS = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")
RE_TIME_MMSS = re.compile(r"^\d{1,3}:\d{2}$")
RE_DECIMAL = re.compile(r"^\d+\.\d+$")
RE_RATIO = re.compile(r"^\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?$")

def normalize_ratio(s: str) -> str:
    s = str(s).strip().replace(" ", "").strip(";,:|")
    if not s:
        return ""
    if RE_TIME_HHMMSS.match(s) or RE_TIME_MMSS.match(s) or RE_DECIMAL.match(s):
        return ""
    if not RE_RATIO.match(s):
        return ""
    a, b = s.split(":")
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return ""
    if af <= 0 or bf <= 0:
        return ""
    if af < 0.05 or bf < 0.05:
        return ""
    if af > 50 or bf > 50:
        return ""

    def tidy_num(x):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x).rstrip("0").rstrip(".")

    return f"{tidy_num(af)}:{tidy_num(bf)}"


def pick_best_ratio(row) -> str:
    """
    Prefer ratios that were already mapped strictly, then 'best anywhere', then raw fields.
    """
    candidates = []

    for col in [
        "solvent_ratio_final",
        "solvent_ratio_FINAL2",
        "solvent_ratio_strict",
        "solvent_ratio_best",
        "solvent_ratio_extracted",
        "solvent_ratio_clean",
        "solvent_ratio",
    ]:
        if col in row and pd.notna(row[col]):
            r = normalize_ratio(row[col])
            if r:
                candidates.append((col, r))

    # Return the first valid ratio following priority order above
    return candidates[0][1] if candidates else ""


def main():
    if not INFILE.exists():
        print(f"Missing input: {INFILE}")
        pd.DataFrame([]).to_csv(OUTFILE, index=False)
        return

    df = pd.read_csv(INFILE)

    # Ensure expected columns exist
    for c in ["solvent_system_extracted", "solvent_system_strict", "solvent_system_best"]:
        if c not in df.columns:
            df[c] = ""

    # Compute final ratio
    df["solvent_ratio_final"] = df.apply(pick_best_ratio, axis=1)

    # Compute a final solvent system (prefer best/strict/extracted)
    def pick_system(r):
        for c in ["solvent_system_best", "solvent_system_strict", "solvent_system_extracted"]:
            v = str(r.get(c, "")).strip()
            if v and v.lower() != "nan":
                return v
        return ""
    df["solvent_system_final"] = df.apply(pick_system, axis=1)

    df.to_csv(OUTFILE, index=False)

    # Print a useful summary
    both = (
        df["solvent_system_final"].fillna("").astype(str).str.strip().ne("") &
        df["solvent_ratio_final"].fillna("").astype(str).str.strip().ne("")
    ).sum()

    print(f"Rows: {len(df)}")
    print(f"Rows with BOTH solvent_system_final and solvent_ratio_final: {both}")
    print(f"Saved: {OUTFILE.name}")


if __name__ == "__main__":
    main()
