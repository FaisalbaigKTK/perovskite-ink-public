import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Use the richest "final" file right before export
INFILE = ROOT / "final_perovskite_ink_recipes_ADDATIVES.csv"

OUT_TIDY = ROOT / "perovskite_ink_dataset_TIDY.csv"
OUT_GOLD = ROOT / "perovskite_ink_dataset_TIDY_GOLD.csv"


RE_TIME_HHMMSS = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")
RE_TIME_MMSS = re.compile(r"^\d{1,3}:\d{2}$")
RE_DECIMAL = re.compile(r"^\d+\.\d+$")
RE_RATIO = re.compile(r"^\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?$")


def nonempty(x) -> bool:
    if x is None:
        return False
    s = str(x).strip()
    return s != "" and s.lower() != "nan"


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
    if af < 0.05 or bf < 0.05 or af > 50 or bf > 50:
        return ""

    def tidy_num(x):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x).rstrip("0").rstrip(".")

    return f"{tidy_num(af)}:{tidy_num(bf)}"


def infer_system_from_solvents(solvents_cell: str) -> str:
    """
    solvents like: 'DMSO;DMF;NMP' or 'DMF;DMSO'
    -> system: 'DMSO:DMF' (first two distinct solvents in order)
    """
    if not isinstance(solvents_cell, str):
        return ""
    parts = [p.strip() for p in solvents_cell.split(";") if p.strip()]
    uniq = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    if len(uniq) >= 2:
        return f"{uniq[0]}:{uniq[1]}"
    return ""


def pick_first_present(row, cols) -> str:
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v and v.lower() != "nan":
                return v
    return ""


def build_solvent_system_final(row) -> str:
    # Prefer explicit system if available
    v = pick_first_present(row, [
        "solvent_system_final", "solvent_system_best", "solvent_system_strict", "solvent_system_extracted"
    ])
    if nonempty(v):
        return v

    # Otherwise infer from solvents list
    if "solvents" in row and nonempty(row["solvents"]):
        inferred = infer_system_from_solvents(str(row["solvents"]))
        if nonempty(inferred):
            return inferred

    return ""


def build_solvent_ratio_final(row) -> str:
    # Prefer already-mapped/clean columns if present
    for c in [
        "solvent_ratio_final", "solvent_ratio_FINAL2", "solvent_ratio_best",
        "solvent_ratio_strict", "solvent_ratio_extracted", "solvent_ratio_clean",
        "solvent_ratio", "ratios"
    ]:
        if c in row and pd.notna(row[c]):
            r = normalize_ratio(row[c])
            if r:
                return r
    return ""


def recipe_class_simple(row) -> str:
    has_m = nonempty(row.get("molarity_M", ""))
    has_r = nonempty(row.get("solvent_ratio_final", ""))
    if has_m and has_r:
        return "FORMULATION_PRIMARY"
    if has_m or has_r:
        return "FORMULATION_SECONDARY"
    return "PROCESS_ONLY"


def recipe_score_simple(row) -> int:
    """
    Score using what TIDY has + inferred solvent_system_final.
    """
    score = 0
    if nonempty(row.get("molarity_M", "")):
        score += 3
    if nonempty(row.get("solvent_ratio_final", "")):
        score += 3

    # solvents count from solvent_system_final (A:B -> 2 solvents)
    sysv = str(row.get("solvent_system_final", "")).strip()
    if ":" in sysv:
        score += 2
    elif sysv:
        score += 1

    if nonempty(row.get("additives_found", "")):
        score += 1

    # formula present
    if nonempty(row.get("perovskite_formula", "")):
        score += 1

    return min(score, 10)


def tier(score: int) -> str:
    if score >= 9:
        return "GOLD"
    if score >= 7:
        return "STRONG"
    if score >= 4:
        return "MODERATE"
    return "WEAK"


def main():
    if not INFILE.exists():
        print(f"[step9k] Missing input: {INFILE}")
        pd.DataFrame([]).to_csv(OUT_TIDY, index=False)
        pd.DataFrame([]).to_csv(OUT_GOLD, index=False)
        return

    df = pd.read_csv(INFILE)
    if df.empty:
        print("[step9k] Input empty.")
        pd.DataFrame([]).to_csv(OUT_TIDY, index=False)
        pd.DataFrame([]).to_csv(OUT_GOLD, index=False)
        return

    # Ensure columns exist
    for c in ["pdf_file", "perovskite_formula", "additives_found", "molarity_M", "solvents"]:
        if c not in df.columns:
            df[c] = ""

    # Build final solvent system + ratio using ALL available fields
    df["solvent_system_final"] = df.apply(build_solvent_system_final, axis=1)
    df["solvent_ratio_final"] = df.apply(build_solvent_ratio_final, axis=1)

    # Build class, score, tier
    df["recipe_class"] = df.apply(recipe_class_simple, axis=1)
    df["recipe_score"] = df.apply(recipe_score_simple, axis=1)
    df["recipe_tier"] = df["recipe_score"].apply(tier)

    tidy = df[[
        "pdf_file",
        "perovskite_formula",
        "solvent_system_final",
        "solvent_ratio_final",
        "additives_found",
        "molarity_M",
        "recipe_class",
        "recipe_score",
        "recipe_tier"
    ]].copy()

    # Save tidy
    tidy.to_csv(OUT_TIDY, index=False)

    # GOLD file: practical definition (not too strict)
    gold = tidy[
        tidy["recipe_class"].isin(["FORMULATION_PRIMARY", "FORMULATION_SECONDARY"])
        & tidy["perovskite_formula"].fillna("").astype(str).str.strip().ne("")
        & tidy["solvent_system_final"].fillna("").astype(str).str.strip().ne("")
        & tidy["solvent_ratio_final"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    gold.to_csv(OUT_GOLD, index=False)

    both_present = (
        tidy["solvent_system_final"].fillna("").astype(str).str.strip().ne("")
        & tidy["solvent_ratio_final"].fillna("").astype(str).str.strip().ne("")
    ).sum()

    print(f"[step9k] Input rows: {len(df)}")
    print(f"[step9k] TIDY rows: {len(tidy)} -> {OUT_TIDY.name}")
    print(f"[step9k] both_present (system+ratio): {both_present}")
    print("[step9k] recipe_class counts:")
    print(tidy["recipe_class"].value_counts().to_string())
    print("[step9k] recipe_tier counts:")
    print(tidy["recipe_tier"].value_counts().to_string())
    print(f"[step9k] GOLD rows: {len(gold)} -> {OUT_GOLD.name}")


if __name__ == "__main__":
    main()
