"""
step9k_export_tidy_dataset.py
==============================
Pipeline stage: 9k of 9 (final export)
Input:  final_perovskite_ink_recipes_ADDATIVES.csv
        (enriched records after all step9a–9j processing: solvent normalisation,
         ratio mapping, additive extraction, and LLM-enhanced fields merged)
Output: perovskite_ink_dataset_TIDY.csv   (all 72 records — the main dataset)
        perovskite_ink_dataset_TIDY_GOLD.csv  (15 manually validated records)

Purpose
-------
Produces the final tidy-format dataset deposited on Zenodo and described in the
paper. 'Tidy' means one row per ink formulation record, with canonical column
names, validated types, and no redundant intermediate columns.

Processing steps
----------------
1. Build 'solvent_system_final':
     Prefer any pre-existing solvent_system_* column (from step9g/9h).
     If absent, infer from the 'solvents' semicolon list (take first two
     unique entries, join with ':', e.g. 'DMF;DMSO;IPA' -> 'DMF:DMSO').

2. Build 'solvent_ratio_final':
     Try candidate ratio columns in priority order (step9h output first,
     then step9i, then raw 'ratios'). Validate each value with normalize_ratio():
     must be in N:M form, both N and M in [0.05, 50], not a time or decimal.

3. Compute 'recipe_class':
     FORMULATION_PRIMARY   — molarity_M AND solvent_ratio_final both present
     FORMULATION_SECONDARY — either molarity_M OR solvent_ratio_final present
     PROCESS_ONLY          — neither present (only temp/time/filter extracted)

4. Compute 'recipe_score' (0–10):
     +3 if molarity_M present
     +3 if solvent_ratio_final present
     +2 if solvent_system_final contains ':' (binary co-solvent)
     +1 if solvent_system_final present but no ':'
     +1 if additives_found present
     +1 if perovskite_formula present
     Score is capped at 10.

5. Compute 'recipe_tier':
     GOLD     — score >= 9
     STRONG   — score 7–8
     MODERATE — score 4–6
     WEAK     — score 0–3

6. Select the 9 canonical output columns and save TIDY.

7. Build GOLD subset:
     Records where recipe_class is PRIMARY or SECONDARY AND perovskite_formula,
     solvent_system_final, AND solvent_ratio_final are all non-empty.
     This is the manually validated subset used for precision/recall evaluation.

Canonical output columns (TIDY schema)
---------------------------------------
  pdf_file             : source PDF filename (traceability key)
  perovskite_formula   : stoichiometric formula string (LLM-extracted)
  solvent_system_final : normalised solvent system (e.g. 'DMF:DMSO')
  solvent_ratio_final  : validated v/v ratio (e.g. '4:1')
  additives_found      : semicolon-delimited additive list
  molarity_M           : primary precursor molarity (M)
  recipe_class         : FORMULATION_PRIMARY | SECONDARY | PROCESS_ONLY
  recipe_score         : integer 0–10 completeness score
  recipe_tier          : GOLD | STRONG | MODERATE | WEAK

Usage
-----
    python step9k_export_tidy_dataset.py
"""

import re
import pandas as pd
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

INFILE   = ROOT / "final_perovskite_ink_recipes_ADDATIVES.csv"
OUT_TIDY = ROOT / "perovskite_ink_dataset_TIDY.csv"
OUT_GOLD = ROOT / "perovskite_ink_dataset_TIDY_GOLD.csv"

# ── Compiled regular expressions for ratio validation ─────────────────────────
RE_TIME_HHMMSS = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")   # e.g. '1:30:00' — time, not ratio
RE_TIME_MMSS   = re.compile(r"^\d{1,3}:\d{2}$")          # e.g. '1:30' — time, not ratio
RE_DECIMAL     = re.compile(r"^\d+\.\d+$")               # bare decimal, not a ratio
RE_RATIO       = re.compile(r"^\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?$")  # valid N:M ratio


# ── Field helpers ─────────────────────────────────────────────────────────────

def nonempty(x) -> bool:
    """Return True if a value is a non-empty, non-NaN string.

    Args:
        x: Any value (string, float, None).

    Returns:
        True if the value is a meaningful string; False otherwise.
    """
    if x is None:
        return False
    s = str(x).strip()
    return s != "" and s.lower() != "nan"


def normalize_ratio(s: str) -> str:
    """Validate and normalise a solvent ratio string to canonical 'N:M' form.

    Rejects values that look like timestamps (1:30:00), bare decimals (4.0),
    or physically implausible ratios (coefficients outside [0.05, 50]).
    Formats valid integers without decimal points (e.g. '4.0:1.0' -> '4:1').

    Args:
        s: Raw ratio string extracted from the recipe text.

    Returns:
        Canonical ratio string (e.g. '4:1') or empty string if invalid.
    """
    s = str(s).strip().replace(" ", "").strip(";,:|")
    if not s:
        return ""
    # Reject time-like formats and bare decimals
    if RE_TIME_HHMMSS.match(s) or RE_TIME_MMSS.match(s) or RE_DECIMAL.match(s):
        return ""
    # Must match N:M pattern
    if not RE_RATIO.match(s):
        return ""
    a, b = s.split(":")
    try:
        af, bf = float(a), float(b)
    except (ValueError, TypeError):
        return ""
    # Reject zero, negative, or extreme values
    if af <= 0 or bf <= 0:
        return ""
    if af < 0.05 or bf < 0.05 or af > 50 or bf > 50:
        return ""

    def tidy_num(x: float) -> str:
        """Format a float as integer if whole, else as minimal decimal string."""
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x).rstrip("0").rstrip(".")

    return f"{tidy_num(af)}:{tidy_num(bf)}"


def infer_system_from_solvents(solvents_cell: str) -> str:
    """Infer a solvent system label from a semicolon-delimited solvent list.

    Takes the first two unique solvents and joins them with ':'.
    E.g. 'DMSO;DMF;NMP' -> 'DMSO:DMF'.
    Returns empty string for single-solvent or empty inputs.

    Args:
        solvents_cell: Semicolon-delimited string of solvent names.

    Returns:
        'SolventA:SolventB' string, or '' if fewer than 2 solvents.
    """
    if not isinstance(solvents_cell, str):
        return ""
    parts = [p.strip() for p in solvents_cell.split(";") if p.strip()]
    uniq = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    return f"{uniq[0]}:{uniq[1]}" if len(uniq) >= 2 else ""


def pick_first_present(row, cols) -> str:
    """Return the first non-empty value from a list of column names.

    Args:
        row:  A pandas Series for one record.
        cols: Ordered list of column names to try.

    Returns:
        First non-empty string value found, or '' if all are empty/NaN.
    """
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v and v.lower() != "nan":
                return v
    return ""


# ── Derived column builders ───────────────────────────────────────────────────

def build_solvent_system_final(row) -> str:
    """Build the canonical solvent_system_final value for one record.

    Priority: explicit system column > inferred from solvents list.

    Args:
        row: pandas Series for one record.

    Returns:
        Solvent system string (e.g. 'DMF:DMSO') or ''.
    """
    # Try columns produced by step9g / step9h (most reliable)
    v = pick_first_present(row, [
        "solvent_system_final", "solvent_system_best",
        "solvent_system_strict", "solvent_system_extracted"
    ])
    if nonempty(v):
        return v

    # Fall back: infer from the semicolon-delimited solvents column
    if "solvents" in row and nonempty(row["solvents"]):
        inferred = infer_system_from_solvents(str(row["solvents"]))
        if nonempty(inferred):
            return inferred

    return ""


def build_solvent_ratio_final(row) -> str:
    """Build the canonical solvent_ratio_final value for one record.

    Tries candidate columns in priority order and normalises the first
    valid ratio found using normalize_ratio().

    Args:
        row: pandas Series for one record.

    Returns:
        Normalised ratio string (e.g. '4:1') or ''.
    """
    for c in [
        "solvent_ratio_final", "solvent_ratio_FINAL2", "solvent_ratio_best",
        "solvent_ratio_strict", "solvent_ratio_extracted",
        "solvent_ratio_clean", "solvent_ratio", "ratios"
    ]:
        if c in row and pd.notna(row[c]):
            r = normalize_ratio(row[c])
            if r:
                return r
    return ""


# ── Classification functions ──────────────────────────────────────────────────

def recipe_class_simple(row) -> str:
    """Assign one of three recipe classes based on field completeness.

    A record with both molarity and ratio present is most useful for
    reproducibility (PRIMARY); one with only one of those is partially
    informative (SECONDARY); records with only process conditions (temp,
    time) but no quantitative formulation data are PROCESS_ONLY.

    Args:
        row: pandas Series after solvent_system/ratio columns are built.

    Returns:
        'FORMULATION_PRIMARY', 'FORMULATION_SECONDARY', or 'PROCESS_ONLY'.
    """
    has_m = nonempty(row.get("molarity_M", ""))
    has_r = nonempty(row.get("solvent_ratio_final", ""))
    if has_m and has_r:
        return "FORMULATION_PRIMARY"
    if has_m or has_r:
        return "FORMULATION_SECONDARY"
    return "PROCESS_ONLY"


def recipe_score_simple(row) -> int:
    """Compute a 0–10 completeness score for one recipe record.

    Scoring breakdown:
      +3 molarity_M present
      +3 solvent_ratio_final present
      +2 solvent_system_final contains ':' (binary co-solvent)
      +1 solvent_system_final present without ':'
      +1 additives_found present
      +1 perovskite_formula present
    Score is capped at 10.

    Args:
        row: pandas Series after all columns are built.

    Returns:
        Integer score in [0, 10].
    """
    score = 0
    if nonempty(row.get("molarity_M", "")):
        score += 3
    if nonempty(row.get("solvent_ratio_final", "")):
        score += 3

    # Reward binary co-solvent system (A:B) more than single solvent
    sysv = str(row.get("solvent_system_final", "")).strip()
    if ":" in sysv:
        score += 2
    elif sysv:
        score += 1

    if nonempty(row.get("additives_found", "")):
        score += 1
    if nonempty(row.get("perovskite_formula", "")):
        score += 1

    return min(score, 10)   # cap at 10


def tier(score: int) -> str:
    """Convert a numeric recipe_score to a named tier label.

    Thresholds:
      GOLD     >= 9  (manually validated subset)
      STRONG   >= 7
      MODERATE >= 4
      WEAK      < 4

    Args:
        score: Integer recipe_score in [0, 10].

    Returns:
        Tier label string.
    """
    if score >= 9:
        return "GOLD"
    if score >= 7:
        return "STRONG"
    if score >= 4:
        return "MODERATE"
    return "WEAK"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Build and export the TIDY and GOLD datasets.

    Reads the enriched additive file, builds all derived columns, selects
    the 9 canonical tidy columns, and writes TIDY + GOLD CSV files.
    The GOLD subset definition: recipe_class in {PRIMARY, SECONDARY} AND
    perovskite_formula, solvent_system_final, AND solvent_ratio_final all present.
    """
    if not INFILE.exists():
        print(f"[step9k] Missing input: {INFILE}")
        pd.DataFrame([]).to_csv(OUT_TIDY, index=False)
        pd.DataFrame([]).to_csv(OUT_GOLD, index=False)
        return

    df = pd.read_csv(INFILE)
    if df.empty:
        print("[step9k] Input is empty.")
        pd.DataFrame([]).to_csv(OUT_TIDY, index=False)
        pd.DataFrame([]).to_csv(OUT_GOLD, index=False)
        return

    # Ensure all required source columns exist (fill with '' if absent)
    for c in ["pdf_file", "perovskite_formula", "additives_found", "molarity_M", "solvents"]:
        if c not in df.columns:
            df[c] = ""

    # ── Step 1–2: Build canonical solvent columns ─────────────────────────────
    df["solvent_system_final"] = df.apply(build_solvent_system_final, axis=1)
    df["solvent_ratio_final"]  = df.apply(build_solvent_ratio_final,  axis=1)

    # ── Step 3–5: Compute class, score, tier ──────────────────────────────────
    df["recipe_class"] = df.apply(recipe_class_simple, axis=1)
    df["recipe_score"] = df.apply(recipe_score_simple, axis=1)
    df["recipe_tier"]  = df["recipe_score"].apply(tier)

    # ── Step 6: Select 9 canonical columns ───────────────────────────────────
    TIDY_COLS = [
        "pdf_file",
        "perovskite_formula",
        "solvent_system_final",
        "solvent_ratio_final",
        "additives_found",
        "molarity_M",
        "recipe_class",
        "recipe_score",
        "recipe_tier",
    ]
    tidy = df[TIDY_COLS].copy()
    tidy.to_csv(OUT_TIDY, index=False)

    # ── Step 7: GOLD subset ────────────────────────────────────────────────────
    # Records must have: recipe class = PRIMARY or SECONDARY (not PROCESS_ONLY)
    # AND all three key formulation fields non-empty.
    gold = tidy[
        tidy["recipe_class"].isin(["FORMULATION_PRIMARY", "FORMULATION_SECONDARY"])
        & tidy["perovskite_formula"].fillna("").astype(str).str.strip().ne("")
        & tidy["solvent_system_final"].fillna("").astype(str).str.strip().ne("")
        & tidy["solvent_ratio_final"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    gold.to_csv(OUT_GOLD, index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    both_present = (
        tidy["solvent_system_final"].fillna("").astype(str).str.strip().ne("")
        & tidy["solvent_ratio_final"].fillna("").astype(str).str.strip().ne("")
    ).sum()

    print(f"[step9k] Input rows         : {len(df)}")
    print(f"[step9k] TIDY rows          : {len(tidy)} -> {OUT_TIDY.name}")
    print(f"[step9k] System + ratio     : {both_present} records have both")
    print("[step9k] recipe_class:\n" + tidy["recipe_class"].value_counts().to_string())
    print("[step9k] recipe_tier:\n"  + tidy["recipe_tier"].value_counts().to_string())
    print(f"[step9k] GOLD rows          : {len(gold)} -> {OUT_GOLD.name}")


if __name__ == "__main__":
    main()
