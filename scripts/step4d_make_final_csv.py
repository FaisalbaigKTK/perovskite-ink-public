"""
step4d_make_final_csv.py
=========================
Pipeline stage: 4d of 9
Input:  step4c_best_recipe_extracted.csv
        (108 records — one per PDF — with raw best_recipe_block text and
         semicolon-delimited regex extractions for solvents, precursors,
         molarity, ratios, temps, times, filter_um)
Output: final_ink_recipes.csv
        (same 108 records with numeric fields cleaned, range-validated,
         and deduped; ready for the perovskite-filter step)

Purpose
-------
Converts the raw, semicolon-delimited extraction strings produced by step4c
into cleaned, validated numeric fields suitable for downstream analysis.

For each field the script:
  1. Re-extracts values from the best_recipe_block using strict regexes.
  2. Parses each match to float and discards physically implausible values
     (e.g. temperatures outside 10–200 °C, molarities outside 0.05–3.0 M).
  3. Deduplicates values within each field.
  4. Joins the validated values back to a semicolon-delimited string.

Output columns
--------------
  pdf_file       : source PDF path (from step4c)
  precursors     : semicolon-delimited chemical names (carried from step4c)
  solvents       : semicolon-delimited solvent names (carried from step4c)
  molarity_M     : validated molarity values in M (0.05–3.0 M range)
  solvent_ratio  : all N:M ratio strings found in the block
  mix_temp_C     : validated mixing temperatures in °C (10–200 °C range)
  times_found    : stirring/dissolution time strings (e.g. '2 h', '30 min')
  filter_um      : validated filter pore sizes in µm (0.1–1.0 µm range)
  evidence_block : first 1200 chars of the recipe text (for audit)

Usage
-----
    python step4d_make_final_csv.py
"""

import re
import pandas as pd

# ── File paths ────────────────────────────────────────────────────────────────
INFILE  = "step4c_best_recipe_extracted.csv"
OUTFILE = "final_ink_recipes.csv"

# ── Compiled regular expressions ──────────────────────────────────────────────
# Temperature: matches values like '60 °C', '100°C', '25 ° C'
RE_TEMP_STRICT = re.compile(r"(\d+(?:\.\d+)?)\s*°\s*C")

# Molarity: matches values like '1.3 M', '1M' (word boundary to avoid false hits)
RE_MOLARITY = re.compile(r"(\d+(?:\.\d+)?)\s*M\b")

# Filter pore size: matches values like '0.2 µm', '0.45 um'
RE_FILTER = re.compile(r"(0\.\d+)\s*(µm|um)\b", re.IGNORECASE)

# Solvent ratio: matches N:M patterns like '4:1', '3:1', '9:1'
RE_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)")

# Stirring/dissolution time: matches '2 h', '30 min', '12 hours', etc.
RE_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*(min|mins|minutes|h|hr|hours)\b", re.IGNORECASE)


def pick_realistic_floats(vals: list, lo: float, hi: float) -> list:
    """Filter a list of string values to those that are valid floats in [lo, hi].

    Used to discard physically implausible extracted numbers (e.g. a regex
    matching '100' when parsing temperatures might capture a page number;
    the 10–200 °C range excludes such outliers).

    Args:
        vals: List of string values extracted by regex.
        lo:   Minimum acceptable value (inclusive).
        hi:   Maximum acceptable value (inclusive).

    Returns:
        Sorted, deduplicated list of valid floats within [lo, hi].
    """
    out = []
    for v in vals:
        try:
            x = float(v)
            if lo <= x <= hi:
                out.append(x)
        except (ValueError, TypeError):
            pass
    return sorted(set(out))


# ── Main processing ───────────────────────────────────────────────────────────

df = pd.read_csv(INFILE)

rows = []
for _, r in df.iterrows():
    block = str(r.get("best_recipe_block", ""))

    # Re-extract numeric fields from the raw text block using strict regexes
    temps  = [t[0]            for t in RE_TEMP_STRICT.findall(block)]
    mols   = [m[0]            for m in RE_MOLARITY.findall(block)]
    filts  = [f[0]            for f in RE_FILTER.findall(block)]
    ratios = [f"{a}:{b}"      for a, b in RE_RATIO.findall(block)]
    times  = [f"{a} {b}"      for a, b in RE_TIME.findall(block)]

    # Validate numeric fields against physical plausibility ranges
    temps_ok = pick_realistic_floats(temps,  lo=10,   hi=200)   # °C: exclude cryogenic or pyrolytic values
    mols_ok  = pick_realistic_floats(mols,   lo=0.05, hi=3.0)   # M:  match Methods §2.6 plausibility check
    filts_ok = pick_realistic_floats(filts,  lo=0.1,  hi=1.0)   # µm: standard PTFE filter range

    # Format floats cleanly (remove trailing zeros, e.g. '1.00' -> '1')
    def fmt(x: float) -> str:
        """Format a float without trailing zeros after the decimal point."""
        return str(x).rstrip("0").rstrip(".")

    rows.append({
        "pdf_file":      r.get("pdf_file",   ""),
        "precursors":    r.get("precursors", ""),   # carry from step4c (already cleaned)
        "solvents":      r.get("solvents",   ""),   # carry from step4c
        "molarity_M":    ";".join(fmt(x) for x in mols_ok),
        "solvent_ratio": ";".join(sorted(set(ratios))),
        "mix_temp_C":    ";".join(fmt(x) for x in temps_ok),
        "times_found":   ";".join(sorted(set(times))),
        "filter_um":     ";".join(fmt(x) for x in filts_ok),
        "evidence_block": block[:1200],              # truncate for readability
    })

out = pd.DataFrame(rows)
out.to_csv(OUTFILE, index=False)

# Console summary
print(out[["pdf_file", "precursors", "solvents", "molarity_M",
           "solvent_ratio", "mix_temp_C", "times_found", "filter_um"]].to_string(index=False))
print(f"\nSaved {len(out)} rows -> {OUTFILE}")
