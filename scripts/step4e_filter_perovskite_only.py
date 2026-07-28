"""
step4e_filter_perovskite_only.py
==================================
Pipeline stage: 4e of 9
Input:  final_ink_recipes.csv
        (108 records from step4d — all papers with any extracted recipe block,
         including non-perovskite materials)
Output: final_perovskite_ink_recipes_only.csv
        (subset confirmed to contain halide perovskite chemistry)

Purpose
-------
Filters the recipe dataset to retain only records that are confirmed to contain
halide perovskite precursor chemistry, discarding records from papers about
other materials (e.g. oxide perovskites, OLEDs, silicon solar cells) whose
Methods sections contained incidental matches to the anchor keywords in step4c.

Filtering criteria (a record is retained if ANY of the following are true):
  1. The 'precursors' field contains a lead or tin halide salt:
       PbI2, PbBr2, PbCl2  — lead halide B-site precursors
       SnI2, SnBr2          — tin halide B-site precursors (lead-free)
  2. The 'evidence_block' text contains the word 'perovskite' (case-insensitive).

Records failing both checks are discarded as false positives from the anchor
keyword search.

Retention statistics (reported run): 71 of 108 records pass (65.7%).

Usage
-----
    python step4e_filter_perovskite_only.py
"""

import pandas as pd

# ── File paths ────────────────────────────────────────────────────────────────
INFILE  = "final_ink_recipes.csv"
OUTFILE = "final_perovskite_ink_recipes_only.csv"

# ── Filter keyword sets ───────────────────────────────────────────────────────
# B-site halide salts that unambiguously indicate a halide perovskite precursor ink.
# Stored as lowercase for case-insensitive matching.
PEROVSKITE_PRECURSOR_KEYWORDS = {"pbi2", "pbbr2", "pbcl2", "sni2", "snbr2"}


def is_perovskite(row) -> bool:
    """Determine whether a recipe record belongs to a halide perovskite paper.

    Applies two independent tests:
      (a) Precursor field test: any entry in the semicolon-delimited 'precursors'
          string matches a known lead or tin halide B-site salt.
      (b) Full-text keyword test: the evidence_block contains the word 'perovskite'.

    Args:
        row: A pandas Series corresponding to one recipe record.

    Returns:
        True if the record should be retained; False if it should be discarded.
    """
    # Test (a) — B-site halide precursor present in extracted precursor list
    prec = str(row.get("precursors", "")).lower()
    if any(kw in prec for kw in PEROVSKITE_PRECURSOR_KEYWORDS):
        return True

    # Test (b) — word 'perovskite' appears anywhere in the recipe text block
    evidence = str(row.get("evidence_block", "")).lower()
    if "perovskite" in evidence:
        return True

    return False


# ── Load, filter, and save ────────────────────────────────────────────────────
df = pd.read_csv(INFILE)

n_before = len(df)
filtered = df[df.apply(is_perovskite, axis=1)].copy()
n_after  = len(filtered)

filtered.to_csv(OUTFILE, index=False)

# Console summary
print(filtered[["pdf_file", "precursors", "solvents", "molarity_M", "solvent_ratio"]].to_string(index=False))
print(f"\nRetained : {n_after} / {n_before} records ({n_after/n_before*100:.1f}%)")
print(f"Discarded: {n_before - n_after} records (non-perovskite or insufficient evidence)")
print(f"Saved    : {OUTFILE}")
