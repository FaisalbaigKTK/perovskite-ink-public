import pandas as pd
import re

INFILE = "final_ink_recipes.csv"
OUTFILE = "final_perovskite_ink_recipes_only.csv"

df = pd.read_csv(INFILE)

def is_perovskite(row):
    prec = str(row.get("precursors", "")).lower()
    evidence = str(row.get("evidence_block", "")).lower()

    if any(x in prec for x in ["pbi2", "pbbr2", "pbcl2"]):
        return True
    if "perovskite" in evidence:
        return True
    return False

filtered = df[df.apply(is_perovskite, axis=1)]

filtered.to_csv(OUTFILE, index=False)

print(filtered[["pdf_file", "precursors", "solvents", "molarity_M", "solvent_ratio"]].to_string(index=False))
print(f"\nSaved: {OUTFILE}")
