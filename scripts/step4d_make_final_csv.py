import re
import pandas as pd

INFILE = "step4c_best_recipe_extracted.csv"
OUTFILE = "final_ink_recipes.csv"

RE_TEMP_STRICT = re.compile(r"(\d+(?:\.\d+)?)\s*°\s*C")
RE_MOLARITY = re.compile(r"(\d+(?:\.\d+)?)\s*M\b")
RE_FILTER = re.compile(r"(0\.\d+)\s*(µm|um)\b", re.IGNORECASE)
RE_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)")
RE_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*(min|mins|minutes|h|hr|hours)\b", re.IGNORECASE)

def pick_realistic_floats(vals, lo, hi):
    out = []
    for v in vals:
        try:
            x = float(v)
            if lo <= x <= hi:
                out.append(x)
        except:
            pass
    return sorted(set(out))

df = pd.read_csv(INFILE)

rows = []
for _, r in df.iterrows():
    block = str(r.get("best_recipe_block", ""))

    temps = [t[0] for t in RE_TEMP_STRICT.findall(block)]
    mols = [m[0] for m in RE_MOLARITY.findall(block)]
    filts = [f[0] for f in RE_FILTER.findall(block)]
    ratios = [f"{a}:{b}" for a, b in RE_RATIO.findall(block)]
    times = [f"{a} {b}" for a, b in RE_TIME.findall(block)]

    temps_ok = pick_realistic_floats(temps, 10, 200)
    mols_ok = pick_realistic_floats(mols, 0.05, 3.0)
    filts_ok = pick_realistic_floats(filts, 0.1, 1.0)

    rows.append({
        "pdf_file": r.get("pdf_file", ""),
        "precursors": r.get("precursors", ""),
        "solvents": r.get("solvents", ""),
        "molarity_M": ";".join(str(x).rstrip("0").rstrip(".") for x in mols_ok),
        "solvent_ratio": ";".join(sorted(set(ratios))),
        "mix_temp_C": ";".join(str(x).rstrip("0").rstrip(".") for x in temps_ok),
        "times_found": ";".join(sorted(set(times))),
        "filter_um": ";".join(str(x).rstrip("0").rstrip(".") for x in filts_ok),
        "evidence_block": block[:1200],
    })

out = pd.DataFrame(rows)
out.to_csv(OUTFILE, index=False)

print(out[["pdf_file", "precursors", "solvents", "molarity_M", "solvent_ratio", "mix_temp_C", "times_found", "filter_um"]].to_string(index=False))
print(f"\nSaved: {OUTFILE}")
