"""
pipeline.py  —  Complete perovskite ink extraction pipeline in ONE file.

This replaces the entire chain of step4d → step4e → step9b → step9c →
step9d → step9e → step9g → step9h → step9i → step9j → step9k scripts.

It has NO hardcoded paths — every path is relative to THIS script's location.

Usage (run from the scripts/ folder OR the repo root — both work):
    python pipeline.py

What it reads (must already exist):
    <repo_root>/data/03_extract/step4c_best_recipe_extracted.csv  (regex output)
    <repo_root>/data/03_extract/LLM_output.csv                    (LLM output)

What it writes:
    <repo_root>/perovskite_ink_dataset_TIDY.csv
    <repo_root>/perovskite_ink_dataset_TIDY_GOLD.csv
    <repo_root>/data/05_final/pipeline_intermediate.csv   (for debugging)
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ── Locate repo root regardless of where script is called from ───────────────
# Works whether you run:  python scripts/pipeline.py
#                    or:  python pipeline.py  (from repo root)
_HERE = Path(__file__).resolve().parent
if (_HERE / "data").exists():
    ROOT = _HERE          # called from repo root
elif (_HERE.parent / "data").exists():
    ROOT = _HERE.parent   # called from scripts/
else:
    ROOT = _HERE.parent   # fallback

DATA_DIR    = ROOT / "data"
EXTRACT_DIR = DATA_DIR / "03_extract"
FINAL_DIR   = DATA_DIR / "05_final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_4C  = EXTRACT_DIR / "step4c_best_recipe_extracted.csv"
INPUT_LLM = EXTRACT_DIR / "LLM_output.csv"
OUT_TIDY  = ROOT / "perovskite_ink_dataset_TIDY.csv"
OUT_GOLD  = ROOT / "perovskite_ink_dataset_TIDY_GOLD.csv"
OUT_DEBUG = FINAL_DIR / "pipeline_intermediate.csv"

print("=" * 60)
print("Perovskite Ink Pipeline — post-extraction processing")
print(f"Repo root: {ROOT}")
print(f"Reading:   {INPUT_4C.name}")
print(f"Reading:   {INPUT_LLM.name}")
print("=" * 60)

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if not INPUT_4C.exists():
    print(f"\nERROR: step4c output not found:\n  {INPUT_4C}")
    print("Run step4c_refine_best_recipe.py first.")
    sys.exit(1)

if not INPUT_LLM.exists():
    print(f"\nWARNING: LLM_output.csv not found:\n  {INPUT_LLM}")
    print("Continuing with regex-only extraction (no LLM enrichment).")
    llm_available = False
else:
    llm_available = True

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_4C)
print(f"\n[Step 1] Loaded {len(df)} records from step4c")

if llm_available:
    llm = pd.read_csv(INPUT_LLM)
    llm_ok = llm[llm["llm_status"] == "ok"].copy()
    print(f"[Step 1] Loaded {len(llm_ok)} LLM-ok records")
else:
    llm_ok = pd.DataFrame()

# ── Step 2: Filter real perovskite ink recipes ────────────────────────────────
# A record passes if it has B-site halide salt OR the word 'perovskite'

PB_SALTS = ["pbi2", "pbbr2", "pbcl2", "sni2", "snbr2"]
A_SITE   = ["mai", "mabr", "macl", "fai", "fabr", "csi", "csbr", "peai", "peabr",
             "methylammonium", "formamidinium", "cesium"]
SOLVENTS = ["dmso", "dmf", "gbl", "gvl", "nmp", "acn", "ipa"]
RECIPE_VERBS = ["dissolv", "prepar", "precursor", "solution", "ink", "stirr", "filter"]
BAD_CONTEXT  = ["antisolvent", "anti-solvent", "washing", "rinsing", "cleaning", "etching"]

def real_ink_score(row) -> int:
    parts = [
        str(row.get("best_recipe_block", "") or ""),
        str(row.get("precursors",         "") or ""),
        str(row.get("solvents",           "") or ""),
        str(row.get("molarity",           "") or ""),
    ]
    text = " ".join(parts).lower()
    score = 0
    if any(s in text for s in PB_SALTS):    score += 5
    if any(s in text for s in A_SITE):      score += 3
    if any(s in text for s in SOLVENTS):    score += 2
    if "perovskite" in text:                 score += 2
    if re.search(r"\b\d+\.?\d*\s*m\b", text): score += 2
    if any(v in text for v in RECIPE_VERBS): score += 1
    # Penalise if context is only antisolvent / cleaning
    if any(b in text for b in BAD_CONTEXT) and score < 5: score -= 3
    return score

df["_real_score"] = df.apply(real_ink_score, axis=1)
real = df[df["_real_score"] >= 4].copy()
print(f"[Step 2] Real-ink filter: {len(real)} / {len(df)} pass")

# ── Step 3: Extract and clean molarity ────────────────────────────────────────
RE_MOL = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mol\s*[lL][-−]?1|mol/[lL]|M)\b")
RE_MM  = re.compile(r"\b(\d+(?:\.\d+)?)\s*mM\b")

def clean_molarity(row) -> str:
    text = " ".join([
        str(row.get("best_recipe_block", "") or ""),
        str(row.get("molarity",           "") or ""),
    ])
    hits = []
    for m in RE_MOL.finditer(text):
        try:
            v = float(m.group(1))
            if 0.05 <= v <= 3.0:
                hits.append(v)
        except: pass
    for m in RE_MM.finditer(text):
        try:
            v = float(m.group(1)) / 1000.0   # convert mM -> M
            if 0.05 <= v <= 3.0:
                hits.append(v)
        except: pass
    if not hits:
        return ""
    # Return median value (avoids noise from multi-component concentration lists)
    hits.sort()
    return str(round(hits[len(hits) // 2], 3)).rstrip("0").rstrip(".")

real["molarity_M"] = real.apply(clean_molarity, axis=1)
print(f"[Step 3] Molarity extracted (regex): {real['molarity_M'].ne('').sum()} records")

# ── Step 4: Extract solvent system ────────────────────────────────────────────
SOLVENT_CANON = {
    "dmf":              "DMF",   "n,n-dimethylformamide": "DMF", "dimethylformamide": "DMF",
    "dmso":             "DMSO",  "dimethyl sulfoxide":    "DMSO","dimethylsulfoxide":  "DMSO",
    "gbl":              "GBL",   "gamma-butyrolactone":   "GBL", "γ-butyrolactone":    "GBL",
    "gvl":              "GVL",   "gamma-valerolactone":   "GVL",
    "nmp":              "NMP",   "n-methyl-2-pyrrolidone":"NMP",
    "acn":              "ACN",   "acetonitrile":          "ACN", "mecn":               "ACN",
    "ipa":              "IPA",   "isopropanol":           "IPA", "2-propanol":         "IPA",
    "etoh":             "EtOH",  "ethanol":               "EtOH",
    "meoh":             "MeOH",  "methanol":              "MeOH",
    "2-me":             "2-ME",  "2-methoxyethanol":      "2-ME","methoxyethanol":     "2-ME",
    "chlorobenzene":    "CB",    "cb":                    "CB",
    "diethyl ether":    "DEE",
    "toluene":          "Tol",
}

RE_SOLV_PAIR = re.compile(
    r"\b(DMF|DMSO|GBL|GVL|NMP|ACN|MeCN|acetonitrile|2-ME|2ME|IPA|isopropanol|"
    r"EtOH|ethanol|MeOH|methanol|chlorobenzene)\s*[:/+,]\s*"
    r"(DMF|DMSO|GBL|GVL|NMP|ACN|MeCN|acetonitrile|2-ME|2ME|IPA|isopropanol|"
    r"EtOH|ethanol|MeOH|methanol|chlorobenzene)\b",
    re.IGNORECASE,
)

def canon_solv(s: str) -> str:
    return SOLVENT_CANON.get(s.lower().strip(), s.strip().upper())

def extract_solvent_system(row) -> str:
    text = str(row.get("best_recipe_block", "") or "")
    # Try to find explicit A:B or A/B pairs first
    m = RE_SOLV_PAIR.search(text)
    if m:
        return f"{canon_solv(m.group(1))}:{canon_solv(m.group(2))}"
    # Fall back to semicolon-delimited solvents column
    s = str(row.get("solvents", "") or "")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    normed = []
    for p in parts:
        c = canon_solv(p)
        if c not in normed:
            normed.append(c)
    if len(normed) >= 2:
        return f"{normed[0]}:{normed[1]}"
    if len(normed) == 1:
        return normed[0]
    return ""

real["solvent_system_final"] = real.apply(extract_solvent_system, axis=1)
print(f"[Step 4] Solvent system extracted: {real['solvent_system_final'].ne('').sum()} records")

# ── Step 5: Extract solvent ratio ─────────────────────────────────────────────
RE_RATIO = re.compile(r"\b(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\b")
RE_TIME  = re.compile(r"^\d{1,3}:\d{2}$")   # reject time-like "1:30"

def clean_ratio(val: str) -> str:
    val = str(val or "").strip()
    if not val or val.lower() in ("nan", "none", "null"):
        return ""
    hits = RE_RATIO.findall(val)
    for a, b in hits:
        af, bf = float(a), float(b)
        if 0.05 <= af <= 50 and 0.05 <= bf <= 50:
            ia = int(af) if af == int(af) else af
            ib = int(bf) if bf == int(bf) else bf
            candidate = f"{ia}:{ib}"
            if not RE_TIME.match(candidate):
                return candidate
    return ""

def best_ratio(row) -> str:
    # Try ratio from recipe block text first
    block = str(row.get("best_recipe_block", "") or "")
    # Look for ratio near a solvent name
    sys = str(row.get("solvent_system_final", "") or "")
    if sys:
        # Search for ratio within 100 chars of solvent name in block
        solvents_in_sys = sys.split(":")
        for sv in solvents_in_sys:
            idx = block.lower().find(sv.lower())
            if idx >= 0:
                window = block[max(0, idx-30):idx+80]
                r = clean_ratio(window)
                if r:
                    return r
    # Fall back to ratios column
    r = clean_ratio(str(row.get("ratios", "") or ""))
    if r:
        return r
    # Last resort: any ratio in block
    r = clean_ratio(block[:2000])
    return r

real["solvent_ratio_final"] = real.apply(best_ratio, axis=1)
print(f"[Step 5] Solvent ratio extracted:   {real['solvent_ratio_final'].ne('').sum()} records")

# ── Step 6: Extract additives ─────────────────────────────────────────────────
ADDITIVE_KEYWORDS = [
    "MACl", "PEABr", "PEAI", "CsI", "RbI", "NH4Cl", "NH4SCN",
    "HI", "HBr", "HCl", "YCl3", "TTDDA", "NMP",
    "Pb(SCN)2", "guanidinium", "GAI", "tBP",
    "chlorobenzene", "toluene", "diethyl ether",
    "MABr", "FABr", "Cs2CO3",
]

def extract_additives(row) -> str:
    text = " ".join([
        str(row.get("best_recipe_block", "") or ""),
        str(row.get("precursors",         "") or ""),
    ])
    found = []
    for kw in ADDITIVE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            if kw not in found:
                found.append(kw)
    return ";".join(found)

real["additives_found"] = real.apply(extract_additives, axis=1)
print(f"[Step 6] Additives found:           {real['additives_found'].ne('').sum()} records")

# ── Step 7: Merge LLM output ──────────────────────────────────────────────────
def pdf_key(x: str) -> str:
    return str(x).replace("\\", "/").split("/")[-1].replace(".pdf", "").strip()

real["_pdf_key"] = real["pdf_file"].apply(pdf_key)

if not llm_ok.empty:
    llm_ok = llm_ok.copy()
    llm_ok["_pdf_key"] = llm_ok["pdf_file"].apply(pdf_key)

    # Columns to bring from LLM
    llm_cols = ["_pdf_key", "perovskite_formula", "solvent_ratio",
                "molarity_M", "additives", "mix_temp_C", "stir_time",
                "filter_um", "extraction_confidence"]
    llm_merge = llm_ok[[c for c in llm_cols if c in llm_ok.columns]].copy()

    merged = real.merge(llm_merge, on="_pdf_key", how="left", suffixes=("", "_llm"))

    def nonempty(x) -> bool:
        return bool(x) and str(x).strip().lower() not in ("", "nan", "none", "null", "[]")

    # LLM fills gaps; regex values take priority if they exist
    if "molarity_M_llm" in merged.columns:
        mask = ~merged["molarity_M"].apply(nonempty) & merged["molarity_M_llm"].apply(nonempty)
        merged.loc[mask, "molarity_M"] = merged.loc[mask, "molarity_M_llm"]

    if "solvent_ratio" in merged.columns:
        mask = ~merged["solvent_ratio_final"].apply(nonempty) & merged["solvent_ratio"].apply(nonempty)
        merged.loc[mask, "solvent_ratio_final"] = merged.loc[mask, "solvent_ratio"].apply(clean_ratio)

    if "perovskite_formula" in merged.columns:
        merged["perovskite_formula"] = merged["perovskite_formula"].fillna("")
        merged["perovskite_formula"] = merged["perovskite_formula"].replace(
            ["null", "None", "nan"], "")

    # Merge additives: union of regex and LLM
    if "additives" in merged.columns:
        def merge_additives(regex_val, llm_val):
            regex_set = set(x.strip() for x in str(regex_val or "").split(";") if x.strip())
            llm_val   = str(llm_val or "")
            # Parse LLM list (may be Python list repr)
            try:
                import ast
                llm_list = ast.literal_eval(llm_val)
                llm_set  = set(x.strip() for x in llm_list if x.strip())
            except Exception:
                llm_set = set(x.strip() for x in llm_val.split(";") if x.strip())
            combined = sorted(regex_set | llm_set)
            return ";".join(combined)
        merged["additives_found"] = merged.apply(
            lambda r: merge_additives(r["additives_found"], r.get("additives", "")), axis=1)

else:
    merged = real.copy()
    merged["perovskite_formula"]   = ""
    merged["extraction_confidence"] = ""

print(f"[Step 7] LLM merge complete")
has_f = merged["perovskite_formula"].apply(lambda x: nonempty(x) if 'nonempty' in dir() else bool(x))
has_m = merged["molarity_M"].apply(lambda x: bool(str(x).strip()) and str(x).strip() not in ("", "nan", "None"))
print(f"  Formula present:  {has_f.sum()} / {len(merged)}")
print(f"  Molarity present: {has_m.sum()} / {len(merged)}")

# ── Step 8: Recipe class, score, tier ─────────────────────────────────────────
def ne(x) -> bool:
    """True if value is non-empty and not a null placeholder."""
    return bool(x) and str(x).strip().lower() not in ("", "nan", "none", "null", "[]", "0")

def recipe_class(row) -> str:
    hm = ne(row.get("molarity_M", ""))
    hr = ne(row.get("solvent_ratio_final", ""))
    if hm and hr: return "FORMULATION_PRIMARY"
    if hm or hr:  return "FORMULATION_SECONDARY"
    return "PROCESS_ONLY"

def recipe_score(row) -> int:
    s = 0
    if ne(row.get("molarity_M",           "")): s += 3
    if ne(row.get("solvent_ratio_final",   "")): s += 3
    sys_v = str(row.get("solvent_system_final", ""))
    if ":" in sys_v: s += 2
    elif sys_v:      s += 1
    if ne(row.get("additives_found",       "")): s += 1
    if ne(row.get("perovskite_formula",    "")): s += 1
    return min(s, 10)

def tier(s: int) -> str:
    if s >= 9: return "GOLD"
    if s >= 7: return "STRONG"
    if s >= 4: return "MODERATE"
    return "WEAK"

merged["recipe_class"] = merged.apply(recipe_class, axis=1)
merged["recipe_score"] = merged.apply(recipe_score, axis=1)
merged["recipe_tier"]  = merged["recipe_score"].apply(tier)

# ── Step 9: Build TIDY and GOLD datasets ──────────────────────────────────────
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

# Keep only TIDY_COLS that actually exist
tidy_cols_present = [c for c in TIDY_COLS if c in merged.columns]
tidy = merged[tidy_cols_present].copy()

# GOLD: class is PRIMARY or SECONDARY, AND formula + system + ratio all present
gold = tidy[
    tidy["recipe_class"].isin(["FORMULATION_PRIMARY", "FORMULATION_SECONDARY"])
    & tidy["perovskite_formula"].fillna("").str.strip().ne("")
    & ~tidy["perovskite_formula"].isin(["null", "None", "nan"])
    & tidy["solvent_system_final"].fillna("").str.strip().ne("")
    & tidy["solvent_ratio_final"].fillna("").str.strip().ne("")
].copy()

# ── Save debug intermediate ───────────────────────────────────────────────────
merged.to_csv(OUT_DEBUG, index=False)

# ── Save final outputs ────────────────────────────────────────────────────────
tidy.to_csv(OUT_TIDY, index=False)
gold.to_csv(OUT_GOLD, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PIPELINE COMPLETE")
print(f"{'='*60}")
print(f"Input records (step4c)  : {len(df)}")
print(f"After real-ink filter   : {len(real)}")
print(f"TIDY dataset            : {len(tidy)} records")
print(f"GOLD dataset            : {len(gold)} records")
print()
print("recipe_class breakdown:")
print(tidy["recipe_class"].value_counts().to_string())
print()
print("recipe_tier breakdown:")
print(tidy["recipe_tier"].value_counts().to_string())
print()
print("Solvent systems (top 10):")
print(tidy["solvent_system_final"].value_counts().head(10).to_string())
print()
has_formula = tidy["perovskite_formula"].fillna("").str.strip().ne("") & \
              ~tidy["perovskite_formula"].isin(["null","None","nan"])
has_molarity = tidy["molarity_M"].fillna("").str.strip().ne("") & \
               ~tidy["molarity_M"].isin(["null","None","nan","0"])
print(f"Formula present         : {has_formula.sum()} / {len(tidy)} ({has_formula.sum()/len(tidy)*100:.0f}%)")
print(f"Molarity present        : {has_molarity.sum()} / {len(tidy)} ({has_molarity.sum()/len(tidy)*100:.0f}%)")
print(f"Additives found         : {tidy['additives_found'].fillna('').ne('').sum()} records")
print()
print(f"Saved: {OUT_TIDY}")
print(f"Saved: {OUT_GOLD}")
print(f"Debug: {OUT_DEBUG}")
