import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INFILE = ROOT / "final_perovskite_ink_recipes_ML_READY.csv"
OUTFILE = ROOT / "final_perovskite_ink_recipes_ML_READY2.csv"

# --- Solvent vocabulary (extend as needed) ---
# Canonical form -> list of aliases to match in text
SOLVENT_ALIASES = {
    "DMF": ["DMF", "N,N-dimethylformamide", "dimethylformamide"],
    "DMSO": ["DMSO", "dimethyl sulfoxide", "dimethylsulfoxide"],
    "GBL": ["GBL", "gamma-butyrolactone", "γ-butyrolactone"],
    "GVL": ["GVL", "gamma-valerolactone", "γ-valerolactone"],
    "NMP": ["NMP", "N-methyl-2-pyrrolidone"],
    "ACN": ["ACN", "acetonitrile", "MeCN"],
    "2-ME": ["2-ME", "2ME", "2-methoxyethanol", "methoxyethanol"],
    "IPA": ["IPA", "isopropanol", "2-propanol"],
    "EtOH": ["EtOH", "ethanol"],
    "MeOH": ["MeOH", "methanol"],
}

# Explicit "system" patterns:
# DMF:DMSO, DMF/DMSO, DMF + DMSO, DMF and DMSO
RE_EXPLICIT_SYS = re.compile(
    r"\b(DMF|DMSO|GBL|GVL|NMP|ACN|MeCN|acetonitrile|2-ME|2ME|IPA|isopropanol|EtOH|ethanol|MeOH|methanol)\b"
    r"\s*[:/+\-]\s*"
    r"\b(DMF|DMSO|GBL|GVL|NMP|ACN|MeCN|acetonitrile|2-ME|2ME|IPA|isopropanol|EtOH|ethanol|MeOH|methanol)\b",
    re.IGNORECASE,
)

RE_AND_SYS = re.compile(
    r"\b(DMF|DMSO|GBL|GVL|NMP|ACN|MeCN|acetonitrile|2-ME|2ME|IPA|isopropanol|EtOH|ethanol|MeOH|methanol)\b"
    r"\s+(?:and|&)\s+"
    r"\b(DMF|DMSO|GBL|GVL|NMP|ACN|MeCN|acetonitrile|2-ME|2ME|IPA|isopropanol|EtOH|ethanol|MeOH|methanol)\b",
    re.IGNORECASE,
)

# Ratio candidates (exclude HH:MM:SS)
RE_RATIO = re.compile(r"\b(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\b")
RE_TIME_HHMMSS = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")
RE_TIME_MMSS = re.compile(r"^\d{1,3}:\d{2}$")
RE_DECIMAL = re.compile(r"^\d+\.\d+$")


def canonicalize_solvent(token: str) -> str:
    t = token.strip().lower()
    for canon, aliases in SOLVENT_ALIASES.items():
        for a in aliases:
            if t == a.lower():
                return canon
    # fallback for already-canonical tokens
    if t.upper() in SOLVENT_ALIASES:
        return t.upper()
    return token.strip().upper()


def infer_from_solvents_column(solvents_cell: str) -> str:
    """
    If df['solvents'] looks like 'DMF;DMSO' or 'DMSO;DMF;NMP',
    infer the system as the first two distinct solvents in order.
    """
    if not isinstance(solvents_cell, str):
        return ""
    raw = [x.strip() for x in solvents_cell.split(";") if x.strip()]
    canon = []
    for r in raw:
        c = canonicalize_solvent(r)
        if c and c not in canon:
            canon.append(c)
    if len(canon) >= 2:
        return f"{canon[0]}:{canon[1]}"
    return ""


def extract_system_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    m = RE_EXPLICIT_SYS.search(text)
    if m:
        a = canonicalize_solvent(m.group(1))
        b = canonicalize_solvent(m.group(2))
        if a != b:
            return f"{a}:{b}"

    m2 = RE_AND_SYS.search(text)
    if m2:
        a = canonicalize_solvent(m2.group(1))
        b = canonicalize_solvent(m2.group(2))
        if a != b:
            return f"{a}:{b}"

    # If no explicit pair, detect all solvents and use first two by appearance
    lowered = text.lower()
    positions = []
    for canon, aliases in SOLVENT_ALIASES.items():
        for a in aliases:
            idx = lowered.find(a.lower())
            if idx != -1:
                positions.append((idx, canon))
                break
    if not positions:
        return ""

    positions.sort(key=lambda x: x[0])
    ordered = []
    for _, c in positions:
        if c not in ordered:
            ordered.append(c)
    if len(ordered) >= 2:
        return f"{ordered[0]}:{ordered[1]}"
    return ""


def normalize_ratio(s: str) -> str:
    s = str(s).strip().replace(" ", "").strip(";,:|")
    if not s:
        return ""
    if RE_TIME_HHMMSS.match(s) or RE_TIME_MMSS.match(s) or RE_DECIMAL.match(s):
        return ""
    # must look like a:b
    if ":" not in s:
        return ""
    a, b = s.split(":", 1)
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


def extract_ratio_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Find first plausible ratio (after normalization)
    for m in RE_RATIO.finditer(text):
        cand = f"{m.group(1)}:{m.group(2)}"
        norm = normalize_ratio(cand)
        if norm:
            return norm
    return ""


def main():
    if not INFILE.exists():
        print(f"Missing input: {INFILE}")
        pd.DataFrame([]).to_csv(OUTFILE, index=False)
        return

    df = pd.read_csv(INFILE)

    # Pick best available evidence column
    text_col = None
    for c in ["best_recipe_block", "evidence_block", "best_recipe_text"]:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        # fallback: combine solvents + ratios columns (if they exist)
        parts = []
        for c in ["solvents", "solvent_ratio", "ratios"]:
            if c in df.columns:
                parts.append(df[c].fillna("").astype(str))
        if parts:
            df["_text"] = parts[0]
            for p in parts[1:]:
                df["_text"] = df["_text"] + " " + p
            text_col = "_text"
        else:
            df["_text"] = ""
            text_col = "_text"

    # Extract solvent system
    df["solvent_system_extracted"] = df[text_col].fillna("").astype(str).apply(extract_system_from_text)

    # If still empty, infer from solvents column
    if "solvents" in df.columns:
        mask_empty = df["solvent_system_extracted"].fillna("").astype(str).str.strip().eq("")
        df.loc[mask_empty, "solvent_system_extracted"] = df.loc[mask_empty, "solvents"].fillna("").astype(str).apply(infer_from_solvents_column)

    # Extract ratio (helper; later steps choose best)
    df["solvent_ratio_extracted"] = df[text_col].fillna("").astype(str).apply(extract_ratio_from_text)

    df.to_csv(OUTFILE, index=False)

    both = (
        df["solvent_system_extracted"].fillna("").astype(str).str.strip().ne("") &
        df["solvent_ratio_extracted"].fillna("").astype(str).str.strip().ne("")
    ).sum()

    print(f"Rows: {len(df)}")
    print(f"Rows with BOTH solvent_system_extracted + solvent_ratio_extracted: {both}")
    print(f"Saved: {OUTFILE.name}")


if __name__ == "__main__":
    main()
