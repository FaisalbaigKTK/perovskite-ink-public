import re
from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

LINKS_FILE = DATA_DIR / "02_downloads" / "step3b_pdf_links.csv"
OUTFILE = ROOT / "step4c_best_recipe_extracted.csv"

SOLVENTS = [
    "DMSO", "DMF", "GBL", "GVL", "NMP", "ACN", "acetonitrile",
    "2-ME", "2ME", "2-methoxyethanol", "methoxyethanol",
    "IPA", "isopropanol", "ethanol", "MeOH", "methanol",
]
PRECURSORS = [
    "PbI2", "PbBr2", "PbCl2", "SnI2", "SnBr2",
    "MAI", "MABr", "MACl", "FAI", "FABr", "CsI", "CsBr",
    "PEAI", "PEABr",
]

ANCHORS = [
    "precursor solution", "perovskite precursor", "perovskite ink", "ink",
    "prepared by dissolving", "was dissolved", "were dissolved", "dissolved in",
    "solution was prepared", "ink was prepared", "formulated", "stirred", "filtered",
]

RE_NUM_RATIO = re.compile(r"\b\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\b")
RE_VV = re.compile(r"\b(v\/v|vol\/vol|volume\/volume)\b", re.IGNORECASE)
RE_WW = re.compile(r"\b(w\/w|wt\/wt|weight\/weight)\b", re.IGNORECASE)

RE_M = re.compile(r"\b\d+(?:\.\d+)?\s*M\b", re.IGNORECASE)
RE_MOL_L = re.compile(r"\b\d+(?:\.\d+)?\s*(mol\s*L[-−]?1|mol\/L)\b", re.IGNORECASE)
RE_MM = re.compile(r"\b\d+(?:\.\d+)?\s*mM\b", re.IGNORECASE)

RE_TEMP = re.compile(r"\b\d+(?:\.\d+)?\s*°\s*C\b")
RE_TIME = re.compile(r"\b\d+(?:\.\d+)?\s*(s|sec|secs|seconds|min|mins|minutes|h|hr|hrs|hours)\b", re.IGNORECASE)
RE_FILTER = re.compile(r"\b0\.\d+\s*(µm|um)\b", re.IGNORECASE)

def shortlist_pdfs():
    if not LINKS_FILE.exists():
        raise FileNotFoundError(f"Missing links file: {LINKS_FILE}")

    df = pd.read_csv(LINKS_FILE)

    # Keep only successful downloads if status exists
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.startswith("downloaded", na=False)].copy()

    if "pdf_file" not in df.columns:
        raise ValueError("step3b_pdf_links.csv is missing 'pdf_file' column.")

    # Drop NaN and empty strings safely
    pdf_series = df["pdf_file"].dropna().astype(str)

    paths = []
    for p in pdf_series.tolist():
        p = p.strip()
        if not p:
            continue

        pp = Path(p)
        if not pp.is_absolute():
            pp = ROOT / pp

        if pp.exists() and pp.is_file():
            paths.append(pp)

    # Deduplicate preserving order
    seen = set()
    uniq = []
    for p in paths:
        s = str(p)
        if s not in seen:
            uniq.append(p)
            seen.add(s)

    return uniq

def read_pdf_text(pdf_path: Path, max_pages: int = 80) -> str:
    try:
        with open(pdf_path, "rb") as f:
            head = f.read(5)
        if head != b"%PDF-":
            return ""
    except Exception:
        return ""

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""

    chunks = []
    try:
        for i in range(min(max_pages, doc.page_count)):
            t = doc.load_page(i).get_text("text") or ""
            if t:
                chunks.append(t)
    finally:
        doc.close()

    text = "\n".join(chunks)
    text = text.replace("\u2212", "-")  # normalize minus
    text = re.sub(r"[ \t]+", " ", text)
    return text

def find_anchor_windows(text: str, window: int = 900):
    t = text.lower()
    hits = []
    for a in ANCHORS:
        a_l = a.lower()
        start = 0
        while True:
            idx = t.find(a_l, start)
            if idx == -1:
                break
            s = max(0, idx - window)
            e = min(len(text), idx + window)
            hits.append(text[s:e])
            start = idx + len(a_l)
    return hits

def extract_from_block(block: str):
    solvents_found = []
    for s in SOLVENTS:
        if re.search(rf"\b{re.escape(s)}\b", block, re.IGNORECASE):
            solvents_found.append(s)

    prec_found = []
    for p in PRECURSORS:
        if re.search(rf"\b{re.escape(p)}\b", block, re.IGNORECASE):
            prec_found.append(p)

    mols = set()
    mols.update([m.group(0) for m in RE_M.finditer(block)])
    mols.update([m.group(0) for m in RE_MOL_L.finditer(block)])
    mols.update([m.group(0) for m in RE_MM.finditer(block)])

    ratios = set()
    ratios.update(RE_NUM_RATIO.findall(block))
    if RE_VV.search(block):
        ratios.add("v/v")
    if RE_WW.search(block):
        ratios.add("w/w")

    temps = set([m.group(0) for m in RE_TEMP.finditer(block)])
    times = set([m.group(0) for m in RE_TIME.finditer(block)])
    filt = set([m.group(0) for m in RE_FILTER.finditer(block)])

    return {
        "precursors": ";".join(sorted(set(prec_found))),
        "solvents": ";".join(sorted(set(solvents_found))),
        "molarity": ";".join(sorted(mols)),
        "ratios": ";".join(sorted(ratios)),
        "temps_C": ";".join(sorted(temps)),
        "times": ";".join(sorted(times)),
        "filter_um": ";".join(sorted(filt)),
    }

def main():
    pdfs = shortlist_pdfs()
    if not pdfs:
        print("No shortlisted PDFs found from step3b_pdf_links.csv.")
        pd.DataFrame([]).to_csv(OUTFILE, index=False)
        print(f"Saved: {OUTFILE}")
        return

    rows = []
    kept = 0
    skipped = 0

    for pdf in pdfs:
        text = read_pdf_text(pdf, max_pages=80)
        if not text:
            skipped += 1
            continue

        blocks = find_anchor_windows(text, window=900)
        if not blocks:
            blocks = [text[:4000]]  # fallback

        scored = []
        for b in blocks[:40]:
            score = 0
            if RE_M.search(b) or RE_MOL_L.search(b) or RE_MM.search(b):
                score += 10
            if RE_NUM_RATIO.search(b) or RE_VV.search(b) or RE_WW.search(b):
                score += 10
            if any(re.search(rf"\b{re.escape(p)}\b", b, re.IGNORECASE) for p in PRECURSORS):
                score += 6
            if any(re.search(rf"\b{re.escape(s)}\b", b, re.IGNORECASE) for s in SOLVENTS):
                score += 5
            if "dissolv" in b.lower() or "prepared" in b.lower():
                score += 4
            scored.append((score, b))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_blocks = [b for _, b in scored[:5]]
        evidence = " || ".join([re.sub(r"\s+", " ", b).strip()[:1200] for b in best_blocks])

        fields = extract_from_block(evidence)

        rows.append({
            "pdf_file": str(pdf),
            "best_recipe_block": evidence,
            **fields
        })
        kept += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUTFILE, index=False)

    print(f"Shortlisted PDFs: {len(pdfs)} | kept: {kept} | skipped: {skipped}")
    if not df.empty:
        print(df[["pdf_file", "precursors", "solvents", "molarity", "ratios"]].to_string(index=False))
    print(f"\nSaved: {OUTFILE}")

if __name__ == "__main__":
    main()
