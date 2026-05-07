import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

IN_CAND = DATA_DIR / "01_search" / "step2_candidates.csv"
OUT = DATA_DIR / "02_downloads" / "step3a_candidates_with_pdf_url.csv"

def main():
    if not IN_CAND.exists():
        raise FileNotFoundError(f"Missing: {IN_CAND}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(IN_CAND)

    # Ensure these exist (downloader uses them)
    if "pdf_url" not in candidates.columns:
        candidates["pdf_url"] = ""
    if "landing_url" not in candidates.columns:
        candidates["landing_url"] = ""

    candidates.to_csv(OUT, index=False)
    print("Saved: step3a_candidates_with_pdf_url.csv")

if __name__ == "__main__":
    main()
