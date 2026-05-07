"""
Robust OpenAlex search that:
- runs multiple queries
- uses paging
- retries + exponential backoff (+ jitter)
- saves: data/01_search/step1_results.csv
"""

import time
import random
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import sys

# Make Windows console printing robust to unicode in titles
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "01_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUT_DIR / "step1_results.csv"
OPENALEX = "https://api.openalex.org/works"

# You can remove mailto if you want; it just helps "polite client" identification.
HEADERS = {
    "User-Agent": "perovskite-ink-scraper (mailto:your_email@example.com)"
}

# Add / remove queries here to scale
QUERIES = [
    'inkjet perovskite "ink preparation" DMSO',
    'perovskite "precursor solution" DMSO',
    'perovskite ink formulation MACl',
    'printed perovskite solvent engineering',
    'inkjet perovskite GVL DMSO',
]

PER_PAGE = 25       # safer for big runs
MAX_PAGES = 20      # per query
TIMEOUT_S = 120
TRIES = 7

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def get_with_retries(url: str, params: Dict[str, Any]) -> requests.Response:
    last_err = None
    for i in range(TRIES):
        try:
            r = SESSION.get(url, params=params, timeout=TIMEOUT_S)
            r.raise_for_status()
            return r
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            last_err = e
            wait = (2 ** i) + random.uniform(0, 1.0)
            print(f"[retry {i+1}/{TRIES}] {type(e).__name__}: {e} -> sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"OpenAlex request failed after retries. Last error: {last_err}")

def extract_row(w: Dict[str, Any]) -> Dict[str, Any]:
    doi_url = w.get("doi") or ""
    doi = doi_url.replace("https://doi.org/", "").strip()

    oa = w.get("open_access") or {}
    is_oa = bool(oa.get("is_oa", False))

    best_oa = w.get("best_oa_location") or {}
    pdf_url = (best_oa.get("pdf_url") or "").strip()
    landing_url = (best_oa.get("landing_page_url") or "").strip()

    primary_loc = w.get("primary_location") or {}
    source = (primary_loc.get("source") or {})
    venue = (source.get("display_name") or "").strip()

    return {
        "title": (w.get("title") or "").strip(),
        "year": w.get("publication_year", ""),
        "doi": doi,
        "openalex_id": (w.get("id") or "").strip(),
        "cited_by": int(w.get("cited_by_count") or 0),

        "is_oa": is_oa,
        "pdf_url": pdf_url,
        "landing_url": landing_url,
        "venue": venue,
    }

def fetch_query(query: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        params = {
            "search": query,
            "per-page": PER_PAGE,
            "page": page,

            # If you want max volume (including paywalled), comment this out:
            "filter": "is_oa:true",
        }

        print(f"\nQuery: {query}\n  Fetch page {page}/{MAX_PAGES} (per-page={PER_PAGE}) ...")
        r = get_with_retries(OPENALEX, params)
        results = r.json().get("results", [])
        if not results:
            print("  No more results.")
            break

        for w in results:
            rows.append(extract_row(w))

        time.sleep(1.0)  # slow down to avoid 10054 resets
    return rows

def main():
    all_rows: List[Dict[str, Any]] = []
    for q in QUERIES:
        all_rows.extend(fetch_query(q))

    if not all_rows:
        print("No results collected.")
        return

    df = pd.DataFrame(all_rows)
    if "openalex_id" in df.columns:
        df = df.drop_duplicates(subset=["openalex_id"])
    df = df.sort_values("cited_by", ascending=False)
    df.to_csv(OUTFILE, index=False)

    print("\n===================================================")
    print(f"Saved: {OUTFILE}")
    print(f"Rows: {len(df)}")
    preview = df.head(10).to_string(index=False)
    print(preview.encode("utf-8", errors="replace").decode("utf-8"))
    print("===================================================\n")

if __name__ == "__main__":
    main()
