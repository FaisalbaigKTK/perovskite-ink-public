"""
step3b_download_pdfs_smart.py
==============================
Pipeline stage: 3b of 9
Input:  data/02_downloads/step3a_candidates_with_pdf_url.csv
        (150 candidates with resolved pdf_url and landing_url columns)
Output: data/02_downloads/step3b_pdf_links.csv
        (same rows + pdf_file path, download status, and final_url)
        pdfs/   (directory of downloaded PDF files)

Purpose
-------
Attempts to download the open-access PDF for each candidate paper using a
three-tier cascade strategy:

  Tier 1 — Direct download:
      Try the pdf_url directly. Accept if Content-Type contains 'pdf' OR
      the response bytes start with '%PDF'.

  Tier 2 — HTML scraping:
      If Tier 1 fails, fetch the landing page and parse all <a href="*.pdf">
      and bare PDF URLs from the HTML. Try each extracted link via Tier 1.

  Tier 3 — Figshare API:
      If the resolved URL is on figshare.com and Tiers 1–2 failed, query the
      Figshare public API (api.figshare.com/v2/articles/{id}) and attempt
      download of each file URL returned.

Status codes written to step3b_pdf_links.csv:
  downloaded_pdf              — Tier 1 success (direct PDF bytes)
  downloaded_from_html_link   — Tier 2 success (link found in HTML)
  downloaded_figshare_api     — Tier 3 success (Figshare API)
  downloaded_but_file_missing — Download claimed success but file <10 KB
  no_accessible_pdf_found     — All tiers exhausted, no PDF obtained
  html_parse_failed           — Landing page request raised an exception
  missing_pdf_url_in_input    — Neither pdf_url nor landing_url in input row

All HTTP requests use a persistent session with a descriptive User-Agent string
and a 0.6–1.2 second random delay between papers to reduce server load.

Usage
-----
    python step3b_download_pdfs_smart.py
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urljoin
import random

# ── Path configuration ────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

INFILE  = DATA_DIR / "02_downloads" / "step3a_candidates_with_pdf_url.csv"
OUTFILE = DATA_DIR / "02_downloads" / "step3b_pdf_links.csv"
PDF_DIR = ROOT / "pdfs"
PDF_DIR.mkdir(exist_ok=True)   # create pdfs/ directory if it does not exist

# ── HTTP session setup ────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PDFBot/1.0",
    "Accept":     "application/pdf,text/html;q=0.9,*/*;q=0.8",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ── Filename helpers ──────────────────────────────────────────────────────────

def clean_filename(s: str) -> str:
    """Sanitise a string for use as a filesystem filename.

    Removes characters that are illegal on Windows/Linux/macOS, collapses
    spaces to underscores, and truncates to 140 characters.

    Args:
        s: Raw filename candidate string (e.g. '{year}_{doi}_{title}').

    Returns:
        Safe filename string without extension.
    """
    s = re.sub(r"[^\w\-\. ]+", "", s)
    s = s.strip().replace(" ", "_")
    return s[:140] if len(s) > 140 else s


def ensure_pdf_ext(name: str) -> str:
    """Append '.pdf' to a filename if not already present.

    Args:
        name: Filename string.

    Returns:
        Filename guaranteed to end with '.pdf'.
    """
    return name if name.lower().endswith(".pdf") else name + ".pdf"


def looks_like_pdf_bytes(b: bytes) -> bool:
    """Check whether a byte string begins with the PDF magic number '%PDF'.

    Args:
        b: First few bytes of a downloaded response.

    Returns:
        True if the bytes indicate a PDF file.
    """
    return b[:4] == b"%PDF"


# ── HTML link extraction ──────────────────────────────────────────────────────

def extract_pdf_links_from_html(html: str, base_url: str) -> list:
    """Parse an HTML page for PDF hyperlinks.

    Finds both href attributes ending in '.pdf' and bare PDF URLs in the
    page source, resolving relative URLs against base_url.

    Args:
        html: Full HTML source of the landing page.
        base_url: Absolute URL used to resolve relative links.

    Returns:
        Deduplicated list of absolute PDF URLs found in the page.
    """
    links = set()
    # href attributes ending with .pdf (with optional query strings)
    for m in re.findall(r'href=["\']([\S]+\.pdf[\S]*)["\']', html, flags=re.I):
        links.add(urljoin(base_url, m))
    # Bare PDF URLs in page source
    for m in re.findall(r'(https?://[^\s"\']+\.pdf[^\s"\']*)', html, flags=re.I):
        links.add(m)
    return list(links)


# ── Figshare API helpers ──────────────────────────────────────────────────────

def extract_figshare_article_id(url: str):
    """Extract the numeric article ID from a Figshare URL.

    Args:
        url: Figshare article URL, e.g. 'https://figshare.com/articles/12345678'.

    Returns:
        Article ID string, or None if not found.
    """
    m = re.search(r"/(\d+)(?:\b|/|$)", url)
    return m.group(1) if m else None


def figshare_api_get_file_urls(article_id: str) -> list:
    """Query the Figshare public API and return download URLs for all files.

    Args:
        article_id: Numeric Figshare article ID string.

    Returns:
        List of download URL strings (may be empty if no files found).

    Raises:
        requests.HTTPError: If the API request fails with a non-2xx status.
    """
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    r = SESSION.get(api_url, timeout=60)
    r.raise_for_status()
    data  = r.json()
    files = data.get("files", []) or []
    urls  = []
    for f in files:
        u = f.get("download_url") or f.get("preview_url") or ""
        if u:
            urls.append(u)
    return urls


# ── Download logic ────────────────────────────────────────────────────────────

def try_download_as_pdf(url: str, out_path: Path) -> tuple:
    """Attempt to download a URL as a PDF file (Tier 1).

    Checks Content-Type header and PDF magic bytes before writing to disk.

    Args:
        url:      URL to fetch.
        out_path: Destination Path for the downloaded file.

    Returns:
        Tuple (success: bool, status: str, final_url: str).
    """
    try:
        r  = SESSION.get(url, timeout=90, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        data = r.content

        # Accept if Content-Type says PDF, URL ends with .pdf, or bytes are PDF
        if ("pdf" in ct) or r.url.lower().endswith(".pdf") or looks_like_pdf_bytes(data):
            out_path.write_bytes(data)
            return True, "downloaded_pdf", r.url

        return False, "not_pdf_response", r.url
    except Exception as e:
        return False, f"download_failed:{type(e).__name__}", url


def smart_download(pdf_or_page_url: str, out_path: Path) -> tuple:
    """Three-tier PDF acquisition cascade.

    Tries Tier 1 (direct), Tier 2 (HTML scraping), and Tier 3 (Figshare API)
    in sequence, returning on the first success.

    Args:
        pdf_or_page_url: URL from the candidate's pdf_url or landing_url field.
        out_path:        Destination Path for the downloaded PDF.

    Returns:
        Tuple (status: str, final_url: str).
    """
    # Tier 1 — direct PDF download
    ok, status, final_url = try_download_as_pdf(pdf_or_page_url, out_path)
    if ok:
        return status, final_url

    # Tier 2 — scrape landing page for PDF links
    try:
        r        = SESSION.get(pdf_or_page_url, timeout=90, allow_redirects=True)
        html     = r.text
        pdf_links = extract_pdf_links_from_html(html, r.url)

        for link in pdf_links[:8]:   # try up to 8 links from the page
            ok2, status2, final2 = try_download_as_pdf(link, out_path)
            if ok2:
                return "downloaded_from_html_link", final2

        # Tier 3 — Figshare API (only for figshare.com landing pages)
        if "figshare.com" in r.url.lower():
            article_id = extract_figshare_article_id(r.url)
            if article_id:
                try:
                    file_urls = figshare_api_get_file_urls(article_id)
                    for fu in file_urls[:5]:
                        ok3, status3, final3 = try_download_as_pdf(fu, out_path)
                        if ok3:
                            return "downloaded_figshare_api", final3
                except Exception:
                    pass  # Figshare API failure is non-fatal; continue to return failure

        return "no_accessible_pdf_found", r.url

    except Exception:
        return "html_parse_failed", pdf_or_page_url


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Iterate over all candidate URLs and attempt PDF download for each.

    Reads step3a output, calls smart_download() for every row, and writes
    enriched results (with pdf_file path and download status) to step3b output.
    Files smaller than 10 KB after download are considered corrupt and flagged.
    A random delay of 0.6–1.2 seconds between requests reduces server load.
    """
    df = pd.read_csv(INFILE)

    rows = []
    for _, row in df.iterrows():
        doi        = str(row.get("doi",         "")).strip()
        title      = str(row.get("title",       "")).strip()
        year       = str(row.get("year",        "")).strip()
        pdf_url    = str(row.get("pdf_url",     "")).strip()
        landing_url = str(row.get("landing_url", "")).strip()

        # Prefer explicit PDF URL; fall back to landing page URL
        url_to_try = pdf_url or landing_url

        if not url_to_try:
            rows.append({
                **row.to_dict(),
                "pdf_file": "", "status": "missing_pdf_url_in_input", "final_url": ""
            })
            continue

        # Build a deterministic output filename: {year}_{doi}_{title}.pdf
        base     = clean_filename(f"{year}_{doi}_{title}")
        fname    = ensure_pdf_ext(base)
        out_path = PDF_DIR / fname

        status, final_url = smart_download(url_to_try, out_path)

        # Validate the downloaded file (must exist and be > 10 KB)
        pdf_file = str(out_path) if out_path.exists() and out_path.stat().st_size > 10_000 else ""
        if status.startswith("downloaded") and not pdf_file:
            status = "downloaded_but_file_missing"  # file written but suspiciously small

        rows.append({
            **row.to_dict(),
            "pdf_file": pdf_file, "status": status, "final_url": final_url
        })

        # Polite delay to avoid overwhelming open-access servers
        time.sleep(0.6 + random.uniform(0, 0.6))

    out = pd.DataFrame(rows)
    out.to_csv(OUTFILE, index=False)

    # Console summary
    show_cols = [c for c in ["status", "year", "doi", "pdf_file", "final_url"] if c in out.columns]
    print(out[show_cols].to_string(index=False))
    print(f"\nStatus counts:\n{out['status'].value_counts().to_string()}")
    print(f"\nSaved : {OUTFILE}")
    print(f"PDFs  : {PDF_DIR.resolve()}")


if __name__ == "__main__":
    main()
