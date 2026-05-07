import re
import time
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urljoin
import random

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

INFILE = DATA_DIR / "02_downloads" / "step3a_candidates_with_pdf_url.csv"
OUTFILE = DATA_DIR / "02_downloads" / "step3b_pdf_links.csv"
PDF_DIR = ROOT / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PDFBot/1.0",
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def clean_filename(s: str) -> str:
    s = re.sub(r"[^\w\-\. ]+", "", s)
    s = s.strip().replace(" ", "_")
    return s[:140] if len(s) > 140 else s

def ensure_pdf_ext(name: str) -> str:
    return name if name.lower().endswith(".pdf") else name + ".pdf"

def looks_like_pdf_bytes(b: bytes) -> bool:
    return b[:4] == b"%PDF"

def extract_pdf_links_from_html(html: str, base_url: str):
    links = set()
    for m in re.findall(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', html, flags=re.I):
        links.add(urljoin(base_url, m))
    for m in re.findall(r'(https?://[^\s"\']+\.pdf[^\s"\']*)', html, flags=re.I):
        links.add(m)
    return list(links)

def extract_figshare_article_id(url: str):
    m = re.search(r"/(\d+)(?:\b|/|$)", url)
    return m.group(1) if m else None

def figshare_api_get_file_urls(article_id: str):
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    r = SESSION.get(api_url, timeout=60)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", []) or []
    urls = []
    for f in files:
        u = f.get("download_url") or f.get("preview_url") or ""
        if u:
            urls.append(u)
    return urls

def try_download_as_pdf(url: str, out_path: Path):
    try:
        r = SESSION.get(url, timeout=90, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        data = r.content

        if ("pdf" in ct) or r.url.lower().endswith(".pdf") or looks_like_pdf_bytes(data):
            out_path.write_bytes(data)
            return True, "downloaded_pdf", r.url

        return False, "not_pdf_response", r.url
    except Exception as e:
        return False, f"download_failed:{type(e).__name__}", url

def smart_download(pdf_or_page_url: str, out_path: Path):
    ok, status, final_url = try_download_as_pdf(pdf_or_page_url, out_path)
    if ok:
        return status, final_url

    try:
        r = SESSION.get(pdf_or_page_url, timeout=90, allow_redirects=True)
        html = r.text
        pdf_links = extract_pdf_links_from_html(html, r.url)

        for link in pdf_links[:8]:
            ok2, status2, final2 = try_download_as_pdf(link, out_path)
            if ok2:
                return "downloaded_from_html_link", final2

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
                    pass

        return "no_accessible_pdf_found", r.url

    except Exception:
        return "html_parse_failed", pdf_or_page_url

def main():
    df = pd.read_csv(INFILE)

    rows = []
    for _, row in df.iterrows():
        doi = str(row.get("doi", "")).strip()
        title = str(row.get("title", "")).strip()
        year = str(row.get("year", "")).strip()

        pdf_url = str(row.get("pdf_url", "")).strip()
        landing_url = str(row.get("landing_url", "")).strip()

        # If pdf_url is missing, we can try landing_url (sometimes it contains pdf links)
        url_to_try = pdf_url or landing_url

        if not url_to_try:
            rows.append({**row.to_dict(), "pdf_file": "", "status": "missing_pdf_url_in_input", "final_url": ""})
            continue

        base = clean_filename(f"{year}_{doi}_{title}")
        fname = ensure_pdf_ext(base)
        out_path = PDF_DIR / fname

        status, final_url = smart_download(url_to_try, out_path)

        pdf_file = str(out_path) if out_path.exists() and out_path.stat().st_size > 10_000 else ""
        if status.startswith("downloaded") and not pdf_file:
            status = "downloaded_but_file_missing"

        rows.append({**row.to_dict(), "pdf_file": pdf_file, "status": status, "final_url": final_url})

        time.sleep(0.6 + random.uniform(0, 0.6))  # reduce server resets

    out = pd.DataFrame(rows)
    out.to_csv(OUTFILE, index=False)

    show_cols = [c for c in ["status", "year", "doi", "pdf_file", "final_url"] if c in out.columns]
    print(out[show_cols].to_string(index=False))
    print(f"\nSaved: {OUTFILE}")
    print("PDF folder:", PDF_DIR.resolve())

if __name__ == "__main__":
    main()
