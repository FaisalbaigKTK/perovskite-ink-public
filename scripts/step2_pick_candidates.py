"""
step2_pick_candidates.py
========================
Pipeline stage: 2 of 9
Input:  data/01_search/step1_results.csv           (959 OpenAlex records)
        data/06_ml/paper_predownload_probability.csv  (optional: gold_probability per doi_key)
        data/06_ml/paper_cluster3_probability.csv     (optional: cluster3_probability per doi_key)
Output: data/01_search/step2_scored_all.csv        (all 959 records with scores)
        data/01_search/step2_candidates.csv         (top-N candidates, default 150)

Purpose
-------
Scores every OpenAlex record returned by step1 using a transparent multi-term
ranking function and retains the top-N papers as candidates for PDF download.

Scoring formula (final_score):
    final_score = citation_score
                + recency_score
                + kw_score
                + ml_score_gold        (0 if ML files absent)
                + ml_score_cluster3    (0 if ML files absent)

Term definitions:
    citation_score   = log(1 + cited_by) * 5.0
    recency_score    = max(0, (year - 2015) / 10)   [linear, not exponential]
    kw_score         = keyword_score(title) * beta_keywords
    ml_score_gold    = gold_probability * alpha_gold
    ml_score_cluster3 = cluster3_probability * alpha_cluster

The two ML probability columns (gold_probability, cluster3_probability) were
produced by classifiers that are part of a private production system. Their
output values for the 150 selected candidates are released inside
step2_candidates.csv so the exact ranking used in the paper is inspectable,
even though the trained model weights are not public. The publicly reproducible
version of this script uses only the keyword/citation/recency heuristic when
the ML probability files are absent.

Usage
-----
    python step2_pick_candidates.py [--top-n 150] [--alpha-gold 12.0]
                                    [--alpha-cluster 8.0] [--beta-keywords 2.0]
                                    [--min-year 0]

Reviewer note (Digital Discovery R2.1)
--------------------------------------
An earlier manuscript draft incorrectly described this step as including
'two binary classifiers' as part of the public pipeline. The public pipeline
uses only the keyword/citation/recency heuristic; the classifier probabilities
are released as data columns but the models are private. See Methods §2.2.
"""

import re
import math
import argparse
import pandas as pd
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

IN_STEP1    = DATA_DIR / "01_search" / "step1_results.csv"
OUT_ALL     = DATA_DIR / "01_search" / "step2_scored_all.csv"
OUT_CAND    = DATA_DIR / "01_search" / "step2_candidates.csv"

# Optional ML probability files (production system; not required for public reproduction)
IN_ML_GOLD    = DATA_DIR / "06_ml" / "paper_predownload_probability.csv"
IN_ML_CLUSTER = DATA_DIR / "06_ml" / "paper_cluster3_probability.csv"


# ── Text normalisation helpers ────────────────────────────────────────────────

def norm_title(s: str) -> str:
    """Normalise a paper title to lowercase ASCII for keyword matching.

    Args:
        s: Raw title string (may contain HTML tags or special characters).

    Returns:
        Lowercase, stripped string with HTML and punctuation removed.
    """
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"<[^>]+>", " ", s)        # remove HTML tags
    s = re.sub(r"[^a-z0-9]+", " ", s)     # keep alphanumeric only
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_doi(s: str) -> str:
    """Strip URL prefixes and normalise a DOI string to bare form.

    Args:
        s: DOI string, possibly prefixed with 'https://doi.org/' or 'doi:'.

    Returns:
        Lowercase bare DOI, e.g. '10.1038/s41586-023-05932-6'.
    """
    s = "" if s is None else str(s).strip()
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    s = s.replace("doi:", "").strip()
    return s.lower()


def doi_key(doi: str) -> str:
    """Convert a bare DOI to a filesystem-safe key (alphanumeric + dot + dash).

    Args:
        doi: Bare DOI string from clean_doi().

    Returns:
        Simplified key string used for CSV joining.
    """
    d = clean_doi(doi)
    return re.sub(r"[^a-z0-9\.\-]+", "", d)


# ── Scoring functions ─────────────────────────────────────────────────────────

def keyword_score(title: str) -> float:
    """Score a paper title based on presence of domain-relevant keywords.

    Each keyword has a weight reflecting its specificity to perovskite ink
    formulation literature (e.g. 'inkjet' = 3.0, 'solution' = 1.0).

    Args:
        title: Raw paper title string.

    Returns:
        Non-negative float keyword score.
    """
    t = (title or "").lower()
    score = 0.0
    for kw, w in [
        ("inkjet", 3.0), ("ink-jet", 3.0),
        ("printable", 2.0), ("printed", 1.5),
        ("precursor", 2.0), ("solution", 1.0),
        ("ink", 2.0),
        ("slot-die", 2.5), ("slot die", 2.5),
        ("blade", 1.5), ("coating", 1.5),
        ("dmso", 2.0), ("dmf", 1.5), ("gbl", 1.0),
        ("acetonitrile", 2.0),
        ("macl", 2.0), ("csi", 2.0),
        ("scalable", 1.0), ("large-area", 1.0), ("large area", 1.0),
        ("molar", 1.0), ("1 m", 1.0), ("1.0 m", 1.0),
    ]:
        if kw in t:
            score += w
    return score


def safe_int(x, default: int = 0) -> int:
    """Safely cast a value to int, returning default on failure or NaN.

    Args:
        x: Value to cast.
        default: Return value on failure.

    Returns:
        Integer value or default.
    """
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default: float = 0.0) -> float:
    """Safely cast a value to float, returning default on failure or NaN.

    Args:
        x: Value to cast.
        default: Return value on failure.

    Returns:
        Float value or default.
    """
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# ── ML file loaders ───────────────────────────────────────────────────────────

def load_ml_gold() -> pd.DataFrame:
    """Load the gold-classifier probability file if it exists.

    The file must contain columns 'doi_key' and 'gold_probability'.
    If the file is absent, returns an empty DataFrame with those columns,
    and the ml_score_gold term is zero for all records.

    Returns:
        DataFrame with columns ['doi_key', 'gold_probability'].
    """
    if not IN_ML_GOLD.exists():
        return pd.DataFrame(columns=["doi_key", "gold_probability"])
    ml = pd.read_csv(IN_ML_GOLD)
    if "doi_key" not in ml.columns or "gold_probability" not in ml.columns:
        raise RuntimeError(f"{IN_ML_GOLD} must contain doi_key and gold_probability")
    ml["doi_key"] = ml["doi_key"].fillna("").astype(str)
    ml["gold_probability"] = ml["gold_probability"].apply(safe_float)
    # Keep highest probability per doi_key (dedup)
    ml = ml.sort_values("gold_probability", ascending=False).drop_duplicates("doi_key")
    return ml[["doi_key", "gold_probability"]]


def load_ml_cluster() -> pd.DataFrame:
    """Load the cluster3-classifier probability file if it exists.

    The file must contain columns 'doi_key' and 'cluster3_probability'.
    If absent, returns an empty DataFrame and ml_score_cluster3 is zero.

    Returns:
        DataFrame with columns ['doi_key', 'cluster3_probability'].
    """
    if not IN_ML_CLUSTER.exists():
        return pd.DataFrame(columns=["doi_key", "cluster3_probability"])
    ml = pd.read_csv(IN_ML_CLUSTER)
    if "doi_key" not in ml.columns or "cluster3_probability" not in ml.columns:
        raise RuntimeError(f"{IN_ML_CLUSTER} must contain doi_key and cluster3_probability")
    ml["doi_key"] = ml["doi_key"].fillna("").astype(str)
    ml["cluster3_probability"] = ml["cluster3_probability"].apply(safe_float)
    ml = ml.sort_values("cluster3_probability", ascending=False).drop_duplicates("doi_key")
    return ml[["doi_key", "cluster3_probability"]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Score all step1 records and write top-N candidates to CSV.

    Reads step1_results.csv, computes five score components for each record,
    merges optional ML probability columns, sorts by final_score descending,
    and writes the top-N records to step2_candidates.csv.
    """
    ap = argparse.ArgumentParser(
        description="Score OpenAlex records and select top-N candidates for PDF download."
    )
    ap.add_argument("--top-n",          type=int,   default=150,  help="Number of candidates to retain")
    ap.add_argument("--alpha-gold",     type=float, default=12.0, help="Weight for gold_probability term")
    ap.add_argument("--alpha-cluster",  type=float, default=8.0,  help="Weight for cluster3_probability term")
    ap.add_argument("--beta-keywords",  type=float, default=2.0,  help="Weight for keyword_score term")
    ap.add_argument("--min-year",       type=int,   default=0,    help="Exclude records before this year (0 = no filter)")
    args = ap.parse_args()

    if not IN_STEP1.exists():
        raise FileNotFoundError(f"Missing input: {IN_STEP1}")

    df = pd.read_csv(IN_STEP1)

    # Ensure required columns exist (fill with empty string if absent)
    for col in ["title", "year", "doi", "openalex_id", "cited_by", "is_oa", "venue", "pdf_url", "landing_url"]:
        if col not in df.columns:
            df[col] = ""

    # Cast and clean column types
    df["title"]     = df["title"].fillna("").astype(str)
    df["year"]      = df["year"].apply(safe_int)
    df["doi"]       = df["doi"].fillna("").astype(str).apply(clean_doi)
    df["doi_key"]   = df["doi"].apply(doi_key)
    df["cited_by"]  = df["cited_by"].apply(safe_int)
    df["is_oa"]     = df["is_oa"].fillna("UNKNOWN").astype(str)
    df["venue"]     = df["venue"].fillna("UNKNOWN").astype(str)
    df["title_norm"] = df["title"].apply(norm_title)

    # Optional year filter
    if args.min_year and args.min_year > 0:
        df = df[df["year"] >= args.min_year].copy()

    # ── Compute baseline score components ─────────────────────────────────────
    # citation_score: log-scaled citation count emphasises high-impact papers
    df["citation_score"] = df["cited_by"].apply(lambda x: math.log1p(max(x, 0))) * 5.0

    # recency_score: linear term — more recent papers score higher
    # Note: manuscript used to say 'exponential decay from 2024'; this is the
    # correct linear form actually implemented (year − 2015) / 10.
    df["recency_score"] = df["year"].apply(
        lambda y: 0.0 if y <= 0 else max(0.0, (y - 2015) / 10.0)
    )

    # kw_score: title keyword match weighted by beta_keywords
    df["kw_score_raw"] = df["title"].apply(keyword_score)
    df["kw_score"]     = df["kw_score_raw"] * float(args.beta_keywords)

    # ── Merge ML signals (optional) ───────────────────────────────────────────
    # Initialise ML columns to zero; overwrite if probability files are present
    df["gold_probability"]     = 0.0
    df["cluster3_probability"] = 0.0
    df["ml_match_gold"]        = ""
    df["ml_match_cluster"]     = ""

    ml_sources = []

    ml_gold = load_ml_gold()
    if len(ml_gold) > 0:
        # Left-join on doi_key; suffix _mlg avoids collision with existing column
        df = df.merge(ml_gold, on="doi_key", how="left", suffixes=("", "_mlg"))
        if "gold_probability_mlg" in df.columns:
            mask = df["gold_probability_mlg"].notna()
            df.loc[mask, "gold_probability"] = df.loc[mask, "gold_probability_mlg"].astype(float)
            df.loc[mask, "ml_match_gold"]    = "doi_key"
            df = df.drop(columns=["gold_probability_mlg"], errors="ignore")
        ml_sources.append("gold")

    ml_cluster = load_ml_cluster()
    if len(ml_cluster) > 0:
        df = df.merge(ml_cluster, on="doi_key", how="left", suffixes=("", "_mlc"))
        if "cluster3_probability_mlc" in df.columns:
            mask = df["cluster3_probability_mlc"].notna()
            df.loc[mask, "cluster3_probability"] = df.loc[mask, "cluster3_probability_mlc"].astype(float)
            df.loc[mask, "ml_match_cluster"]     = "doi_key"
            df = df.drop(columns=["cluster3_probability_mlc"], errors="ignore")
        ml_sources.append("cluster3")

    # ── Compute weighted ML score components ──────────────────────────────────
    df["ml_score_gold"]     = df["gold_probability"].astype(float)    * float(args.alpha_gold)
    df["ml_score_cluster3"] = df["cluster3_probability"].astype(float) * float(args.alpha_cluster)
    df["ml_score_total"]    = df["ml_score_gold"] + df["ml_score_cluster3"]

    # ── Final combined score and ranking ──────────────────────────────────────
    df["final_score"] = (
        df["citation_score"]
        + df["recency_score"]
        + df["kw_score"]
        + df["ml_score_total"]
    )

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    # ── Write outputs ─────────────────────────────────────────────────────────
    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ALL, index=False)

    # Top-N candidates for PDF download
    cand = df.head(int(args.top_n)).copy()
    cand.to_csv(OUT_CAND, index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"Loaded step1 rows      : {len(df)}")
    print(f"ML sources present     : {', '.join(ml_sources) if ml_sources else 'none (heuristic only)'}")
    print(f"Candidates selected    : {len(cand)} (top-{args.top_n})")
    print(f"Score range            : {df['final_score'].min():.2f} – {df['final_score'].max():.2f}")
    print(f"Saved all scored       : {OUT_ALL}")
    print(f"Saved candidates       : {OUT_CAND}")

    show_cols = [c for c in ["title", "year", "doi", "cited_by",
                              "gold_probability", "cluster3_probability",
                              "ml_score_total", "final_score"] if c in cand.columns]
    print("\nTop 10 candidates preview:")
    print(cand[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
