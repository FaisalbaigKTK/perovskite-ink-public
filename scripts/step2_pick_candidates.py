import re
import math
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

IN_STEP1 = DATA_DIR / "01_search" / "step1_results.csv"
OUT_ALL = DATA_DIR / "01_search" / "step2_scored_all.csv"
OUT_CAND = DATA_DIR / "01_search" / "step2_candidates.csv"

# Preferred upstream ML files (pre-download)
IN_ML_GOLD = DATA_DIR / "06_ml" / "paper_predownload_probability.csv"      # gold_probability
IN_ML_CLUSTER = DATA_DIR / "06_ml" / "paper_cluster3_probability.csv"      # cluster3_probability


def norm_title(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_doi(s: str) -> str:
    s = "" if s is None else str(s).strip()
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    s = s.replace("doi:", "").strip()
    return s.lower()


def doi_key(doi: str) -> str:
    d = clean_doi(doi)
    return re.sub(r"[^a-z0-9\.\-]+", "", d)


def keyword_score(title: str) -> float:
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


def safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_ml_gold():
    if not IN_ML_GOLD.exists():
        return pd.DataFrame(columns=["doi_key", "gold_probability"])
    ml = pd.read_csv(IN_ML_GOLD)
    if "doi_key" not in ml.columns or "gold_probability" not in ml.columns:
        raise RuntimeError(f"{IN_ML_GOLD} must contain doi_key and gold_probability")
    ml["doi_key"] = ml["doi_key"].fillna("").astype(str)
    ml["gold_probability"] = ml["gold_probability"].apply(safe_float)
    ml = ml.sort_values("gold_probability", ascending=False).drop_duplicates("doi_key")
    return ml[["doi_key", "gold_probability"]]


def load_ml_cluster():
    if not IN_ML_CLUSTER.exists():
        return pd.DataFrame(columns=["doi_key", "cluster3_probability"])
    ml = pd.read_csv(IN_ML_CLUSTER)
    if "doi_key" not in ml.columns or "cluster3_probability" not in ml.columns:
        raise RuntimeError(f"{IN_ML_CLUSTER} must contain doi_key and cluster3_probability")
    ml["doi_key"] = ml["doi_key"].fillna("").astype(str)
    ml["cluster3_probability"] = ml["cluster3_probability"].apply(safe_float)
    ml = ml.sort_values("cluster3_probability", ascending=False).drop_duplicates("doi_key")
    return ml[["doi_key", "cluster3_probability"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=150)
    ap.add_argument("--alpha-gold", type=float, default=12.0)
    ap.add_argument("--alpha-cluster", type=float, default=8.0)
    ap.add_argument("--beta-keywords", type=float, default=2.0)
    ap.add_argument("--min-year", type=int, default=0)
    args = ap.parse_args()

    if not IN_STEP1.exists():
        raise FileNotFoundError(f"Missing input: {IN_STEP1}")

    df = pd.read_csv(IN_STEP1)

    for col in ["title", "year", "doi", "openalex_id", "cited_by", "is_oa", "venue", "pdf_url", "landing_url"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["year"] = df["year"].apply(safe_int)
    df["doi"] = df["doi"].fillna("").astype(str).apply(clean_doi)
    df["doi_key"] = df["doi"].apply(doi_key)
    df["cited_by"] = df["cited_by"].apply(safe_int)
    df["is_oa"] = df["is_oa"].fillna("UNKNOWN").astype(str)
    df["venue"] = df["venue"].fillna("UNKNOWN").astype(str)
    df["title_norm"] = df["title"].apply(norm_title)

    if args.min_year and args.min_year > 0:
        df = df[df["year"] >= args.min_year].copy()

    # Baseline scores
    df["citation_score"] = df["cited_by"].apply(lambda x: math.log1p(max(x, 0))) * 5.0
    df["recency_score"] = df["year"].apply(lambda y: 0.0 if y <= 0 else max(0.0, (y - 2015) / 10.0))
    df["kw_score_raw"] = df["title"].apply(keyword_score)
    df["kw_score"] = df["kw_score_raw"] * float(args.beta_keywords)

    # Merge ML signals
    df["gold_probability"] = 0.0
    df["cluster3_probability"] = 0.0
    df["ml_match_gold"] = ""
    df["ml_match_cluster"] = ""

    ml_gold = load_ml_gold()
    ml_cluster = load_ml_cluster()

    ml_sources = []
    if len(ml_gold) > 0:
        df = df.merge(ml_gold, on="doi_key", how="left", suffixes=("", "_mlg"))
        if "gold_probability_mlg" in df.columns:
            m = df["gold_probability_mlg"].notna()
            df.loc[m, "gold_probability"] = df.loc[m, "gold_probability_mlg"].astype(float)
            df.loc[m, "ml_match_gold"] = "doi_key"
            df = df.drop(columns=["gold_probability_mlg"], errors="ignore")
        ml_sources.append("gold")

    if len(ml_cluster) > 0:
        df = df.merge(ml_cluster, on="doi_key", how="left", suffixes=("", "_mlc"))
        if "cluster3_probability_mlc" in df.columns:
            m = df["cluster3_probability_mlc"].notna()
            df.loc[m, "cluster3_probability"] = df.loc[m, "cluster3_probability_mlc"].astype(float)
            df.loc[m, "ml_match_cluster"] = "doi_key"
            df = df.drop(columns=["cluster3_probability_mlc"], errors="ignore")
        ml_sources.append("cluster3")

    # ML scores
    df["ml_score_gold"] = df["gold_probability"].astype(float) * float(args.alpha_gold)
    df["ml_score_cluster3"] = df["cluster3_probability"].astype(float) * float(args.alpha_cluster)
    df["ml_score_total"] = df["ml_score_gold"] + df["ml_score_cluster3"]

    # Final score
    df["final_score"] = df["citation_score"] + df["recency_score"] + df["kw_score"] + df["ml_score_total"]

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ALL, index=False)

    cand = df.head(int(args.top_n)).copy()
    cand.to_csv(OUT_CAND, index=False)

    # Print summary
    print(f"Loaded step1 rows: {len(df)}")
    print(f"ML sources present: {', '.join(ml_sources) if ml_sources else 'none'}")
    print("Top-N ML matches (gold):")
    print(cand["ml_match_gold"].value_counts(dropna=False).to_string())
    print("Top-N ML matches (cluster3):")
    print(cand["ml_match_cluster"].value_counts(dropna=False).to_string())
    print(f"Top-N mean gold_probability: {cand['gold_probability'].mean():.3f}")
    print(f"Top-N mean cluster3_probability: {cand['cluster3_probability'].mean():.3f}")
    print(f"Saved all scored: {OUT_ALL}")
    print(f"Saved candidates: {OUT_CAND}")

    print("\nTop 10 candidates preview:")
    show_cols = [
        "title", "year", "doi", "cited_by",
        "gold_probability", "cluster3_probability",
        "ml_score_total", "final_score"
    ]
    show_cols = [c for c in show_cols if c in cand.columns]
    print(cand[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
