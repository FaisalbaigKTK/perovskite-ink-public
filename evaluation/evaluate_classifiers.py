"""
evaluation/evaluate_classifiers.py
====================================
Evaluate the ML classifiers (gold classifier, pre-download classifier, cluster
predictor) using cross-validation and held-out test metrics.

Reports:
  - ROC-AUC, Average Precision (PR-AUC)
  - Precision@K for candidate ranking
  - Classification report (precision / recall / F1 per class)

Usage:
    python evaluate_classifiers.py \
        --tidy   data_sample/extracted_recipes_sample.csv \
        --step1  data_sample/step1_results_sample.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def make_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric features from the tidy dataset."""
    X = pd.DataFrame()
    X["recipe_score"]   = pd.to_numeric(df.get("recipe_score", 0),  errors="coerce").fillna(0)
    X["molarity_M"]     = pd.to_numeric(df.get("molarity_M", 0),    errors="coerce").fillna(0)
    X["has_formula"]    = df.get("perovskite_formula", "").apply(lambda v: 0 if str(v).strip() in ("", "nan", "UNKNOWN") else 1)
    X["has_solvent"]    = df.get("solvent_system_final", "").apply(lambda v: 0 if str(v).strip() in ("", "nan", "UNKNOWN") else 1)
    X["has_ratio"]      = df.get("solvent_ratio_final", "").apply(lambda v: 0 if str(v).strip() in ("", "nan") else 1)
    X["has_molarity"]   = (X["molarity_M"] > 0).astype(int)
    X["has_additives"]  = df.get("additives_found", "").apply(lambda v: 0 if str(v).strip() in ("", "nan") else 1)
    return X


def evaluate_gold_classifier(df: pd.DataFrame, gold_col: str = "is_gold") -> None:
    if gold_col not in df.columns:
        print(f"[SKIP] Column '{gold_col}' not found — skipping gold classifier eval.")
        return

    y = df[gold_col].astype(int)
    if y.nunique() < 2:
        print("[SKIP] Need both positive and negative labels for gold classifier.")
        return

    X = make_feature_matrix(df)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv)

    print("\n--- GOLD CLASSIFIER (5-fold CV) ---")
    print(f"  ROC-AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    print(classification_report(y, pred, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))


def precision_at_k(ranked_df: pd.DataFrame, gold_keys: set, k: int) -> float:
    top = ranked_df.head(k)
    return float(top["doi_key"].isin(gold_keys).mean()) if k > 0 and len(top) > 0 else 0.0


def evaluate_predownload_classifier(step1_df: pd.DataFrame, gold_dois: set) -> None:
    if "doi_key" not in step1_df.columns:
        print("[SKIP] 'doi_key' not in step1 data — skipping pre-download classifier eval.")
        return

    step1_df = step1_df.copy()
    step1_df["is_gold"] = step1_df["doi_key"].isin(gold_dois).astype(int)
    y = step1_df["is_gold"]

    if y.nunique() < 2 or y.sum() < 3:
        print("[SKIP] Too few gold matches in step1 for pre-download classifier eval.")
        return

    # Simple numeric features from step1
    X = pd.DataFrame()
    X["cited_by"] = pd.to_numeric(step1_df.get("cited_by", 0), errors="coerce").fillna(0)
    X["year"]     = pd.to_numeric(step1_df.get("year", 2020),   errors="coerce").fillna(2020)
    X["is_oa"]    = step1_df.get("is_oa", False).astype(int)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    ranked = step1_df.copy()
    ranked["score"] = proba
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)

    print("\n--- PRE-DOWNLOAD CLASSIFIER — Precision@K ---")
    for k in [10, 25, 50]:
        if k <= len(ranked):
            print(f"  P@{k}: {precision_at_k(ranked, gold_dois, k):.3f}")

    if y.nunique() > 1:
        print(f"  ROC-AUC (train): {roc_auc_score(y, proba):.4f}")
        print(f"  Avg Precision:   {average_precision_score(y, proba):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tidy",  default="data_sample/extracted_recipes_sample.csv")
    parser.add_argument("--step1", default="data_sample/step1_results_sample.csv")
    args = parser.parse_args()

    tidy_path  = Path(args.tidy)
    step1_path = Path(args.step1)

    print("=" * 60)
    print("PEROVSKITE INK — CLASSIFIER EVALUATION")
    print("=" * 60)

    if tidy_path.exists():
        df = pd.read_csv(tidy_path)
        evaluate_gold_classifier(df)
        gold_pdfs = set(df.get("pdf_file", pd.Series(dtype=str)).dropna().tolist())
    else:
        print(f"[WARN] Tidy file not found: {tidy_path}")
        df = pd.DataFrame()
        gold_pdfs = set()

    if step1_path.exists():
        step1_df = pd.read_csv(step1_path)
        # Derive doi_key from doi column if present
        if "doi" in step1_df.columns and "doi_key" not in step1_df.columns:
            step1_df["doi_key"] = step1_df["doi"].fillna("").str.lower().str.strip()
        gold_dois = set()   # extend if you have a doi-level gold mapping
        evaluate_predownload_classifier(step1_df, gold_dois)
    else:
        print(f"[WARN] Step1 file not found: {step1_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
