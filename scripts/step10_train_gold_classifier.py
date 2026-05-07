import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]

TIDY = ROOT / "data" / "05_final" / "perovskite_ink_dataset_TIDY.csv"
GOLD = ROOT / "data" / "05_final" / "perovskite_ink_dataset_TIDY_GOLD.csv"

OUT_DIR = ROOT / "data" / "06_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PROBS = OUT_DIR / "ranking input.csv"
OUT_REPORT = OUT_DIR / "gold_classifier_report.txt"


def safe_nonempty(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    return (s != "") & (s.str.lower() != "nan")


def count_solvents_from_system(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    return s.apply(lambda x: len([p for p in x.split(":") if p.strip()]) if x.strip() else 0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # Build as int (0/1) to avoid sklearn bool/imputer issues
    feat["has_formula"] = safe_nonempty(df.get("perovskite_formula", pd.Series([""] * len(df)))).astype(int)
    feat["has_system"] = safe_nonempty(df.get("solvent_system_final", pd.Series([""] * len(df)))).astype(int)
    feat["has_ratio"] = safe_nonempty(df.get("solvent_ratio_final", pd.Series([""] * len(df)))).astype(int)
    feat["has_molarity"] = safe_nonempty(df.get("molarity_M", pd.Series([""] * len(df)))).astype(int)
    feat["has_additives"] = safe_nonempty(df.get("additives_found", pd.Series([""] * len(df)))).astype(int)

    feat["solvent_count"] = count_solvents_from_system(df.get("solvent_system_final", pd.Series([""] * len(df)))).astype(float)

    # Categorical
    feat["recipe_class"] = df.get("recipe_class", pd.Series([""] * len(df))).fillna("").astype(str)
    feat["recipe_tier"] = df.get("recipe_tier", pd.Series([""] * len(df))).fillna("").astype(str)

    # Numeric
    feat["recipe_score"] = pd.to_numeric(df.get("recipe_score", pd.Series([0] * len(df))), errors="coerce").astype(float)

    return feat


def main():
    if not TIDY.exists():
        raise FileNotFoundError(f"Missing: {TIDY}")
    if not GOLD.exists():
        raise FileNotFoundError(f"Missing: {GOLD}")

    df_tidy = pd.read_csv(TIDY)
    df_gold = pd.read_csv(GOLD)

    if "pdf_file" not in df_tidy.columns:
        raise ValueError("TIDY must contain pdf_file column.")
    if df_tidy.empty:
        raise ValueError("TIDY is empty.")

    gold_set = set(df_gold["pdf_file"].fillna("").astype(str).tolist()) if "pdf_file" in df_gold.columns else set()
    df_tidy["is_gold"] = df_tidy["pdf_file"].fillna("").astype(str).apply(lambda x: 1 if x in gold_set else 0)

    X = build_features(df_tidy)
    y = df_tidy["is_gold"].astype(int)

    if y.nunique() < 2:
        # Not enough positives/negatives to train a classifier
        out = df_tidy.copy()
        out["gold_probability"] = 0.0
        out.to_csv(OUT_PROBS, index=False)
        OUT_REPORT.write_text(
            "Not enough class diversity to train. Need both GOLD and non-GOLD examples.\n",
            encoding="utf-8"
        )
        print("[ML] Not enough class diversity to train. Saved empty probs/report.")
        return

    stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    num_cols = ["solvent_count", "recipe_score", "has_formula", "has_system", "has_ratio", "has_molarity", "has_additives"]
    cat_cols = ["recipe_class", "recipe_tier"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=3000, class_weight="balanced")

    clf = Pipeline(steps=[("pre", pre), ("model", model)])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    report_lines = []
    report_lines.append("=== GOLD CLASSIFIER (Logistic Regression) ===\n")
    report_lines.append(f"Total rows: {len(df_tidy)} | GOLD positives: {int(y.sum())}\n")
    report_lines.append(classification_report(y_test, y_pred, digits=3))
    report_lines.append(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.3f}\n")

    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    df_tidy["gold_probability"] = clf.predict_proba(X)[:, 1]

    out = df_tidy[[
        "pdf_file", "perovskite_formula", "solvent_system_final", "solvent_ratio_final",
        "molarity_M", "additives_found", "recipe_class", "recipe_score", "recipe_tier",
        "gold_probability", "is_gold"
    ]].copy()

    out = out.sort_values("gold_probability", ascending=False)
    out.to_csv(OUT_PROBS, index=False)

    print(f"[ML] Saved report: {OUT_REPORT}")
    print(f"[ML] Saved probabilities: {OUT_PROBS}")
    print("\n[ML] Top 10 ranked candidates:")
    print(out[["pdf_file", "gold_probability", "is_gold", "recipe_score", "recipe_tier"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
