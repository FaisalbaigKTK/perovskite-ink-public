"""
evaluation/evaluate_extraction.py
==================================
Evaluation script for assessing the quality of LLM-extracted perovskite ink
recipes against a manually annotated gold-standard set.

Metrics computed:
  - Field-level precision / recall / F1 (per field)
  - Recipe-level completeness score
  - Overall extraction accuracy

Usage:
    python evaluate_extraction.py \
        --gold  data_sample/gold_annotations.csv \
        --pred  data_sample/extracted_recipes_sample.csv \
        --out   evaluation_report.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


SCORED_FIELDS = [
    "perovskite_formula",
    "solvent_system_final",
    "solvent_ratio_final",
    "molarity_M",
    "additives_found",
]

NUMERIC_FIELDS = {"molarity_M"}
NUMERIC_TOL = 0.05   # ±5% relative tolerance for numeric match


def normalise(val) -> str:
    """Lowercase, strip, collapse whitespace."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return " ".join(str(val).lower().strip().split())


def numeric_match(a: str, b: str, tol: float = NUMERIC_TOL) -> bool:
    try:
        fa, fb = float(a), float(b)
        if fa == 0 and fb == 0:
            return True
        return abs(fa - fb) / max(abs(fa), abs(fb)) <= tol
    except (ValueError, ZeroDivisionError):
        return False


def field_match(field: str, pred_val: str, gold_val: str) -> bool:
    if not gold_val and not pred_val:
        return True          # both empty → correct
    if not gold_val or not pred_val:
        return False         # one empty → wrong
    if field in NUMERIC_FIELDS:
        return numeric_match(pred_val, gold_val)
    return normalise(pred_val) == normalise(gold_val)


def evaluate(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """
    Merge on 'pdf_file' and compute per-field and overall metrics.
    """
    merged = gold_df.merge(pred_df, on="pdf_file", suffixes=("_gold", "_pred"), how="inner")
    n = len(merged)
    if n == 0:
        return {"error": "No matching pdf_file keys between gold and predictions."}

    results = {"n_matched": n, "fields": {}}
    tp_total = fp_total = fn_total = 0

    for field in SCORED_FIELDS:
        gc = f"{field}_gold"
        pc = f"{field}_pred"
        if gc not in merged.columns or pc not in merged.columns:
            results["fields"][field] = {"note": "column not found in one of the files"}
            continue

        tp = fp = fn = 0
        for _, row in merged.iterrows():
            g = normalise(row[gc])
            p = normalise(row[pc])
            match = field_match(field, p, g)
            if g and p:
                if match:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif g and not p:
                fn += 1
            elif p and not g:
                fp += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results["fields"][field] = {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "F1":        round(f1,   4),
        }
        tp_total += tp
        fp_total += fp
        fn_total += fn

    # Macro-average
    f1s = [v["F1"] for v in results["fields"].values() if "F1" in v]
    results["macro_F1"] = round(np.mean(f1s), 4) if f1s else 0.0

    # Recipe-level completeness (fraction of SCORED_FIELDS that are non-null in pred)
    completeness = []
    for _, row in pred_df.iterrows():
        filled = sum(1 for f in SCORED_FIELDS
                     if f in row and normalise(row[f]) != "")
        completeness.append(filled / len(SCORED_FIELDS))
    results["mean_recipe_completeness"] = round(np.mean(completeness), 4) if completeness else 0.0

    return results


def format_report(results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("PEROVSKITE INK EXTRACTION — EVALUATION REPORT")
    lines.append("=" * 60)
    if "error" in results:
        lines.append(f"ERROR: {results['error']}")
        return "\n".join(lines)

    lines.append(f"Matched records (gold ∩ pred): {results['n_matched']}")
    lines.append(f"Mean recipe completeness:      {results['mean_recipe_completeness']:.1%}")
    lines.append(f"Macro-average F1:              {results['macro_F1']:.4f}")
    lines.append("")
    lines.append(f"{'Field':<30} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    lines.append("-" * 65)
    for field, m in results["fields"].items():
        if "note" in m:
            lines.append(f"{field:<30}  {m['note']}")
        else:
            lines.append(
                f"{field:<30} {m['precision']:>7.4f} {m['recall']:>7.4f} "
                f"{m['F1']:>7.4f} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5}"
            )
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction quality vs gold standard.")
    parser.add_argument("--gold", required=True, help="Path to gold annotations CSV")
    parser.add_argument("--pred", required=True, help="Path to predicted/extracted CSV")
    parser.add_argument("--out",  default="evaluation_report.txt", help="Output report path")
    args = parser.parse_args()

    gold_df = pd.read_csv(args.gold)
    pred_df = pd.read_csv(args.pred)

    results = evaluate(gold_df, pred_df)
    report  = format_report(results)

    print(report)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
