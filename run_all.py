"""
run_all.py
==========
Orchestration script for the perovskite ink formulation extraction pipeline.
Runs all pipeline steps in sequence from OpenAlex search through to TIDY export.

Usage
-----
    python scripts/run_all.py [--skip-download] [--skip-llm] [--top-n 150]

Arguments
---------
  --skip-download : Skip step 3b (PDF download). Use this if PDFs are already
                    present in the pdfs/ directory from a previous run.
  --skip-llm      : Skip step 4c-LLM (LLM extraction). Requires ANTHROPIC_API_KEY
                    to be set in the environment; skip during testing.
  --top-n N       : Number of candidates to select (default: 150).

Pipeline stages executed
------------------------
  Step 1  : step1_search_openalex.py       — Query OpenAlex API (959 records)
  Step 2  : step2_pick_candidates.py       — Score and filter candidates (150)
  Step 3a : step3a_merge_pdf_urls.py       — Resolve PDF URLs
  Step 3b : step3b_download_pdfs_smart.py  — Download PDFs (113 of 150)
  Step 4c : step4c_refine_best_recipe.py   — Regex extraction of recipe blocks
  Step 4c-LLM: step4c_llm_enhance.py      — LLM-assisted structured extraction
  Step 4d : step4d_make_final_csv.py       — Clean and validate numeric fields
  Step 4e : step4e_filter_perovskite_only.py — Filter to perovskite records (71)
  Step 9b : step9b_filter_real_ink_recipes.py — Filter to real formulations (59)
  Step 9c : step9c_clean_ratios.py         — Normalise solvent ratios
  Step 9g : step9g_extract_solvent_system.py — Extract solvent systems
  Step 9h : step9h_fix_solvent_ratio_mapping.py — Map ratios to solvents
  Step 9i : step9i_best_ratio_anywhere.py  — Find best ratio from any field
  Step 9j : step9j_extract_additives.py   — Extract additive chemicals
  Step 9k : step9k_export_tidy_dataset.py  — Export TIDY + GOLD datasets (72 / 15)

Expected runtime
----------------
  Steps 1–3b (with download) : ~90–120 minutes (dominated by PDF download latency)
  Steps 4c–4e (regex)        : ~5 minutes
  Step 4c-LLM                : ~6 minutes (requires ANTHROPIC_API_KEY)
  Steps 9b–9k (processing)   : ~2 minutes
  Total                      : ~2 hours (first run with download)

Requirements
------------
  pip install pandas requests pymupdf
  export ANTHROPIC_API_KEY=sk-ant-...   (only needed if --skip-llm is not set)
"""

import subprocess
import sys
import argparse
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT        = SCRIPTS_DIR.parent


def run(script: str, extra_args: list = None) -> None:
    """Run a pipeline script using the current Python interpreter.

    Args:
        script:     Filename of the script (relative to scripts/ directory).
        extra_args: Optional list of additional CLI arguments to pass.

    Raises:
        SystemExit: If the script returns a non-zero exit code.
    """
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[run_all] ERROR: {script} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def main() -> None:
    """Parse arguments and execute all pipeline stages in order."""
    ap = argparse.ArgumentParser(
        description="Run the full perovskite ink formulation extraction pipeline."
    )
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip PDF download step (use existing pdfs/ directory)")
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip LLM extraction step (requires ANTHROPIC_API_KEY)")
    ap.add_argument("--top-n", type=int, default=150,
                    help="Number of candidates to select in step 2 (default: 150)")
    args = ap.parse_args()

    print("Perovskite Ink Formulation Extraction Pipeline")
    print(f"Root directory : {ROOT}")
    print(f"Skip download  : {args.skip_download}")
    print(f"Skip LLM       : {args.skip_llm}")
    print(f"Top-N candidates: {args.top_n}")

    # ── Stage 1: Literature search ────────────────────────────────────────────
    run("step1_search_openalex.py")

    # ── Stage 2: Candidate scoring and filtering ──────────────────────────────
    run("step2_pick_candidates.py", ["--top-n", str(args.top_n)])

    # ── Stage 3a: Resolve PDF URLs ────────────────────────────────────────────
    run("step3a_merge_pdf_urls.py")

    # ── Stage 3b: Download PDFs ───────────────────────────────────────────────
    if not args.skip_download:
        run("step3b_download_pdfs_smart.py")
    else:
        print("\n[run_all] Skipping step 3b (--skip-download set)")

    # ── Stage 4c: Regex extraction ────────────────────────────────────────────
    run("step4c_refine_best_recipe.py")

    # ── Stage 4c-LLM: LLM-assisted extraction ────────────────────────────────
    if not args.skip_llm:
        run("step4c_llm_enhance.py")
    else:
        print("\n[run_all] Skipping LLM extraction (--skip-llm set)")

    # ── Stage 4d: Clean and validate numeric fields ───────────────────────────
    run("step4d_make_final_csv.py")

    # ── Stage 4e: Filter to perovskite records ────────────────────────────────
    run("step4e_filter_perovskite_only.py")

    # ── Stages 9b–9k: Enrichment and TIDY export ─────────────────────────────
    for script in [
        "step9b_filter_real_ink_recipes.py",
        "step9c_clean_ratios.py",
        "step9g_extract_solvent_system.py",
        "step9h_fix_solvent_ratio_mapping.py",
        "step9i_best_ratio_anywhere.py",
        "step9j_extract_additives.py",
        "step9k_export_tidy_dataset.py",
    ]:
        run(script)

    print("\n" + "="*60)
    print("Pipeline complete.")
    print(f"TIDY dataset : {ROOT}/perovskite_ink_dataset_TIDY.csv")
    print(f"GOLD dataset : {ROOT}/perovskite_ink_dataset_TIDY_GOLD.csv")
    print("="*60)


if __name__ == "__main__":
    main()
