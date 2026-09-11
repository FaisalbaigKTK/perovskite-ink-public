"""
run_all.py  —  Master orchestrator for the perovskite ink extraction pipeline.

Common usage patterns:
  Full run from scratch:
      python scripts/run_all.py

  Already have PDFs, skip download:
      python scripts/run_all.py --skip-download

  Already have step1 results AND PDFs, just re-run extraction + cleaning:
      python scripts/run_all.py --skip-search --skip-download

  Already have LLM output, just re-run cleaning steps:
      python scripts/run_all.py --skip-search --skip-download --skip-llm

  Only run the cleaning/export steps (9b-9k):
      python scripts/run_all.py --only-clean
"""

import sys
import argparse
import subprocess
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def run(script_name: str, extra_args: list = None) -> None:
    script = SCRIPTS_DIR / script_name
    if not script.exists():
        print(f"\n[run_all] ERROR: script not found: {script}")
        sys.exit(1)
    cmd = [sys.executable, str(script)] + (extra_args or [])
    print(f"\n{'='*60}\nRunning: {script_name}\n{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n[run_all] FAILED: {script_name} (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the perovskite ink extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("--skip-search",   action="store_true",
                    help="Skip step1 (OpenAlex search) — use existing step1_results.csv")
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip step3b (PDF download) — use existing pdfs/ folder")
    ap.add_argument("--skip-llm",      action="store_true",
                    help="Skip step4L (LLM extraction) — use existing LLM_output.csv")
    ap.add_argument("--only-clean",    action="store_true",
                    help="Only run steps 9b-9k (cleaning + TIDY export). Skips 1,2,3,4.")
    ap.add_argument("--top-n", type=int, default=150,
                    help="Top-N candidates for step2 (default: 150)")
    args = ap.parse_args()

    # --only-clean implies skipping everything before step 9
    if args.only_clean:
        args.skip_search   = True
        args.skip_download = True
        args.skip_llm      = True

    print("=" * 60)
    print("Perovskite Ink Formulation Extraction Pipeline")
    print(f"Repo root     : {REPO_ROOT}")
    print(f"Skip search   : {args.skip_search}")
    print(f"Skip download : {args.skip_download}")
    print(f"Skip LLM      : {args.skip_llm}")
    print(f"Only clean    : {args.only_clean}")
    print("=" * 60)

    # Stage 1 — Literature search
    if not args.skip_search:
        run("step1_search_openalex.py")
    else:
        print("\n[run_all] Skipping step1 (--skip-search)")

    # Stage 2 — Candidate scoring
    if not args.skip_search:
        run("step2_pick_candidates.py", ["--top-n", str(args.top_n)])
    else:
        print("[run_all] Skipping step2 (--skip-search)")

    # Stage 3a — Resolve PDF URLs
    if not args.skip_search:
        run("step3a_merge_pdf_urls.py")
    else:
        print("[run_all] Skipping step3a (--skip-search)")

    # Stage 3b — Download PDFs
    if not args.skip_download and not args.skip_search:
        run("step3b_download_pdfs_smart.py")
    else:
        print("[run_all] Skipping step3b (--skip-download)")

    # Stage 4c — Regex extraction
    if not args.only_clean:
        run("step4c_refine_best_recipe.py")
    else:
        print("[run_all] Skipping step4c (--only-clean)")

    # Stage 4L — LLM extraction
    if not args.skip_llm and not args.only_clean:
        run("step4L_llm_enhance.py")
    else:
        print("[run_all] Skipping step4L (--skip-llm or --only-clean)")

    # Stages 9b-9k — Cleaning + TIDY export (always run unless error)
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

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"TIDY : {REPO_ROOT / 'perovskite_ink_dataset_TIDY.csv'}")
    print(f"GOLD : {REPO_ROOT / 'perovskite_ink_dataset_TIDY_GOLD.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
