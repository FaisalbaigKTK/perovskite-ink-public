"""
run_all.py  —  Master orchestrator for the perovskite ink extraction pipeline.

Usage:
  Full run from scratch:
      python scripts/run_all.py

  Skip PDF download (PDFs already in pdfs/ folder):
      python scripts/run_all.py --skip-download

  Skip download + LLM (both already done):
      python scripts/run_all.py --skip-download --skip-llm

  Only run post-extraction processing (have step4c CSV + LLM_output.csv):
      python scripts/run_all.py --only-process
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


def skip(label: str) -> None:
    print(f"[run_all] Skipping {label}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the perovskite ink extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip PDF download — use existing pdfs/ folder")
    ap.add_argument("--skip-llm",      action="store_true",
                    help="Skip LLM extraction — use existing LLM_output.csv")
    ap.add_argument("--only-process",  action="store_true",
                    help="Only run post-extraction (pipeline.py). "
                         "Needs: step4c CSV + LLM_output.csv already present.")
    ap.add_argument("--top-n", type=int, default=150,
                    help="Top-N candidates for step2 (default: 150)")
    args = ap.parse_args()

    print("=" * 60)
    print("Perovskite Ink Formulation Extraction Pipeline")
    print(f"Repo root     : {REPO_ROOT}")
    print(f"Skip download : {args.skip_download}")
    print(f"Skip LLM      : {args.skip_llm}")
    print(f"Only process  : {args.only_process}")
    print("=" * 60)

    if not args.only_process:
        # Stage 1 — Search
        run("step1_search_openalex.py")
        run("step2_pick_candidates.py", ["--top-n", str(args.top_n)])
        run("step3a_merge_pdf_urls.py")

        # Stage 2 — Download
        if not args.skip_download:
            run("step3b_download_pdfs_smart.py")
        else:
            skip("step3b (--skip-download)")

        # Stage 3 — Regex extraction
        run("step4c_refine_best_recipe.py")

        # Stage 4 — LLM extraction
        if not args.skip_llm:
            run("step4L_llm_enhance.py")
        else:
            skip("step4L (--skip-llm)")
    else:
        skip("steps 1-4 (--only-process)")

    # Stage 5 — Post-extraction processing (single script, no path issues)
    run("pipeline.py")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"TIDY : {REPO_ROOT / 'perovskite_ink_dataset_TIDY.csv'}")
    print(f"GOLD : {REPO_ROOT / 'perovskite_ink_dataset_TIDY_GOLD.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
