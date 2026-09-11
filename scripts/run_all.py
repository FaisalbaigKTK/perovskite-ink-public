"""
run_all.py  —  Master orchestrator for the perovskite ink extraction pipeline.

Run from the repository root:
    python scripts/run_all.py
    python scripts/run_all.py --skip-download
    python scripts/run_all.py --skip-llm
    python scripts/run_all.py --skip-download --skip-llm
"""

import sys
import argparse
import subprocess
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# This file lives in scripts/; the repo root is one level up.
REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def run(script_name: str, extra_args: list = None) -> None:
    """Run a script from the scripts/ directory and stop on failure."""
    script = SCRIPTS_DIR / script_name
    if not script.exists():
        print(f"\n[run_all] ERROR: script not found: {script}")
        sys.exit(1)
    cmd = [sys.executable, str(script)] + (extra_args or [])
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n[run_all] FAILED: {script_name} exited {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full perovskite ink formulation extraction pipeline."
    )
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip PDF download (use existing pdfs/ folder from a previous run)")
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip LLM extraction step (no ANTHROPIC_API_KEY required)")
    ap.add_argument("--top-n", type=int, default=150,
                    help="Number of top candidates to retain in step 2 (default: 150)")
    args = ap.parse_args()

    print("=" * 60)
    print("Perovskite Ink Formulation Extraction Pipeline")
    print(f"Repo root     : {REPO_ROOT}")
    print(f"Skip download : {args.skip_download}")
    print(f"Skip LLM      : {args.skip_llm}")
    print(f"Top-N         : {args.top_n}")
    print("=" * 60)

    # Stage 1 — Literature search
    run("step1_search_openalex.py")

    # Stage 2 — Candidate scoring
    run("step2_pick_candidates.py", ["--top-n", str(args.top_n)])

    # Stage 3a — Resolve PDF URLs
    run("step3a_merge_pdf_urls.py")

    # Stage 3b — Download PDFs
    if not args.skip_download:
        run("step3b_download_pdfs_smart.py")
    else:
        print("\n[run_all] Skipping step 3b (--skip-download)")

    # Stage 4c — Regex extraction
    run("step4c_refine_best_recipe.py")

    # Stage 4L — LLM extraction
    if not args.skip_llm:
        run("step4L_llm_enhance.py")
    else:
        print("\n[run_all] Skipping LLM extraction (--skip-llm)")

    # Stages 9b–9k — Cleaning, filtering, enrichment, TIDY export
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
    print(f"Main dataset : {REPO_ROOT}/perovskite_ink_dataset_TIDY.csv")
    print(f"GOLD dataset : {REPO_ROOT}/perovskite_ink_dataset_TIDY_GOLD.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
